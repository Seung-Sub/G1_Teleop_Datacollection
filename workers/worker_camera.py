# workers/worker_Realsense.py

import os
import time

import numpy as np
import pyrealsense2 as rs

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA, WORKER_FREQ

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

from utils.rate import Rate


def worker_camera(shared_event, shm_name, shared_lock, serial=None):
    """RealSense color@30Hz 워커.

    Args:
        serial: RealSense device serial (str). None 이면 첫 번째 device.
                D435i / D455 / D405 등 모두 동일 pipeline API.

    Timestamp 전략 (Phase K1, P0-1.1):
      RealSense Global Time 활성 → frame.get_timestamp() (host epoch ms) 사용.
      epoch ↔ perf_counter_ns() 1차 오프셋 워커 시작 시 1회 측정 후 변환.
      Global Time 미가용 (SYSTEM_TIME 또는 HARDWARE_CLOCK) 시 perf_counter_ns()
      fallback (이전 동작) + 경고 로그.
    """

    # 3) SharedMemoryManager 초기화 (필요하다면 사용)
    camera_shm = SharedMemoryManager(CAMERA, shared_lock["camera_lock"], shm_name["camera_shm"])
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])

    rate = Rate(30.0)

    # 4) RealSense pipeline 설정
    pipeline = rs.pipeline()
    config = rs.config()
    if serial is not None:
        try:
            config.enable_device(str(serial))
            logger_mp.info(f"[Realsense] enable_device(serial={serial})")
        except Exception as e:
            logger_mp.warning(f"[Realsense] enable_device({serial}) 실패: {e} — fallback any device")

    # 640x480 @ 30 Hz color (BGR8). depth 는 필요 시 enable.
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

    # 파이프라인 시작
    try:
        profile = pipeline.start(config)
        logger_mp.info("[Realsense] RealSense pipeline started.")
    except Exception as e:
        logger_mp.warning(f"[Realsense] Failed to start RealSense pipeline: {e}")
        logger_mp.info("[Realsense] Continuing without RealSense camera...")
        return

    # ---- Global Time 활성 + epoch→monotonic 오프셋 (Phase K1) -------------
    use_global_time = False
    try:
        dev = profile.get_device()
        for s in dev.query_sensors():
            try:
                if s.supports(rs.option.global_time_enabled):
                    s.set_option(rs.option.global_time_enabled, 1)
                    use_global_time = True
            except Exception:
                pass
        if use_global_time:
            logger_mp.info("[Realsense] global_time_enabled set on all supported sensors.")
        else:
            logger_mp.warning("[Realsense] global_time_enabled not supported — fallback to perf_counter_ns at write.")
    except Exception as e:
        logger_mp.warning(f"[Realsense] global_time setup 실패: {e}")

    # 두 클럭의 기준점을 거의 동시에 측정 (drift 무시 1차 근사; 에피소드 단위 충분).
    # RealSense Global Time = host epoch ns. 우리 공통 축 = perf_counter_ns().
    _sys0  = time.time_ns()
    _mono0 = time.perf_counter_ns()

    def epoch_ns_to_mono(epoch_ns: int) -> int:
        return int(epoch_ns - _sys0 + _mono0)

    domain_logged = False

    # 5) 주기 루프: 약 60Hz(프레임이 들어오는 속도)로 프레임을 읽어들임
    try:
        missed_count = 0
        max_missed = 5
        last_time = time.perf_counter_ns()  # 최초 타임스탬프

        while not shared_event['shutdown'].is_set() and not shared_event['emergency'].is_set():

            freq_shm.write_data(camera_freq =rate.tick_hz())


            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    raise RuntimeError("Frame timeout")

                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("No color frame")

                # 첫 프레임에서 timestamp domain 확인 — GLOBAL_TIME 이 아니면 경고.
                if not domain_logged:
                    try:
                        dom = color_frame.get_frame_timestamp_domain()
                        if dom == rs.timestamp_domain.global_time:
                            logger_mp.info("[Realsense] timestamp_domain = GLOBAL_TIME (epoch ms)")
                        else:
                            logger_mp.warning(
                                f"[Realsense] timestamp_domain = {dom} — Global Time 미가용, "
                                f"perf_counter_ns fallback 사용."
                            )
                    except Exception as e:
                        logger_mp.warning(f"[Realsense] get_frame_timestamp_domain 실패: {e}")
                    domain_logged = True

                # 캡처 시각 (Global Time 가용 시 epoch → mono 변환, 아니면 perf_counter_ns)
                if use_global_time:
                    try:
                        cap_epoch_ns = int(color_frame.get_timestamp() * 1e6)  # ms → ns
                        cap_mono_ns  = epoch_ns_to_mono(cap_epoch_ns)
                    except Exception:
                        cap_mono_ns = time.perf_counter_ns()
                else:
                    cap_mono_ns = time.perf_counter_ns()

                color_image = np.asanyarray(color_frame.get_data())
                camera_shm.write_data(
                    realsense=color_image,
                    camera_realsense_ts=np.int64(cap_mono_ns),
                )

                missed_count = 0  # 정상 수신 시 리셋

            except Exception as e:
                missed_count += 1
                logger_mp.warning(f"[Realsense] Missed frame ({missed_count}/{max_missed}): {e}")
                if missed_count >= max_missed:
                    logger_mp.error("[Realsense] Too many missed frames. Triggering emergency.")
                    shared_event['emergency'].set()
                    break

                # 조금 기다리고 다시 시도 (선택)
                time.sleep(0.05)

            rate.sleep()

    except Exception as e:
        logger_mp.error(f"[Realsense] Unexpected error in capture loop: {e}")
        # shared_event['emergency'].set()

    finally:
        # 파이프라인을 안전하게 중지
        try:
            pipeline.stop()
            logger_mp.info("[Realsense] RealSense pipeline stopped.")
        except Exception:
            pass

        # SHM close
        camera_shm.worker_close()
        freq_shm.worker_close()
        logger_mp.info("[Realsense] Worker exiting cleanly.")
