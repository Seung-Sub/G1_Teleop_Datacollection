# workers/worker_camera.py — RealSense 1-camera / 1-SHM worker (Phase K7-A)
#
# 한 인스턴스 = 한 RealSense device. main.py 가 yaml/CLI 로 role↔serial 매핑을
# 만들어 N 인스턴스 spawn (예: ego + wrist_l + wrist_r). 각 인스턴스는 자기
# role 의 CAMERA_VIEW SHM 만 owner-attach 한다.

import os
import time

import numpy as np
import pyrealsense2 as rs

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA_VIEW, WORKER_FREQ

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

from utils.rate import Rate


def worker_camera(shared_event, shm_name, shared_lock,
                  serial=None, role='ego', shm_key=None, lock_key=None):
    """RealSense color@30Hz 워커 (단일 device).

    Args:
        serial:   RealSense device serial (str). None 이면 첫 번째 device.
                  D435i / D455 / D405 등 모두 동일 pipeline API.
        role:     'ego' / 'wrist_l' / 'wrist_r' 등. log + frame TS 보고용.
        shm_key:  본 워커가 attach 할 CAMERA_VIEW SHM 의 key (예: 'rs_ego_shm').
                  None 이면 'camera_shm' (단일 카메라 / backward-compat).
        lock_key: shared_lock dict 의 lock 키 (예: 'rs_ego_lock').
                  None 이면 'camera_lock'.

    Timestamp 전략 (Phase K1):
        RealSense Global Time 활성 → frame.get_timestamp() (host epoch ms) 사용.
        epoch↔perf_counter_ns 1차 오프셋 워커 시작 시 1회 측정 후 변환.
        Global Time 미가용 (SYSTEM_TIME 또는 HARDWARE_CLOCK) 시 perf_counter_ns()
        fallback + 경고.
    """
    if shm_key  is None: shm_key  = 'camera_shm'
    if lock_key is None: lock_key = 'camera_lock'

    # SHM owner-attach (main.py 가 이미 owner-create). CAMERA_VIEW schema 통일.
    view_shm = SharedMemoryManager(CAMERA_VIEW, shared_lock[lock_key], shm_name[shm_key])
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])

    # Rate 는 *측정 전용* (tick_hz). 캡처 속도는 wait_for_frames 가 카메라 fps(60)에
    # 맞춰 블로킹하므로 rate.sleep() 으로 추가 제한하지 않는다. (과거 Rate(30)+sleep 은
    # 60fps 로 열어도 30Hz 로 묶는 이중 제한 버그였음.) RealSense 권장: 프레임을 device
    # fps 만큼 빠르게 빼가야 드롭이 없다 → sleep 제거.
    rate = Rate(60.0)

    # RealSense pipeline 설정
    pipeline = rs.pipeline()
    config = rs.config()
    if serial is not None:
        try:
            config.enable_device(str(serial))
            logger_mp.info(f"[Realsense:{role}] enable_device(serial={serial})")
        except Exception as e:
            logger_mp.warning(f"[Realsense:{role}] enable_device({serial}) 실패: {e} — fallback any device")

    # 640x360 @ 60 Hz color (BGR8). 640x360 = 16:9 (센서 native 종횡비), 3대 동시
    # 60fps 드롭 0% 실측 검증됨 (check_camera_hw.py). depth 는 enable 안 함 (color-only,
    # 대역폭/부하 절감 — 실시간 depth 불요).
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)

    try:
        profile = pipeline.start(config)
        logger_mp.info(f"[Realsense:{role}] pipeline started.")
    except Exception as e:
        logger_mp.warning(f"[Realsense:{role}] pipeline start 실패: {e}")
        logger_mp.info(f"[Realsense:{role}] Continuing without this camera...")
        return

    # USB descriptor 로깅 — 3대를 한 머신에 물릴 때 USB2 로 떨어지면 경합 위험.
    try:
        dev = profile.get_device()
        usb_str = dev.get_info(rs.camera_info.usb_type_descriptor)
        logger_mp.info(f"[Realsense:{role}] USB type descriptor = {usb_str}")
        if usb_str.startswith('2'):
            logger_mp.warning(
                f"[Realsense:{role}] USB2.x detected — 3대 동시 grab 시 대역폭 부족 위험. "
                f"USB3 컨트롤러/허브 분산 권장."
            )
    except Exception as e:
        logger_mp.debug(f"[Realsense:{role}] usb_type_descriptor 조회 실패: {e}")

    # Global Time 활성 (Phase K1)
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
            logger_mp.info(f"[Realsense:{role}] global_time_enabled set.")
        else:
            logger_mp.warning(f"[Realsense:{role}] global_time_enabled 미지원 — perf_counter_ns fallback.")
    except Exception as e:
        logger_mp.warning(f"[Realsense:{role}] global_time setup 실패: {e}")

    _sys0  = time.time_ns()
    _mono0 = time.perf_counter_ns()
    def epoch_ns_to_mono(epoch_ns: int) -> int:
        return int(epoch_ns - _sys0 + _mono0)

    domain_logged = False

    try:
        missed_count = 0
        max_missed = 5

        while not shared_event['shutdown'].is_set() and not shared_event['emergency'].is_set():
            # role 별 freq 를 freq_shm 의 camera_freq 에 그대로 쓰면 마지막 writer 가
            # 덮어쓴다. ego 우선 (가장 마지막에 spawn 된 워커가 보고 — 안정성용 임시 결정).
            # 정확한 멀티-cam freq monitoring 은 후속 항목 (freq_shm schema 확장 필요).
            freq_shm.write_data(camera_freq=rate.tick_hz())

            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    raise RuntimeError("Frame timeout")

                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("No color frame")

                if not domain_logged:
                    try:
                        dom = color_frame.get_frame_timestamp_domain()
                        if dom == rs.timestamp_domain.global_time:
                            logger_mp.info(f"[Realsense:{role}] timestamp_domain = GLOBAL_TIME")
                        else:
                            logger_mp.warning(
                                f"[Realsense:{role}] timestamp_domain = {dom} — "
                                f"Global Time 미가용, perf_counter_ns fallback."
                            )
                    except Exception as e:
                        logger_mp.warning(f"[Realsense:{role}] get_frame_timestamp_domain 실패: {e}")
                    domain_logged = True

                if use_global_time:
                    try:
                        cap_epoch_ns = int(color_frame.get_timestamp() * 1e6)
                        cap_mono_ns  = epoch_ns_to_mono(cap_epoch_ns)
                    except Exception:
                        cap_mono_ns = time.perf_counter_ns()
                else:
                    cap_mono_ns = time.perf_counter_ns()

                color_image = np.asanyarray(color_frame.get_data())
                view_shm.write_data(
                    frame_left =color_image,
                    frame_ts   =np.int64(cap_mono_ns),
                    is_stereo  =np.int8(0),  # RealSense 는 mono (frame_right 미사용)
                )

                missed_count = 0

            except Exception as e:
                missed_count += 1
                logger_mp.warning(f"[Realsense:{role}] Missed frame ({missed_count}/{max_missed}): {e}")
                if missed_count >= max_missed:
                    logger_mp.error(f"[Realsense:{role}] Too many missed frames. Triggering emergency.")
                    shared_event['emergency'].set()
                    break
                time.sleep(0.05)

            # rate.sleep() 제거: wait_for_frames(60fps 블로킹)가 속도를 정한다.
            # sleep 을 추가하면 프레임을 제때 못 빼가 드롭이 생긴다 (RealSense 권장).

    except Exception as e:
        logger_mp.error(f"[Realsense:{role}] Unexpected error: {e}")

    finally:
        try:
            pipeline.stop()
            logger_mp.info(f"[Realsense:{role}] pipeline stopped.")
        except Exception:
            pass
        view_shm.worker_close()
        freq_shm.worker_close()
        logger_mp.info(f"[Realsense:{role}] Worker exiting cleanly.")
