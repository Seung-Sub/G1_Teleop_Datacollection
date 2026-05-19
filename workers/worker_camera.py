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


def worker_camera(shared_event, shm_name, shared_lock):
    """
    RealSense 카메라를 60Hz로 읽어오는 워커 프로세스.
    shared_event: {'shutdown', 'emergency', ...} 딕셔너리
    shm_name: 공유 메모리 이름
    lock: SharedMemory 접근용 Lock
    """


    # 3) SharedMemoryManager 초기화 (필요하다면 사용)
    camera_shm = SharedMemoryManager(CAMERA, shared_lock["camera_lock"], shm_name["camera_shm"])
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])

    rate = Rate(30.0)

    # 4) RealSense pipeline 설정
    pipeline = rs.pipeline()
    config = rs.config()

    # 60fps 해상도 설정 (예: 640x480 @ 60Hz Depth + Color)
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

                color_image = np.asanyarray(color_frame.get_data())
                camera_shm.write_data(
                    realsense=color_image,
                    camera_realsense_ts=np.int64(time.perf_counter_ns()),
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
