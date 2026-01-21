import os
import time
import numpy as np

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA, TELEVISION, RECORD_MODE_LAYOUT, WORKER_FREQ

from open_television.tv_wrapper import TeleVisionWrapper

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# TELEVISION = [
#     ("head_rmat",        (4, 4),    np.float64),
#     ("left_wrist_mat",   (4, 4),    np.float64),
#     ("right_wrist_mat",  (4, 4),    np.float64),
#     ("left_hand",        (5, 3),    np.float64),
#     ("right_hand",       (5, 3),    np.float64),
# ]

def worker_vr(shared_event, shm_name, shared_lock):


    camera_shm = SharedMemoryManager(CAMERA, shared_lock["camera_lock"], shm_name["camera_shm"])
    television_shm = SharedMemoryManager(TELEVISION, shared_lock["television_lock"], shm_name["television_shm"])
    record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_name["record_mode_shm"])
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])

    freq = 50.0
    period_ns = int(1e9 / freq)
    next_time = time.perf_counter_ns()
    last_time = next_time
    count = 0
    
    img_shape = (640, 480, 3)
    tv_wrapper = TeleVisionWrapper(True, img_shape, shm_name["camera_shm"])

    home_left_wrist = None
    home_right_wrist = None



    while not shared_event['shutdown'].is_set():
        now = time.perf_counter_ns()
        delta_ms = (now - last_time) / 1e6
        actual_hz = 1.0 / ((now - last_time) / 1e9)

        # freq_data = freq_shm.read_data()
        # freq_data["vr_freq"] = actual_hz
        freq_shm.write_data(vr_freq = actual_hz)
        # 데이터 읽기

        head_rmat, left_wrist, right_wrist, left_hand, right_hand, right_distal, right_proximal = tv_wrapper.get_data_with_segments()

        mode_data = record_mode_shm.read_data()
        home = mode_data["home"]
        if home :
            # 홈 상태일 때 처음 한 번만 기준 포즈 저장
            home_left_wrist = left_wrist.copy()
            home_right_wrist = right_wrist.copy()
        
        if home_left_wrist is not None and home_right_wrist is not None:
            # 회전 행렬은 원래 그대로 사용
            rel_left_wrist = left_wrist.copy()
            rel_right_wrist = right_wrist.copy()

            # translation 차이 계산
            rel_left_wrist[:3, 3] = left_wrist[:3, 3] - home_left_wrist[:3, 3]
            rel_right_wrist[:3, 3] = right_wrist[:3, 3] - home_right_wrist[:3, 3]
        else:
            rel_left_wrist = left_wrist
            rel_right_wrist = right_wrist

        
        television_shm.write_data(
            head_rmat=head_rmat,
            left_wrist_mat=rel_left_wrist,
            right_wrist_mat=rel_right_wrist,
            left_hand=left_hand,
            right_hand=right_hand,
            right_distal=right_distal,
            right_proximal=right_proximal,
        )

        # print(f"head_rmat : {head_rmat}")
        # print(f"left_wrist_mat : {rel_left_wrist}")
        # print(f"right_wrist_mat : {rel_right_wrist}")
        # print(f"left_hand : {left_hand}")
        # print(f"right_hand : {right_hand}")


        last_time = now
        count += 1

        next_time += period_ns
        sleep_ns = next_time - time.perf_counter_ns()
        if sleep_ns > 0:
            if sleep_ns > 1_000_000:
                time.sleep((sleep_ns - 500_000) / 1e9)
            while time.perf_counter_ns() < next_time:
                pass
        else:
            next_time = time.perf_counter_ns()

    logger_mp.info("[VR] 종료 신호 수신. 종료합니다.")
    camera_shm.worker_close()
    television_shm.worker_close()
    record_mode_shm.worker_close()
    freq_shm.worker_close()


if __name__ == "__main__":
    import multiprocessing as mp
    import signal
    import sys
    import time

    # 플랫폼/라이브러리 충돌 최소화를 위해 spawn 권장 (이미 설정돼 있으면 무시)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    # Windows 빌드 호환
    mp.freeze_support()

    # --- 공유 객체 구성 ---
    shutdown_event = mp.Event()
    shared_event = {"shutdown": shutdown_event}

    shared_lock = {
        "camera_lock": mp.Lock(),
        "television_lock": mp.Lock(),
        "record_lock": mp.Lock(),
        "freq_lock": mp.Lock(),
    }

    shm_name = {
        "camera_shm": "camera_shm",
        "television_shm": "television_shm",
        "record_mode_shm": "record_mode_shm",
        "freq_shm": "freq_shm",
    }

    # --- 시그널 핸들러 (Ctrl+C, SIGTERM) ---
    def _handle_signal(signum, frame):
        try:
            logger_mp.info(f"[MAIN] Received signal {signum}. Requesting shutdown...")
        except Exception:
            pass
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            # 일부 플랫폼에서 SIGTERM 미지원 가능
            pass

    # --- 워커 프로세스 시작 ---
    proc = mp.Process(
        target=worker_vr,
        args=(shared_event, shm_name, shared_lock),
        name="VRWorker",
        daemon=False,  # 정리 루틴을 위해 데몬 비활성 권장
    )
    proc.start()

    # --- 메인 루프: 자식 종료 대기 및 그레이스풀 셧다운 ---
    try:
        while proc.is_alive():
            proc.join(timeout=0.5)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        shutdown_event.set()
        proc.join(timeout=5.0)
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            proc.join(timeout=2.0)

    sys.exit(0)
