import os
import time
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# from sharedmemory.shmManager import SharedMemoryManager
# from sharedmemory.shm_schema import LAYOUT1

from utils.rate import Rate

def worker_250hz(shared_event, shm_name, shared_lock):



    # mgr = SharedMemoryManager(LAYOUT1, shared_lock["robot_lock"], shm_name["robot_shm"])
    freq = 250.0
    period_ns = int(1e9 / freq)
    next_time = time.perf_counter_ns()
    last_time = next_time
    count = 0
    rate = Rate(50.0)

    while not shared_event['shutdown'].is_set():

        print(rate.tick_hz())
        # logger_mp.info(f"[250Hz] #{count} Δ{delta_ms:.3f} ms, {actual_hz:.1f} Hz")

        # 데이터 읽기
        # data = mgr.read_data()

        rate.sleep()

    logger_mp.info("[250Hz] 종료 신호 수신. 종료합니다.")
    # mgr.close()
