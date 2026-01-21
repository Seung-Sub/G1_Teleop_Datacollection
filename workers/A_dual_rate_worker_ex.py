# g1_ctrl_worker_split.py
import os
import time
import threading
import numpy as np

from workers.A_dual_rate_worker import DualRateWorker  # 기존 DualRateWorker 사용

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    WORKER_FREQ, RECORD_MODE_LAYOUT, ROBOT_ACTION, ROBOT_OBS, ROBOT_AMO_OBS
)

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

OBS_HZ = 300.0
ACT_HZ = 50.0  # 느린 루프 주파수로 사용

class ExampleWorker(DualRateWorker):

    def __init__(self, shared_event, shm_name, shared_lock, mode):
        super().__init__(slow_hz=ACT_HZ, fast_hz=OBS_HZ)

        # 외부 이벤트
        self._shared_event = shared_event
        self.mode = mode

        # ── SHM ───────────────────────────────────────────
        self.freq_shm         = SharedMemoryManager(WORKER_FREQ,        shared_lock["freq_lock"],         shm_name["freq_shm"])

        logger_mp.info("[ExampleWorker] start")


    def do_slow(self) -> None:
        pass


    def do_fast(self) -> None:
        pass

    # 리소스 정리
    def stop(self) -> None:
        super().stop()
        try:
            self.freq_shm.worker_close()

        finally:
            logger_mp.info("[ExampleWorker] 종료 및 SHM 정리 완료")


# ────────────── 실행 진입점 예시 ──────────────
def worker_example(shared_event, shm_name, shared_lock,mode):
    w = ExampleWorker(shared_event, shm_name, shared_lock, mode)
    try:
        w.start()
        while not shared_event['shutdown'].is_set():
            time.sleep(0.1)
    finally:
        w.stop()
