
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import threading
import time
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# from sharedmemory.shmManager import SharedMemoryManager
# from sharedmemory.shm_schema import LAYOUT1

from utils.rate import Rate

class DualRateWorker:
    """20Hz와 300Hz 루프를 각각 별도 스레드에서 돌리는 기본 구조.

    Phase L6 (Part 2 P2-7) GIL 한계 명시:
        본 클래스는 같은 *프로세스* 의 두 스레드 (slow / fast) 로 do_slow / do_fast
        실행. Python GIL 때문에 두 스레드는 진짜 병렬이 아니다. 다만:
          - do_fast 의 obs 읽기는 대부분 C 확장 (numpy, DDS Read) 이라 GIL 을 자주
            놓아 실측상 300Hz 가 나올 수 있다.
          - Phase K2 (LowState recv_ts) 적용 후 do_fast 가 같은 LowState 를 중복
            기록하지 않으므로 300Hz 폴링이 LowState 실제 갱신율 (~500Hz 가정) 이하
            라면 dedup 으로 버려지는 낭비.
        실측 후 (REMAINING_VERIFICATION_TASKS.md L4) freq_shm.g1_freq 가 의도한
        300Hz 인지 확인. 안 나오면 do_fast 주기를 실제 갱신율에 맞춰 조정.
    """

    def __init__(self, slow_hz: float = 20, fast_hz: float = 300):
        # 주기 정의
        self.slow_rate = Rate(slow_hz)
        self.fast_rate = Rate(fast_hz)

        self.slow_count = 0
        self.fast_count = 0

        # 정지 플래그
        self._stop_event = threading.Event()

        # 스레드 객체 준비
        self._slow_thread = threading.Thread(
            target=self._slow_loop, name=f"{slow_hz}HzLoop", daemon=True
        )
        self._fast_thread = threading.Thread(
            target=self._fast_loop, name=f"{fast_hz}HzLoop", daemon=True
        )

    # ────────────── 퍼블릭 메서드 ──────────────
    def start(self) -> None:
        """두 루프를 시작."""
        self._slow_thread.start()
        self._fast_thread.start()

    def stop(self) -> None:
        """루프를 안전하게 종료하고 스레드를 join."""
        self._stop_event.set()
        self._slow_thread.join()
        self._fast_thread.join()

    # ────────────── 루프 내부 구현 ──────────────
    def _slow_loop(self) -> None:
        while not self._stop_event.is_set():
            self.do_slow()  

            self.slow_rate.sleep()

    def _fast_loop(self) -> None:
        while not self._stop_event.is_set():
            self.do_fast()  

            self.fast_rate.sleep()


    # ────────────── 실제 작업 (오버라이드/콜백용) ──────────────
    def do_slow(self) -> None:
        """20 Hz마다 실행되는 작업 예시."""
        self.slow_count +=1
        print(f"slow task : {self.slow_count}")

    def do_fast(self) -> None:
        """300 Hz마다 실행되는 작업 예시."""
        self.fast_count +=1
        print(f"fast task : {self.fast_count}")

# ────────────── 데모 실행 ──────────────
if __name__ == "__main__":
    worker = DualRateWorker()
    try:
        worker.start()
        time.sleep(2)       # 2초간 데모 실행
    finally:
        worker.stop()
