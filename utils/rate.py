import time

class Rate:
    """sleep + 마지막 미세 스핀으로 50Hz 등 주기 맞추기"""
    def __init__(self, hz: float):
        self.period_ns = int(1e9 / hz)
        self.next_t = time.perf_counter_ns()
        self._last_tick = None

    def sleep(self):
        self.next_t += self.period_ns
        while True:
            now = time.perf_counter_ns()
            rem = self.next_t - now
            if rem <= 0:
                # 오버런 시 드리프트 복구
                self.next_t = now
                return
            if rem > 1_000_000:  # 1ms 이상 남으면 거칠게 슬립
                # 0.5ms 버퍼 남기고 슬립
                time.sleep((rem - 500_000) / 1e9)
            else:
                # <~1ms 구간은 바쁜 대기(미세 조정)
                pass

    def tick_hz(self) -> float:
        now = time.perf_counter_ns()
        if self._last_tick is None:
            self._last_tick = now
            return 0.0
        dt = now - self._last_tick
        self._last_tick = now
        return 1e9 / dt if dt > 0 else 0.0
