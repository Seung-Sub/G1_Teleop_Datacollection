"""RawStreamBuffer: thread-safe collector for timestamped stream samples.

worker_record 가 episode 동안 SHM 별로 인스턴스를 만들고, 백그라운드 poller
thread 가 SHM 을 polling 하여 (ts, payload_dict) 를 dedup 후 append.

Episode close 시점에 dump() 로 (ts_arr, payload_list) 를 한 번에 받아
utils.align.interp_to_axis 로 공통 시간축에 정렬 보간한다.

Timestamps are `time.perf_counter_ns()` (monotonic ns). 0 은 "아직 publish
되지 않음" 을 의미하므로 dedup 단계에서 거른다.
"""
from __future__ import annotations
import threading
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

import numpy as np


class RawStreamBuffer:
    def __init__(self, name: str, maxlen: int = 200_000):
        self.name = name
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._last_ts: Optional[int] = None

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._last_ts = None

    def append(self, ts: int, payload: Dict[str, Any]) -> bool:
        """Append (ts, payload).

        Skips if ts <= last_ts (dedup) or ts == 0 (uninitialised SHM).
        Returns True if appended.
        """
        ts_int = int(ts)
        if ts_int <= 0:
            return False
        with self._lock:
            if self._last_ts is not None and ts_int <= self._last_ts:
                return False
            self._buf.append((ts_int, payload))
            self._last_ts = ts_int
            return True

    def dump(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Drain buffer.  Returns (ts_arr int64, list-of-payload-dicts)
        sorted by ts ascending.  Buffer is reset after dump."""
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            self._last_ts = None
        if not items:
            return np.empty(0, dtype=np.int64), []
        ts_arr   = np.array([t for t, _ in items], dtype=np.int64)
        payloads = [p for _, p in items]
        return ts_arr, payloads
