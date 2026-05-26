# shm_manager.py
import numpy as np
from multiprocessing import shared_memory
import multiprocessing as mp

class SharedMemoryManager:
    def __init__(self, schema, lock: mp.Lock, name: str):
        """
        schema: [(field_name, shape, dtype), ...]
        - 서로 다른 dtype 혼용 지원
        """
        self.lock = lock
        self.name = name
        self._parse_schema(schema)           # 필드별 byte offset 계산
        self._create_or_connect()
        # 전체 버퍼는 "바이트 배열"로 보관
        self._buf = np.ndarray((self.total_bytes,), dtype=np.uint8, buffer=self.shm.buf)
        if self._owner:
            with self.lock:
                self._buf[:] = 0

    def _parse_schema(self, schema):
        self._fields = {}
        offset_bytes = 0
        for fname, shape, dt in schema:
            dt = np.dtype(dt)
            count = int(np.prod(shape))
            nbytes = count * dt.itemsize
            self._fields[fname] = {
                "shape": tuple(shape),
                "dtype": dt,
                "offset_bytes": offset_bytes,
                "nbytes": nbytes,
                "count": count,
            }
            offset_bytes += nbytes
        self.total_bytes = offset_bytes       # 전체 공유 메모리 크기(바이트)

    def _create_or_connect(self):
        try:
            self.shm = shared_memory.SharedMemory(name=self.name)
            self._owner = False
        except FileNotFoundError:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.total_bytes)
                self._owner = True
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(name=self.name)
                self._owner = False

        if not self._owner:
            # bpo-38119 우회: attach-only(non-owner) 프로세스가 종료될 때
            # resource_tracker 가 owner 도 아닌 SHM 을 자동 unlink 하여 다른
            # 프로세스(예: main.py)의 SHM 을 파괴하는 것을 방지한다. owner 만
            # 추적/정리(main_unlink)하도록 non-owner 는 tracker 등록을 해제.
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(self.shm._name, "shared_memory")
            except Exception:
                pass

    def write_data(self, **kwargs):
        with self.lock:
            # 바이트 단위로 각 필드만 덮어쓰기
            for k, v in kwargs.items():
                meta = self._fields[k]
                arr = np.asarray(v, dtype=meta["dtype"])
                if arr.size != meta["count"]:
                    raise ValueError(f"{k}: size mismatch. got {arr.size}, expected {meta['count']}")
                # 연속 메모리 보장 후 바이트 뷰로 변환
                arr_c = np.ascontiguousarray(arr)
                bytes_view = arr_c.view(np.uint8).reshape(-1)
                s = meta["offset_bytes"]
                e = s + meta["nbytes"]
                self._buf[s:e] = bytes_view

    def read_data(self):
        with self.lock:
            # 필요한 필드만 dtype에 맞게 복원해서 반환 (copy로 독립화)
            out = {}
            for k, meta in self._fields.items():
                s = meta["offset_bytes"]
                e = s + meta["nbytes"]
                bytes_view = self._buf[s:e]
                arr = np.frombuffer(bytes_view, dtype=meta["dtype"], count=meta["count"]).reshape(meta["shape"]).copy()
                out[k] = arr
            return out

    def main_unlink(self):
        self.shm.close()
        self.shm.unlink()

    def worker_close(self):
        self.shm.close()