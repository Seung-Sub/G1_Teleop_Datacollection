# shm_manager.py
import numpy as np
from multiprocessing import shared_memory
import multiprocessing as mp
from multiprocessing import resource_tracker as _rt


def _disable_shm_resource_tracking():
    """bpo-38119 우회: resource_tracker 가 shared_memory 를 추적/자동 unlink 하지 않게 한다.

    여러 워커가 같은 SHM 을 attach 하면 resource_tracker 가 (1) 종료 시 owner 도 아닌
    SHM 을 자동 unlink 하여 다른 프로세스 메모리를 파괴하고, (2) 같은 이름을 중복
    unregister 하다 `KeyError` 트레이스백을 종료 시 다발로 출력한다.
    SHM 수명은 앱이 직접 관리(main.py preflight stale 정리 + main_unlink)하므로,
    트래커의 shared_memory 추적 자체를 비활성화하면 양쪽 문제가 모두 사라진다.
    모든 프로세스에서 SHM 생성/attach 이전에 1회 적용되어야 한다(이 모듈 import 시 자동).
    """
    if getattr(_rt, "_shm_tracking_disabled", False):
        return
    orig_register, orig_unregister = _rt.register, _rt.unregister

    def register(name, rtype):
        if rtype == "shared_memory":
            return
        return orig_register(name, rtype)

    def unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return orig_unregister(name, rtype)

    _rt.register = register
    _rt.unregister = unregister
    _rt._CLEANUP_FUNCS.pop("shared_memory", None)
    _rt._shm_tracking_disabled = True


_disable_shm_resource_tracking()


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
        # SHM 추적은 모듈 로드 시 _disable_shm_resource_tracking() 으로 비활성화됨
        # (owner/non-owner 모두 트래커 미등록 → 종료 시 자동 unlink·KeyError 없음).

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