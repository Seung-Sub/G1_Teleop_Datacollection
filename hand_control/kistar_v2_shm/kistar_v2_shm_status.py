import posix_ipc
import mmap
import struct
from typing import Optional


class KistarV2ShmStatus:
    '''
    ───── Slot Layout (C++ → Python) ─────
      hand_target_arrived : uint8 x 1 → 1
      padding (align-8)               → 7
      ------------------------------------
      SLOT_SIZE                        8 B

    Header : read_pos | write_pos | count  (uint64x3 = 24 B)
    TOTAL_SIZE = SLOT_SIZE x BUFFER_COUNT + 24
    '''

    BUFFER_COUNT = 10

    _SLOT_FORMAT = '<B7x'          # 1B + 7B pad
    SLOT_SIZE    = struct.calcsize(_SLOT_FORMAT)   # 8
    BODY_SIZE    = SLOT_SIZE * BUFFER_COUNT        # 80

    _HEADER_FMT  = '<QQQ'
    HEADER_SIZE  = struct.calcsize(_HEADER_FMT)    # 24
    TOTAL_SIZE   = BODY_SIZE + HEADER_SIZE         # 104

    def __init__(self,
                 shm_name: str,
                 sem_mutex: str,
                 sem_full: str,
                 sem_empty: str,
                 create: bool = False):
        flags = posix_ipc.O_CREAT if create else 0

        # shared memory
        self.shm = posix_ipc.SharedMemory(shm_name, flags, size=self.TOTAL_SIZE)
        self.map = mmap.mmap(self.shm.fd, self.TOTAL_SIZE,
                             mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.shm.close_fd()

        # semaphores
        self.mutex = posix_ipc.Semaphore(sem_mutex, flags, initial_value=1)
        self.full  = posix_ipc.Semaphore(sem_full,  flags, initial_value=0)
        self.empty = posix_ipc.Semaphore(sem_empty, flags,
                                         initial_value=self.BUFFER_COUNT)

        if create:
            self._init_memory()

    # ───────────────────── helpers ─────────────────────
    def _init_memory(self):
        self.map.seek(0)
        self.map.write(b'\x00' * self.TOTAL_SIZE)
        self._write_header(0, 0, 0)

    def _read_header(self):
        self.map.seek(self.BODY_SIZE)
        return struct.unpack(self._HEADER_FMT,
                             self.map.read(self.HEADER_SIZE))

    def _write_header(self, r, w, c):
        self.map.seek(self.BODY_SIZE)
        self.map.write(struct.pack(self._HEADER_FMT, r, w, c))
        self.map.flush()

    def _read_slot(self, idx: int) -> int:
        self.map.seek(idx * self.SLOT_SIZE)
        return struct.unpack(self._SLOT_FORMAT,
                             self.map.read(self.SLOT_SIZE))[0]

    # ───────────────────── public API ──────────────────
    def read(self) -> int:
        # lock
        self.full.acquire()
        self.mutex.acquire()

        r_pos, w_pos, cnt = self._read_header()
        flag = self._read_slot(r_pos)

        r_pos = (r_pos + 1) % self.BUFFER_COUNT
        cnt  -= 1
        self._write_header(r_pos, w_pos, cnt)

        # lock release
        self.mutex.release()
        self.empty.release()

        return flag

    def get_cur_size(self) -> int:
        return self._read_header()[2]

    def get_max_size(self) -> int:
        return self.BUFFER_COUNT

    # ───────────────────── cleanup ─────────────────────
    def close(self, unlink: bool = False):
        if unlink:
            for s in (self.mutex, self.full, self.empty):
                try:
                    s.unlink()
                except posix_ipc.ExistentialError:
                    pass
        for s in (self.mutex, self.full, self.empty):
            s.close()

        if unlink:
            try:
                self.shm.unlink()
            except posix_ipc.ExistentialError:
                pass

        self.map.close()
