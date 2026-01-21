import posix_ipc
import mmap
import struct
from typing import Sequence

class KistarV2ShmCommand:
    '''
    ───── Write-Slot Layout  (Python → C++) ─────
      1. motion_status                 float32 x 1    →   4
      2. hand_status                   float32 x 1    →   4
      3. target_joint                  float32 x 16   →  64
      4. contact_condition             float16 x 4    →  16
      5. wrist_SE3 (4x4)              float32 x 16    →  64
      6. link3_set                     float32 x 4    →  16
      7. opening_direction_set (3x4)   float32 x 12   →  48
      8. finger_direction_set  (3x4)   float32 x 12   →  48
      9. contact_point_set    (3x4)    float32 x 12   →  48
      ──────────────────────────────────────────────
      subtotal                                        296
      -----------------------------------------------------
      SLOT_SIZE                                       296 bytes

    Ring-buffer header : read_pos | write_pos | count (uint64x3 = 24 B)
    TOTAL_SIZE = SLOT_SIZE x BUFFER_COUNT + 24
    '''

    BUFFER_COUNT = 10

    _SLOT_FORMAT = (
        "<"      # little endian
        "f"      # motion_status
        "f"      # hand_status
        "16f"    # target_joint
        "4f"     # contact_condition
        "16f"    # wrist_SE3
        "4f"     # link3_set
        "12f"    # opening_direction_set
        "12f"    # finger_direction_set
        "12f"    # contact_point_set
    )
    SLOT_SIZE   = struct.calcsize(_SLOT_FORMAT)         # 296
    print(SLOT_SIZE)
    BODY_SIZE   = SLOT_SIZE * BUFFER_COUNT              # 2960

    _HEADER_FMT = "<QQQ"
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)          # 24
    TOTAL_SIZE  = BODY_SIZE + HEADER_SIZE               # 2984

    # ───────────────────────── init ──────────────────────────
    def __init__(self, shm_name, sem_mutex, sem_full, sem_empty, create=False):
        flags = posix_ipc.O_CREAT if create else 0

        # create shared memory
        self.shm = posix_ipc.SharedMemory(shm_name, flags, size=self.TOTAL_SIZE)
        self.map = mmap.mmap(self.shm.fd, self.TOTAL_SIZE,
                             mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.shm.close_fd()

        # create semaphores
        self.mutex = posix_ipc.Semaphore(sem_mutex, flags, initial_value=1)
        self.full  = posix_ipc.Semaphore(sem_full,  flags, initial_value=0)
        self.empty = posix_ipc.Semaphore(sem_empty, flags, initial_value=self.BUFFER_COUNT)

        if create:
            self._init_memory()

    # ───────────────────── helpers ─────────────────────
    def _init_memory(self):
        self.map.seek(0)
        self.map.write(b'\x00' * self.TOTAL_SIZE)
        self._write_header(0, 0, 0)

    def _read_header(self):
        self.map.seek(self.BODY_SIZE)
        return struct.unpack(self._HEADER_FMT, self.map.read(self.HEADER_SIZE))

    def _write_header(self, r, w, c):
        self.map.seek(self.BODY_SIZE)
        self.map.write(struct.pack(self._HEADER_FMT, r, w, c))
        self.map.flush()

    def _write_slot(self, idx, data: bytes):
        self.map.seek(idx * self.SLOT_SIZE)
        self.map.write(data)
        self.map.flush()

    # ───────────────────── public API ──────────────────
    def write(self,
              motion_status: float,
              hand_status: float,
              joint_target: Sequence[float],             # len 16
              contact_condition: Sequence[float],        # len 4
              wrist_SE3: Sequence[float],                # len 16
              link3_set: Sequence[float],                # len 4
              opening_direction_set: Sequence[float],    # len 12
              finger_direction_set: Sequence[float],     # len 12
              contact_point_set: Sequence[float]):       # len 12

        # data size check
        assert len(joint_target)          == 16
        assert len(contact_condition)     == 4
        assert len(wrist_SE3)             == 16
        assert len(link3_set)             == 4
        assert len(opening_direction_set) == 12
        assert len(finger_direction_set)  == 12
        assert len(contact_point_set)     == 12

        packed = struct.pack(
            self._SLOT_FORMAT,
            motion_status,
            hand_status,
            *joint_target,
            *contact_condition,
            *wrist_SE3,
            *link3_set,
            *opening_direction_set,
            *finger_direction_set,
            *contact_point_set,
        )

        self.empty.acquire()
        self.mutex.acquire()

        r_pos, w_pos, cnt = self._read_header()
        self._write_slot(w_pos, packed)

        w_pos = (w_pos + 1) % self.BUFFER_COUNT
        cnt   = min(cnt + 1, self.BUFFER_COUNT)
        self._write_header(r_pos, w_pos, cnt)

        self.mutex.release()
        self.full.release()

    def get_cur_size(self) -> int: 
        return self._read_header()[2]

    def get_max_size(self) -> int: 
        return self.BUFFER_COUNT

    def close(self, unlink=False):
        if unlink:
            for s in (self.mutex, self.full, self.empty):
                try: s.unlink()
                except posix_ipc.ExistentialError: pass
        for s in (self.mutex, self.full, self.empty): s.close()

        if unlink:
            try: self.shm.unlink()
            except posix_ipc.ExistentialError: pass

        self.map.close()