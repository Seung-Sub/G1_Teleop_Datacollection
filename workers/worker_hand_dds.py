
import time
import numpy as np
from inspire_sdkpy import inspire_sdk, inspire_hand_defaut
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import LEFT_TOUCH_SENSOR_LAYOUT, RIGHT_TOUCH_SENSOR_LAYOUT


# ── DDS↔Modbus 브리지 ───────────────────────────────────────────────────────
# SDK 의 handler.read() 는 루프당 터치 17블록 + 상태 7블록(=24 Modbus-TCP 왕복)을 모두
# 읽고 상태 DDS 를 루프당 1회만 publish → 상태 도착 Hz 가 묶이고, 활성 teleop 중엔
# ctrl write 경합까지 겹쳐 ~58Hz 로 내려갔다(실측).
#
# 기록되는 손 상태는 angle_act 뿐이므로:
#   - angle_act 만 매 루프 고속으로 읽어 publish (= robot_obs hand 100Hz+),
#   - 나머지 안전/모니터 필드(status/err/temp/force/current/pos)는 라운드로빈으로 매 루프
#     1개씩만 갱신(≈루프Hz/6 ≈ 20Hz; fault/온도는 느리게 변해 충분),
#   - 터치는 tactile=on 일 때만.
# → angle Hz↑ + 루프당 Modbus 읽기 2건으로 부하↓(이전 7건). SDK 는 수정하지 않고
#   get_inspire_hand_state / state_pub / read_and_parse_registers 공개 building block 만 사용.

_ANGLE = (1546, 6, 'short')          # (addr, length, dtype) — 매 루프
_SAFETY_FIELDS = [                   # (attr, addr, length, dtype) — 라운드로빈
    ('status',      1612, 3, 'byte'),
    ('err',         1606, 3, 'byte'),
    ('temperature', 1618, 3, 'byte'),
    ('force_act',   1582, 6, 'short'),
    ('current',     1594, 6, 'short'),
    ('pos_act',     1534, 6, 'short'),
]


def _poll_all_safety(handler, msg):
    """시작 시 안전필드 전체 1회 채움 (첫 publish 부터 유효값)."""
    for attr, addr, ln, dt in _SAFETY_FIELDS:
        setattr(msg, attr, handler.read_and_parse_registers(addr, ln, dt))


def _read_touch(handler, touch_shm, prefix):
    """터치 17블록 Modbus read → touch DDS publish + touch SHM write. tactile=on 전용."""
    touch_msg = inspire_hand_defaut.get_inspire_hand_touch()
    mapped = {}
    for (name, addr, length, size, var) in handler.data:
        value = handler.read_and_parse_registers(addr, length // 2, 'short')
        if value is not None:
            setattr(touch_msg, var, value)
            mapped[f"{prefix}_{var}"] = np.array(value, dtype=np.int16).reshape(size)
    handler.pub.Write(touch_msg)
    mapped[f"{prefix}_touch_ts"] = np.int64(time.perf_counter_ns())
    touch_shm.write_data(**mapped)


def _run_bridge(handler, touch_shm, LR, tactile_on):
    a_addr, a_len, a_dt = _ANGLE
    n_safety = len(_SAFETY_FIELDS)
    msg = inspire_hand_defaut.get_inspire_hand_state()
    _poll_all_safety(handler, msg)
    i = 0
    time.sleep(0.5)
    try:
        while True:
            msg.angle_act = handler.read_and_parse_registers(a_addr, a_len, a_dt)   # 매 루프 (고속)
            attr, addr, ln, dt = _SAFETY_FIELDS[i % n_safety]                        # 라운드로빈 1개
            setattr(msg, attr, handler.read_and_parse_registers(addr, ln, dt))
            i += 1
            handler.state_pub.Write(msg)
            if tactile_on:
                _read_touch(handler, touch_shm, LR)
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        touch_shm.worker_close()


def worker_hand_l_dds(ip, LR, name, shm_name, shared_lock, tactile='off', network=None):
    handler = inspire_sdk.ModbusDataHandler(network=network, ip=ip, LR=LR, device_id=1)
    left_touch_shm = SharedMemoryManager(LEFT_TOUCH_SENSOR_LAYOUT, shared_lock["left_touch_lock"], shm_name["left_touch_shm"])
    _run_bridge(handler, left_touch_shm, LR, str(tactile).lower() == 'on')


def worker_hand_r_dds(ip, LR, name, shm_name, shared_lock, tactile='off', network=None):
    handler = inspire_sdk.ModbusDataHandler(network=network, ip=ip, LR=LR, device_id=1)
    right_touch_shm = SharedMemoryManager(RIGHT_TOUCH_SENSOR_LAYOUT, shared_lock["right_touch_lock"], shm_name["right_touch_shm"])
    _run_bridge(handler, right_touch_shm, LR, str(tactile).lower() == 'on')
