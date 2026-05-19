
import time
from inspire_sdkpy import inspire_sdk, inspire_hand_defaut
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import LEFT_TOUCH_SENSOR_LAYOUT, RIGHT_TOUCH_SENSOR_LAYOUT
import numpy as np

def worker_hand_l_dds(ip, LR, name,  shm_name, shared_lock, network=None):
    handler = inspire_sdk.ModbusDataHandler(network=network, ip=ip, LR=LR, device_id=1)
    left_touch_shm = SharedMemoryManager(LEFT_TOUCH_SENSOR_LAYOUT, shared_lock["left_touch_lock"], shm_name["left_touch_shm"])

    call_count = 0
    start_time = time.perf_counter()
    time.sleep(0.5)

    try:
        while True:
            data_dict = handler.read()
            call_count += 1

            touch_dict = data_dict["touch"]
            # touch_dict의 키 이름은 LEFT_TOUCH_SENSOR_LAYOUT의 필드 이름과 일치해야 합니다.
            mapped = { f"l_{k}": np.array(v, dtype=np.int16) for k, v in touch_dict.items() }
            mapped["l_touch_ts"] = np.int64(time.perf_counter_ns())
            left_touch_shm.write_data(**mapped)
            time.sleep(0.001)

            time.sleep(0.001)
            
            if call_count % 10 == 0:
                elapsed_time = time.perf_counter() - start_time
                frequency = call_count / elapsed_time
    except KeyboardInterrupt:
        elapsed_time = time.perf_counter() - start_time
        frequency = call_count / elapsed_time if elapsed_time > 0 else 0

    finally:
        left_touch_shm.worker_close()

def worker_hand_r_dds(ip, LR, name, shm_name, shared_lock, network=None):
    handler = inspire_sdk.ModbusDataHandler(network=network, ip=ip, LR=LR, device_id=1)
    right_touch_shm = SharedMemoryManager(RIGHT_TOUCH_SENSOR_LAYOUT, shared_lock["right_touch_lock"], shm_name["right_touch_shm"])

    call_count = 0
    start_time = time.perf_counter()
    time.sleep(0.5)
    
    try:
        while True:
            data_dict = handler.read()
            call_count += 1

            # ── 터치 센서 매트릭스만 꺼내서 SHM에 기록 ──
            touch_dict = data_dict["touch"]
            # touch_dict의 키 이름은 RIGHT_TOUCH_SENSOR_LAYOUT의 필드 이름과 일치해야 합니다.
            mapped = { f"r_{k}": np.array(v, dtype=np.int16) for k, v in touch_dict.items() }
            mapped["r_touch_ts"] = np.int64(time.perf_counter_ns())
            right_touch_shm.write_data(**mapped)

            time.sleep(0.001)
            
            if call_count % 10 == 0:
                elapsed_time = time.perf_counter() - start_time
                frequency = call_count / elapsed_time
    except KeyboardInterrupt:
        elapsed_time = time.perf_counter() - start_time
        frequency = call_count / elapsed_time if elapsed_time > 0 else 0
    finally:
        right_touch_shm.worker_close()
