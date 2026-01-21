from kistar_v2_shm_cmd import KistarV2ShmCommand
from kistar_v2_shm_status import KistarV2ShmStatus
from typing import List
import time

if __name__ == "__main__":
    # defined shared memory
    kistar_shm_status = KistarV2ShmStatus("/memory_status", "/mutex_status", "/full_status", "/empty_status", create=True)
    kistar_shm_cmd = KistarV2ShmCommand("/memory_cmd", "/mutex_cmd", "/full_cmd", "/empty_cmd", create=True)

    # write test data
    motion_status: int = 30
    contact_condition: List[float] = [1, 2, 3, 4]
    wrist_SE3: List[float] = [6., 3., 1., 2., 
                              1., 4., 1., 2., 
                              1., 4., 1., 2., 
                              9., 2., 1., 2.]
    link3_set: List[float] = [6., 3., 1., 2.]
    opening_direction_set: List[float] = [6., 3., 1., 2., 
                                          1., 4., 1., 2., 
                                          9., 2., 1., 2.]
    finger_direction_set: List[float] = [0., 3., 1., 2., 
                                         4., 3., 1., 2., 
                                         1., 3., 1., 2.]
    contact_point_set: List[float] = [6., 7., 1., 2., 
                                      0., 4., 2., 2., 
                                      8., 2., 1., 6.]

    try:
        while(True):
            # write shared memory command
            if kistar_shm_cmd.get_cur_size() < kistar_shm_cmd.get_max_size():
                kistar_shm_cmd.write(motion_status,
                                    contact_condition,
                                    wrist_SE3,
                                    link3_set,
                                    opening_direction_set,
                                    finger_direction_set,
                                    contact_point_set)
                motion_status += 1
                if motion_status > 255:
                    motion_status = 0
                print(motion_status)

            # read shared memory command
            if kistar_shm_status.get_cur_size() > 0:
                hand_target_arrived = kistar_shm_status.read()
                # print(f"read_data! : {hand_target_arrived}")

            time.sleep(0.01) # 10hz read write rate
    except KeyboardInterrupt:
        print("close shared memory")
        kistar_shm_status.close(unlink=True)
        kistar_shm_cmd.close(unlink=True)