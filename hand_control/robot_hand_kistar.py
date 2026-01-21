from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from inspire_sdkpy import inspire_hand_defaut, inspire_dds
from hand_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, shared_memory, Array, Lock
import traceback

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import RECORD_MODE_LAYOUT

from kistar_v2_shm.kistar_v2_shm_cmd import KistarV2ShmCommand
from kistar_v2_shm.kistar_v2_shm_status import KistarV2ShmStatus

from utils.rate import Rate


import logging_mp
logger_mp = logging_mp.get_logger(__name__)

kistar_tip_indices = [4, 9, 14, 19, 24]
Kistar_Num_Motors = 16

OBS_HZ = 300.0
ACTION_HZ = 100.0

class Kistar_Controller:
    def __init__(self, shm_name, shared_lock,  right_hand_array, hand_data_lock = None, out_hand_state_array = None, \
                 out_hand_action_array = None ):
        logger_mp.info("Initialize Kistar_Controller...")


        self.obs_rate = Rate(OBS_HZ)
        self.action_rate = Rate(ACTION_HZ)


        self.record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_name["record_mode_shm"])

        self.kistar_shm_status = KistarV2ShmStatus("/memory_status", "/mutex_status", "/full_status", "/empty_status", create=True)
        self.kistar_shm_cmd = KistarV2ShmCommand("/memory_cmd", "/mutex_cmd", "/full_cmd", "/empty_cmd", create=True)


        self.hand_retargeting = HandRetargeting(HandType.KISTAR_V2)

        self.right_hand_state_array = Array('d', Kistar_Num_Motors, lock=True)
        

        # Separate subscribe threads for each hand
        self.subscribe_Rstate_thread = threading.Thread(target=self._subscribe_hand_state, args=(self.right_hand_state_array))
  
        self.subscribe_Rstate_thread.daemon = True
        self.subscribe_Rstate_thread.start()

        hand_control_thread = threading.Thread(target=self.control_process, args=(right_hand_array,  self.right_hand_state_array,
                                                                          hand_data_lock, out_hand_state_array, out_hand_action_array))
        hand_control_thread.daemon = True
        hand_control_thread.start()


        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self, state_array):
        while True:
            if self.kistar_shm_status.get_cur_size() > 0:
                hand_target_arrived = self.kistar_shm_status.read()

            self.obs_rate.sleep()

    def ctrl_dual_hand(self, left_q_target, right_q_target):

        left_scaled  = [int(v * 1000) for v in left_q_target]
        right_scaled = [int(v * 1000) for v in right_q_target]

        # 2) 클램프(0 ~ 65535) 및 타입 검증
        def clamp_uint16(x):
            if x < 0:
                return 0
            elif x > 0xFFFF:
                return 0xFFFF
            else:
                return x

        left_clamped  = [clamp_uint16(v) for v in left_scaled]
        right_clamped = [clamp_uint16(v) for v in right_scaled]

        # 3) cmd 객체에 적용
        self.cmd_L.angle_set = left_clamped
        self.cmd_L.mode      = 0b0001

        self.cmd_R.angle_set = right_clamped
        self.cmd_R.mode      = 0b0001


        try:
            self.HandCmb_publisher_L.Write(self.cmd_L)
            self.HandCmb_publisher_R.Write(self.cmd_R)
            # logger_mp.info(f"Sending DDS Command - Left: {self.cmd_L.angle_set}, Right: {self.cmd_R.angle_set}")
        except TypeError as e:
            logger_mp.error("[Error] Failed to send DDS command due to TypeError:", e)
        except Exception as e:
            logger_mp.error("[Error] Failed to send DDS command due to unexpected error:", e)
            
            
    def control_process(self, right_hand_array, right_hand_state_array,
                              hand_data_lock = None, out_hand_state_array = None, out_hand_action_array = None):
        self.running = True
        right_q_target = np.full(Kistar_Num_Motors, 1.0)
        try:
            while self.running:

                mode_data = self.record_mode_shm.read_data()
                home = mode_data["home"]
                replay = mode_data["replay"]
                deploy = mode_data["deploy"]

                start_time = time.time()
                # get dual hand state
                right_hand_mat = np.array(right_hand_array[:]).reshape(5, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = state_data / 1000.0  # milli-radian -> radian 변환


                if not np.all(right_hand_mat[4] == np.array([-0.8, 0.3, 0.15])): # if hand data has been initialized.
                    ref_right_value = right_hand_mat#[inspire_tip_indices]

                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                                

                    # The q_target now is in radians, ranges:
                    #     - idx 0:   0~1.52
                    #     - idx 1:   0~1.05
                    #     - idx 2~5: 0~1.47

                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Kistar_Num_Motors):
                        if idx <= 3:
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.52)
                        elif idx == 4:
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.05)
                        elif idx == 5:
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.47)

                # get dual hand action
                else : 
                    left_q_target  = np.full(Kistar_Num_Motors, 1.0)
                    right_q_target = np.full(Kistar_Num_Motors, 1.0)

                action_data = np.concatenate((left_q_target, right_q_target))    
                if out_hand_state_array and out_hand_action_array:
                    with hand_data_lock:
                        out_hand_state_array[:] = state_data
                        out_hand_action_array[:] = action_data

                # logger_mp.info("\n=== [DEBUG] Sending DDS Command ===")
                if not replay and not deploy:
                    self.ctrl_dual_hand(left_q_target, right_q_target)


                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger_mp.error("KeyboardInterrupt, exiting program...")
        except Exception as e:
            logger_mp.error(f"[Main Error] {e}")
            traceback.print_exc()

        finally:
            logger_mp.info("Inspire_Controller has been closed.")
            self.record_mode_shm.worker_close()         

class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11