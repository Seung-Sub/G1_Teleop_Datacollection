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

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

inspire_tip_indices = [4, 9, 14, 19, 24]
Inspire_Num_Motors = 6
kTopicInspireCommand_L = "rt/inspire_hand/ctrl/l"
kTopicInspireCommand_R = "rt/inspire_hand/ctrl/r"
kTopicInspireState_L = "rt/inspire_hand/state/l"
kTopicInspireState_R = "rt/inspire_hand/state/r"

class Inspire_Controller:
    def __init__(self, shm_name, shared_lock, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None, \
                 dual_hand_action_array = None, fps = 100.0, Unit_Test = False):
        logger_mp.info("Initialize Inspire_Controller...")

        import yaml
        cfg = yaml.safe_load(open("utils/lan_config.yaml"))
        ChannelFactoryInitialize(0, cfg["network_interface"])
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_name["record_mode_shm"])

        self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)

        # Separate publishers and subscribers for left and right hand
        self.HandCmb_publisher_L = ChannelPublisher(kTopicInspireCommand_L, inspire_dds.inspire_hand_ctrl)
        self.HandCmb_publisher_L.Init()

        self.HandCmb_publisher_R = ChannelPublisher(kTopicInspireCommand_R, inspire_dds.inspire_hand_ctrl)
        self.HandCmb_publisher_R.Init()

        self.HandState_subscriber_L = ChannelSubscriber(kTopicInspireState_L, inspire_dds.inspire_hand_state)
        self.HandState_subscriber_L.Init()

        self.HandState_subscriber_R = ChannelSubscriber(kTopicInspireState_R, inspire_dds.inspire_hand_state)
        self.HandState_subscriber_R.Init()

        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        
        self.cmd_L = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.cmd_R = inspire_hand_defaut.get_inspire_hand_ctrl()


        # Separate subscribe threads for each hand
        self.subscribe_Lstate_thread = threading.Thread(target=self._subscribe_hand_state, args=(self.HandState_subscriber_L, self.left_hand_state_array))
        self.subscribe_Rstate_thread = threading.Thread(target=self._subscribe_hand_state, args=(self.HandState_subscriber_R, self.right_hand_state_array))
        self.subscribe_Lstate_thread.daemon = True
        self.subscribe_Lstate_thread.start()        
        self.subscribe_Rstate_thread.daemon = True
        self.subscribe_Rstate_thread.start()
        
        while True:
            if any(self.right_hand_state_array): # any(self.left_hand_state_array) and 
                break
            time.sleep(0.01)
            logger_mp.info("[Inspire_Controller] Waiting to subscribe dds...")

        hand_control_thread = threading.Thread(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_thread.daemon = True
        hand_control_thread.start()


        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self, subscriber, state_array):
        while True:
            hand_msg = subscriber.Read()
            if hand_msg is not None:
                for idx in range(Inspire_Num_Motors):
                    state_array[idx] = hand_msg.angle_act[idx]
            time.sleep(0.002)

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
            
            
    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        self.running = True
        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)
        try:
            while self.running:

                mode_data = self.record_mode_shm.read_data()
                home = mode_data["home"]
                replay = mode_data["replay"]
                deploy = mode_data["deploy"]

                start_time = time.time()
                # get dual hand state
                left_hand_mat  = np.array(left_hand_array[:]).reshape(5, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(5, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))
                state_data = state_data / 1000.0  # milli-radian -> radian 변환


                if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[4] == np.array([-0.8, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_mat#[inspire_tip_indices]
                    ref_right_value = right_hand_mat#[inspire_tip_indices]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                                

                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                # get dual hand action
                else : 
                    left_q_target  = np.full(Inspire_Num_Motors, 1.0)
                    right_q_target = np.full(Inspire_Num_Motors, 1.0)

                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                # logger_mp.info("\n=== [DEBUG] Sending DDS Command ===")
                if not replay and not deploy:
                    right_q_target =np.array([1,1,1,1,1,1])

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