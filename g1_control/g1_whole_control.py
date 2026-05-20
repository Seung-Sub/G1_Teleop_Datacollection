import numpy as np
import threading
import time
from enum import IntEnum
import yaml

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_                                 # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from g1_control.remote_controller import RemoteController, KeyMap
from utils.rate import Rate

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
 

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None

class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]
        self.imu_quat = np.zeros(4, dtype=np.float32)
        self.gyroscope = np.zeros(3, dtype=np.float32)
        # Phase K2 (P0-1.2): DDS 수신 시각 (perf_counter_ns) + robot-internal tick (uint32).
        # SDK IDL: LowState_.tick: types.uint32 — robot-internal clock 출처.
        self.recv_ts = 0      # host perf_counter_ns at Read() 직후
        self.robot_tick = 0   # msg.tick 그대로 (robot-internal)

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class G1_29_ArmController:
    def __init__(self, mode):
        logger_mp.info("Initialize G1_29_ArmController...")

        self.mode = mode
        self.remote = RemoteController()

        with open("g1_control/joint_setting.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        prof = cfg["modes"][self.mode]  # "base" 또는 "amo"
        self.kp  = np.array(prof["kp"], dtype=np.float32)
        self.kd  = np.array(prof["kd"], dtype=np.float32)
        self.default_dof_pos = np.array(prof["default_dof_pos"], dtype=np.float32)

        self.leg_q_target = self.default_dof_pos[:12]
        self.leg_tauff_target = np.zeros(12)
        self.waist_q_target = self.default_dof_pos[12:15]
        self.waist_tauff_target = np.zeros(3)
        self.arm_q_target = self.default_dof_pos[15:29]
        self.arm_tauff_target = np.zeros(14)

        self.all_motor_q = None
        self.leg_velocity_limit = 20.0
        self.waist_velocity_limit = 20.0
        self.arm_velocity_limit = 20.0
        
        self.control_dt = 1.0 / 250.0
        self.obs_dt = 1.0 / 500.0

        self.control_rate   = Rate(250.0)
        self.obs_rate       = Rate(500.0)

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None


        # initialize lowcmd publisher and lowstate subscriber
        cfg = yaml.safe_load(open("utils/lan_config.yaml"))
        ChannelFactoryInitialize(0, cfg["network_interface"])
        
        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand, LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, LowState_)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        self._stop_event = threading.Event()
        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.01)
            logger_mp.info("[G1_29_ArmController] Waiting to subscribe dds...")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()


        self.zero_torque_state()
        self.move_to_default_pos()
        self.default_pos_state()


        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_29_ArmController OK!\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def _subscribe_motor_state(self):
        while not self._stop_event.is_set():
            msg = self.lowstate_subscriber.Read()

            if msg is not None:
                # 수신 즉시 host 시각 캡처 (Phase K2). msg.tick (robot-internal uint32) 도 보존.
                recv_ts = time.perf_counter_ns()
                self.remote.set(msg.wireless_remote)
                lowstate = G1_29_LowState()
                lowstate.recv_ts = recv_ts
                try:
                    lowstate.robot_tick = int(msg.tick)
                except Exception:
                    lowstate.robot_tick = 0
                for i in range(G1_29_Num_Motors):
                    lowstate.motor_state[i].q  = msg.motor_state[i].q
                    lowstate.motor_state[i].dq = msg.motor_state[i].dq
                lowstate.imu_quat = np.array(msg.imu_state.quaternion, dtype=np.float32)
                lowstate.gyroscope = np.array(msg.imu_state.gyroscope, dtype=np.float32)

                self.lowstate_buffer.SetData(lowstate)

            self.obs_rate.sleep()

    # ----- Phase K2 helpers ------------------------------------------------
    def get_state_recv_ts(self) -> int:
        """Latest LowState 의 host perf_counter_ns 수신 시각. 0 이면 아직 미수신."""
        state = self.lowstate_buffer.GetData()
        return int(getattr(state, "recv_ts", 0)) if state is not None else 0

    def get_state_robot_tick(self) -> int:
        """Latest LowState 의 robot-internal tick (uint32). 0 이면 아직 미수신."""
        state = self.lowstate_buffer.GetData()
        return int(getattr(state, "robot_tick", 0)) if state is not None else 0
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def clip_leg_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_leg_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_leg_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_leg_q_target
    
    def clip_waist_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_waist_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_waist_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_waist_q_target
    
    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target
    
    def _ctrl_motor_state(self):
        while not self._stop_event.is_set():
            start_time = time.time()

            with self.ctrl_lock:
                leg_q_target     = self.leg_q_target
                leg_tauff_target = self.leg_tauff_target

                waist_q_target     = self.waist_q_target
                waist_tauff_target = self.waist_tauff_target   

                arm_q_target     = self.arm_q_target
                arm_tauff_target = self.arm_tauff_target

            cliped_leg_q_target = self.clip_leg_q_target(leg_q_target, velocity_limit = self.leg_velocity_limit)
            cliped_waist_q_target = self.clip_waist_q_target(waist_q_target, velocity_limit = self.waist_velocity_limit)
            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(G1_29_JointLegIndex):
                # self.msg.motor_cmd[id].q = cliped_leg_q_target[idx]
                self.msg.motor_cmd[id].q = leg_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = leg_tauff_target[idx]      

            for idx, id in enumerate(G1_29_JointWaistIndex):
                self.msg.motor_cmd[id].q = cliped_waist_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = waist_tauff_target[idx]      

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.leg_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))
                self.waist_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))   #?
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))     #?

            self.control_rate.sleep()

    def ctrl_leg(self, leg_q_target, leg_tauff_target=None):
        '''Set control target values q & tau of the left and right arm motors.'''
        if leg_tauff_target is None:
            leg_tauff_target = np.zeros(12)
        with self.ctrl_lock:
            self.leg_q_target = leg_q_target
            self.leg_tauff_target = leg_tauff_target

    def ctrl_waist(self, waist_q_target, waist_tauff_target=None):
        '''Set control target values q & tau of the left and right arm motors.'''
        if waist_tauff_target is None:
            waist_tauff_target = np.zeros(3)
        with self.ctrl_lock:
            self.waist_q_target = waist_q_target
            self.waist_tauff_target = waist_tauff_target    
            
    def ctrl_arm(self, arm_q_target, arm_tauff_target=None):
        '''Set control target values q & tau of the left and right arm motors.'''
        if arm_tauff_target is None:
            arm_tauff_target = np.zeros(14)
        with self.ctrl_lock:
            self.arm_q_target = arm_q_target
            self.arm_tauff_target = arm_tauff_target


    def _read_from_indices(self, indices, field: str, out=None):
        """
        indices: IntEnum 컬렉션 (예: G1_29_JointLegIndex)
        field:   'q' 또는 'dq'
        out:     재사용할 numpy 배열(선택). shape=(len(indices),)
        """
        state = self.lowstate_buffer.GetData()
        if state is None:
            return None if out is None else out  # 초기 구독 전이면 None 반환(기존 로직과 호환)

        ms = state.motor_state
        n = len(indices)

        if out is None:
            arr = np.empty(n, dtype=np.float32)
            for i, idx in enumerate(indices):
                arr[i] = getattr(ms[int(idx)], field)
            return arr
        else:
            # 성능을 위해 타입/길이는 호출 측에서 맞춰주는 것을 권장
            for i, idx in enumerate(indices):
                out[i] = getattr(ms[int(idx)], field)
            return out

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self, out=None):
        """Return current state q of all body motors."""
        return self._read_from_indices(G1_29_JointIndex, "q", out=out)

    # --- Arm ---
    def get_current_arm_q(self, out=None):
        """Return current state q of the left and right arm motors."""
        return self._read_from_indices(G1_29_JointArmIndex, "q", out=out)

    def get_current_arm_dq(self, out=None):
        """Return current state dq of the left and right arm motors."""
        return self._read_from_indices(G1_29_JointArmIndex, "dq", out=out)

    # --- Waist ---
    def get_current_waist_q(self, out=None):
        """Return current state q of the waist motors."""
        return self._read_from_indices(G1_29_JointWaistIndex, "q", out=out)

    def get_current_waist_dq(self, out=None):
        """Return current state dq of the waist motors."""
        return self._read_from_indices(G1_29_JointWaistIndex, "dq", out=out)

    # --- Leg ---
    def get_current_leg_q(self, out=None):
        """Return current state q of the leg motors."""
        return self._read_from_indices(G1_29_JointLegIndex, "q", out=out)

    def get_current_leg_dq(self, out=None):
        """Return current state dq of the leg motors."""
        return self._read_from_indices(G1_29_JointLegIndex, "dq", out=out)

    # IMU도 버퍼 재사용 옵션 추가 (선택)
    def get_imu_state_data(self, out_quat=None, out_gyro=None):
        """
        Returns:
            (imu_quat, gyro)
            - out_* 제공 시 그 버퍼를 채워서 반환
        """
        state = self.lowstate_buffer.GetData()
        if state is None:
            return None, None

        if out_quat is None:
            imu_quat = state.imu_quat.copy()
        else:
            out_quat[...] = state.imu_quat
            imu_quat = out_quat

        if out_gyro is None:
            gyro = state.gyroscope.copy()   # gyroscope shape=(3,)
        else:
            out_gyro[...] = state.gyroscope
            gyro = out_gyro

        return imu_quat, gyro



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def ctrl_defualt_pose(self):
        '''Move both the left and right arms of the robot to their home position by repeatedly sending target joint angles and checking until close enough.'''
        logger_mp.info("[G1_29_ArmController] ctrl_defualt_pose start...")
        with self.ctrl_lock:
            self.leg_q_target = self.default_dof_pos[:12]
            self.leg_tauff_target = np.zeros(12)
            self.waist_q_target = self.default_dof_pos[12:15]
            self.waist_tauff_target = np.zeros(3)
            self.arm_q_target = self.default_dof_pos[15:29]
            self.arm_tauff_target = np.zeros(14)

        tolerance = 0.05  # 허용 오차
        max_iter = 300    # 무한 루프 방지를 위한 안전 장치
        iter_count = 0

        while True:
            current_leg_q = self.get_current_leg_q()
            current_waist_q = self.get_current_waist_q()
            current_arm_q = self.get_current_arm_q()

            # 실제 제어 명령 전송
            self.ctrl_leg(self.leg_q_target, self.leg_tauff_target)
            self.ctrl_waist(self.waist_q_target, self.waist_tauff_target)
            self.ctrl_arm(self.arm_q_target, self.arm_tauff_target)

            if np.all(np.abs(current_arm_q - self.arm_q_target) < tolerance) and np.all(np.abs(current_waist_q - self.waist_q_target) < tolerance):
                logger_mp.info("[G1_29_ArmController] both arms have reached the home position.")
                break

            iter_count += 1
            if iter_count > max_iter:
                logger_mp.info("[G1_29_ArmController] Home movement timeout.")
                break

            time.sleep(0.05)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.leg_velocity_limit = 30.0
        self.waist_velocity_limit = 30.0
        self.arm_velocity_limit = 30.0

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def zero_torque_state(self):
        print("[G1_29_ArmController] zero_torque_state press start button to default pos")

        while self.remote.button[KeyMap.start] != 1:
            for id in G1_29_JointIndex:
                self.msg.motor_cmd[id].mode = 1
                self.msg.motor_cmd[id].q = 0
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].kp = 0
                self.msg.motor_cmd[id].kd = 0
                self.msg.motor_cmd[id].tau = 0


            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

    def daming_state(self):
        print("[G1_29_ArmController] Setting dual arm to soft mode...")
        while self.remote.button[KeyMap.start] != 1:

            for id in G1_29_JointIndex:
                self.msg.motor_cmd[id].mode = 1
                self.msg.motor_cmd[id].q = 0
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].kp = 0
                self.msg.motor_cmd[id].kd = 8
                self.msg.motor_cmd[id].tau = 0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
        

    def move_to_default_pos(self):
        """팔과 허리를 기본 자세(self.default_dof_pos)로 2 초에 걸쳐 선형 보간 이동."""
        print("Moving to default position.")
        total_time = 2.0
        steps = int(total_time / self.control_dt)

        # 현재 각도 읽기
        init_pos = np.array(
            [self.lowstate_buffer.GetData().motor_state[i].q for i in G1_29_JointIndex],
            dtype=np.float32
        )

        for step in range(steps + 1):
            alpha = step / steps         # 0 → 1 로 선형 증가
            for idx, motor_id in enumerate(G1_29_JointIndex):
                target_q = init_pos[idx] * (1 - alpha) + self.default_dof_pos[idx] * alpha
                self.msg.motor_cmd[motor_id].mode = 1
                self.msg.motor_cmd[motor_id].q  = float(target_q)
                self.msg.motor_cmd[motor_id].dq = 0.0
                self.msg.motor_cmd[motor_id].kp = float(self.kp[idx] if idx < len(self.kp) else 0.0)
                self.msg.motor_cmd[motor_id].kd = float(self.kd[idx]  if idx < len(self.kd)  else 0.0)
                self.msg.motor_cmd[motor_id].tau = 0.0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            time.sleep(self.control_dt)


    def default_pos_state(self) -> None:
        """기본 자세를 유지하며 A 버튼이 눌릴 때까지 대기."""
        print("In default position state. Press A to end initial setting.")
        while self.remote.button[KeyMap.A] != 1:
            for idx, motor_id in enumerate(G1_29_JointIndex):
                self.msg.motor_cmd[motor_id].mode = 1
                self.msg.motor_cmd[motor_id].q  = float(self.default_dof_pos[idx])
                self.msg.motor_cmd[motor_id].dq = 0.0
                self.msg.motor_cmd[motor_id].kp = float(self.kp[idx] if idx < len(self.kp) else 0.0)
                self.msg.motor_cmd[motor_id].kd = float(self.kd[idx]  if idx < len(self.kd)  else 0.0)
                self.msg.motor_cmd[motor_id].tau = 0.0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            time.sleep(self.control_dt)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    def stop(self) -> None:
        """루프를 안전하게 종료하고 스레드를 join."""
        self._stop_event.set()
        self.subscribe_thread.join()
        self.publish_thread.join()


class G1_29_JointWaistIndex(IntEnum):
    # Waist
    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21


    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointLegIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5
    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # # not used
    # kNotUsedJoint0 = 29
    # kNotUsedJoint1 = 30
    # kNotUsedJoint2 = 31
    # kNotUsedJoint3 = 32
    # kNotUsedJoint4 = 33
    # kNotUsedJoint5 = 34
