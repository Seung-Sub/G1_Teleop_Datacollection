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

kTopicLowCommand       = "rt/lowcmd"     # Phase N: hoist 모드 (전체 35 motor 직접 제어)
kTopicLowCommand_Motion = "rt/arm_sdk"   # Phase N: loco 모드 (motion mode, arm 14 + weight)
kTopicLowState         = "rt/lowstate"
G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35

# Phase N (PART4 §0.1, xr_teleoperate robot_arm.py 검증 값):
# kp/kd 게인 — motion mode init 시 관절 종류별 차등 적용.
_MOTION_KP_HIGH = 300.0   # leg/waist 기본 (강한 모터)
_MOTION_KD_HIGH = 3.0
_MOTION_KP_LOW  = 80.0    # weak motor (ankle pitch, shoulder, elbow) + arm 기본
_MOTION_KD_LOW  = 3.0
_MOTION_KP_WRIST = 40.0   # wrist roll/pitch/yaw
_MOTION_KD_WRIST = 1.5

# arm_sdk weight bit motor index. G1_29 의 not-used joint 0 = motor id 29.
# motor_cmd[29].q 가 motion controller 의 arm_sdk weight 신호로 해석됨 (0..1).
_ARM_SDK_WEIGHT_MOTOR_ID = 29
 

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

class _ContextNop:
    """no-op context manager — ctrl_lock 이 아직 생성되기 전 _init_motion_mode_lock 에서 사용."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


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
    def __init__(self, mode, lower_body='hoist'):
        """G1 본체 제어 컨트롤러.

        Args:
            mode: 'teleop' / 'gr00t' / 'gr00t_zed' — kp/kd/default_dof_pos 프로파일
                  (g1_control/joint_setting.yaml).
            lower_body: 'hoist' (default) 또는 'loco' (Phase N).
                'hoist' = rt/lowcmd, 35 모터 전체 직접 제어. 호이스트로 매달았거나
                          다리 안 쓰는 환경 전용 (사용자 운용).
                'loco'  = rt/arm_sdk, motion mode. 상체 (arm 14) 만 명령하고 하체
                          (leg/waist) 는 motion controller (내장 LocoClient/SDK) 가
                          밸런싱 + 보행 처리. arm_sdk weight (motor_cmd[29].q) 가
                          0→1 ramp 후 arm 명령 활성.
        """
        logger_mp.info(f"Initialize G1_29_ArmController (lower_body={lower_body})...")

        self.mode = mode
        self.lower_body = lower_body
        self._arm_sdk_weight = 0.0   # loco 모드 — engage_arm_sdk 가 0→1 ramp
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

        # Phase N: 토픽 분기. lowcmd = hoist (직접 제어) / arm_sdk = loco (motion mode).
        _pub_topic = kTopicLowCommand_Motion if self.lower_body == 'loco' else kTopicLowCommand
        self.lowcmd_publisher = ChannelPublisher(_pub_topic, LowCmd_)
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


        # Phase N: init 시퀀스 분기.
        # hoist: 기존 zero_torque → move_to_default → default_pos_state (물리 리모컨 start/A 대기)
        # loco : 팔 빼고 현재 q 로 잠금 + kp/kd 차등. motion controller 가 leg/waist 제어.
        #        zero_torque/move_to_default 호출 안 함 (motion controller 와 충돌 위험).
        if self.lower_body == 'loco':
            self._init_motion_mode_lock()
        else:
            self.zero_torque_state()
            self.move_to_default_pos()
            self.default_pos_state()


        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info(f"Initialize G1_29_ArmController OK (lower_body={self.lower_body})!\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def _subscribe_motor_state(self):
        # daemon thread — silent exception 으로 죽으면 self.remote 갱신이 멈춰서
        # default_pos_state 등 button 폴링 코드가 hang 된다. 따라서 try/except 로
        # 둘러싸고 예외 발생 시 print + thread 살리기.
        loop_n = 0
        last_hb = time.time()
        while not self._stop_event.is_set():
            try:
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
                loop_n += 1
                # 5초마다 heartbeat — A/start 버튼 hang 디버깅용.
                now = time.time()
                if now - last_hb >= 5.0:
                    print(f"[G1_Ctrl/sub_hb] alive  loops={loop_n} "
                          f"button.start={self.remote.button[KeyMap.start]} "
                          f"button.A={self.remote.button[KeyMap.A]}", flush=True)
                    last_hb = now
            except Exception as e:
                # 절대로 thread 죽으면 안 됨 — 예외 로깅 후 계속.
                import traceback
                print(f"[G1_Ctrl/sub_ERROR] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                time.sleep(0.1)

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

    # ----- Phase N: motion mode helpers -----------------------------------
    @staticmethod
    def _is_weak_motor(motor_id_int: int) -> bool:
        """xr_teleoperate robot_arm.py 의 weak_motors 와 동일.
        weak = ankle pitch × 2 + shoulder × 6 + elbow × 2 = 8 모터.
        """
        return motor_id_int in {
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        }

    @staticmethod
    def _is_wrist_motor(motor_id_int: int) -> bool:
        """wrist roll/pitch/yaw 양팔 = 6 모터."""
        return motor_id_int in {
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        }

    def _init_motion_mode_lock(self):
        """loco 모드 init — 팔 빼고 전부 현재 q 로 잠금 + kp/kd 차등.
        (xr_teleoperate robot_arm.py:108-131 방식 그대로 이식.)

        motion controller 가 leg/waist 를 제어하므로 zero_torque/move_to_default
        호출하지 않는다. 잠금 q 는 lowstate 구독으로 받은 현재값.
        """
        # lowstate 구독 thread 가 이미 시작됐고 첫 메시지 도착 후. all_motor_q 캐시.
        all_q = self.get_current_motor_q()
        if all_q is None:
            logger_mp.error("[G1_29_ArmController] _init_motion_mode_lock: lowstate 미수신 — abort")
            return

        arm_ids = {i.value for i in G1_29_JointArmIndex}
        for jid in G1_29_JointIndex:
            self.msg.motor_cmd[jid].mode = 1
            if jid.value in arm_ids:
                if self._is_wrist_motor(jid.value):
                    kp, kd = _MOTION_KP_WRIST, _MOTION_KD_WRIST
                else:
                    kp, kd = _MOTION_KP_LOW, _MOTION_KD_LOW
            else:
                if self._is_weak_motor(jid.value):
                    kp, kd = _MOTION_KP_LOW, _MOTION_KD_LOW
                else:
                    kp, kd = _MOTION_KP_HIGH, _MOTION_KD_HIGH
            self.msg.motor_cmd[jid].kp = float(kp)
            self.msg.motor_cmd[jid].kd = float(kd)
            self.msg.motor_cmd[jid].q  = float(all_q[jid.value])
            self.msg.motor_cmd[jid].dq = 0.0
            self.msg.motor_cmd[jid].tau = 0.0
        # arm_sdk weight bit motor — init 시 0 (engage_arm_sdk 가 1 로 ramp).
        self.msg.motor_cmd[_ARM_SDK_WEIGHT_MOTOR_ID].q = 0.0
        # 초기 arm_q_target 도 현재 arm q 로 (engage_arm_sdk ramp 동안 급가속 방지).
        with self.ctrl_lock if hasattr(self, 'ctrl_lock') else _ContextNop():
            self.arm_q_target = np.asarray(
                [all_q[jid.value] for jid in G1_29_JointArmIndex], dtype=np.float32
            )
        logger_mp.info("[G1_29_ArmController] motion mode init lock OK")

    def engage_arm_sdk(self, ramp_sec: float = 2.0):
        """loco 전용 — arm_sdk weight 0→1 점진 (안전).

        xr_teleoperate 는 1.0 직접 설정 (clip_arm_q_target 이 막아준다는 가정) 이나
        공식 문서는 점진 권장. 우리는 안전하게 점진.
        ramp 동안 arm_q_target = 현재 arm q (외부에서 갱신 안 되도록 호출자 책임).
        """
        if self.lower_body != 'loco':
            return
        steps = max(1, int(ramp_sec / 0.02))
        for i in range(steps + 1):
            w = float(i) / steps
            self._arm_sdk_weight = w
            time.sleep(0.02)
        logger_mp.info("[G1_29_ArmController] arm_sdk engaged (weight=1.0)")

    def disengage_arm_sdk(self, ramp_sec: float = 2.0):
        """loco 전용 — arm_sdk weight 1→0 점진. 종료 시 호출.

        xr_teleoperate ctrl_dual_arm_go_home 의 종료 ramp 와 동일 패턴 (np.linspace(1,0,101)
        + sleep(0.02) → 2.02 초).
        """
        if self.lower_body != 'loco':
            return
        steps = max(1, int(ramp_sec / 0.02))
        for i in range(steps + 1):
            w = 1.0 - float(i) / steps
            self._arm_sdk_weight = w
            time.sleep(0.02)
        logger_mp.info("[G1_29_ArmController] arm_sdk disengaged (weight=0.0)")
    
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

            # Phase N: loco 모드는 arm 14 + arm_sdk weight 만 갱신. leg/waist 는
            # _init_motion_mode_lock 의 잠금값 유지 (motion controller 가 실제 제어).
            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            if self.lower_body == 'loco':
                for idx, id in enumerate(G1_29_JointArmIndex):
                    self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                    self.msg.motor_cmd[id].dq = 0
                    self.msg.motor_cmd[id].tau = arm_tauff_target[idx]
                # arm_sdk weight bit (motor 29). 0=disengaged → motion controller 만 동작,
                # 1=engaged → arm 명령 따라감. engage_arm_sdk 가 0→1 ramp 적용.
                self.msg.motor_cmd[_ARM_SDK_WEIGHT_MOTOR_ID].q = float(self._arm_sdk_weight)
            else:
                # hoist (기존 동작 100% 보존): leg+waist+arm 전체 발행.
                cliped_leg_q_target = self.clip_leg_q_target(leg_q_target, velocity_limit = self.leg_velocity_limit)
                cliped_waist_q_target = self.clip_waist_q_target(waist_q_target, velocity_limit = self.waist_velocity_limit)

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

        # Phase N 이후 G1_29_JointIndex 에 motion mode 의 arm_sdk weight bit (id 29)
        # 와 unused slot (30~34) 이 추가되어 enum 총 길이 35. 하지만 hoist 모드의
        # init/default_pos 는 실 G1-29 본체의 진짜 29 joint (id 0..28) 만 명령해야
        # 한다 (default_dof_pos 도 length 29, 30~34 motor 는 G1-29 본체 미존재).
        real_joints = list(G1_29_JointIndex)[:29]

        # 현재 각도 읽기 (실 29 joint 만)
        init_pos = np.array(
            [self.lowstate_buffer.GetData().motor_state[i].q for i in real_joints],
            dtype=np.float32
        )

        for step in range(steps + 1):
            alpha = step / steps         # 0 → 1 로 선형 증가
            for idx, motor_id in enumerate(real_joints):
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
        # move_to_default_pos 와 동일 — 실 29 joint 만 명령 (Phase N 이후 enum 길이 35).
        real_joints = list(G1_29_JointIndex)[:29]
        loop_n = 0
        last_hb = time.time()
        while self.remote.button[KeyMap.A] != 1:
            for idx, motor_id in enumerate(real_joints):
                self.msg.motor_cmd[motor_id].mode = 1
                self.msg.motor_cmd[motor_id].q  = float(self.default_dof_pos[idx])
                self.msg.motor_cmd[motor_id].dq = 0.0
                self.msg.motor_cmd[motor_id].kp = float(self.kp[idx] if idx < len(self.kp) else 0.0)
                self.msg.motor_cmd[motor_id].kd = float(self.kd[idx]  if idx < len(self.kd)  else 0.0)
                self.msg.motor_cmd[motor_id].tau = 0.0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            time.sleep(self.control_dt)
            loop_n += 1
            # 1초마다 heartbeat — A 폴링이 실시간으로 button 보고 있는지 확인.
            # (만약 subscribe thread 가 죽었다면 button.A 가 영원히 0 유지 — 그래도
            #  이 print 는 계속 나와서 main thread 가 정상임을 확인 가능.)
            now = time.time()
            if now - last_hb >= 1.0:
                print(f"[G1_Ctrl/dps_hb] loops={loop_n} "
                      f"button.A={self.remote.button[KeyMap.A]} "
                      f"button.start={self.remote.button[KeyMap.start]} "
                      f"button.B={self.remote.button[KeyMap.B]} "
                      f"(누른 순간 1 로 변해야 함)", flush=True)
                last_hb = now
        print("[G1_Ctrl] default_pos_state 종료 — A 감지 성공", flush=True)
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
    
    # Phase N: kNotUsedJoint0 (motor id 29) = arm_sdk weight bit (motion mode 전용).
    # 나머지 30~34 는 G1_29 에서 사용 안 함 (다른 G1 변종에서 사용 가능).
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34
