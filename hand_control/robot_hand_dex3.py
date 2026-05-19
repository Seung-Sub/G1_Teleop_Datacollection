"""Unitree DEX3 양손 컨트롤러 (Inspire_Controller 와 같은 인터페이스).

xr_teleoperate `teleop/robot_control/robot_hand_unitree.Dex3_1_Controller` 의
DDS 통신 / IntEnum / msg layout / retargeting 호출 패턴을 그대로 따르고,
우리 워크스페이스의 vr_input='hand'|'controller' 분기와 SHM 기반 record/replay/deploy
플래그 처리만 덧붙였다.

vr_input == "hand":
    dex_retargeting 으로 VR hand keypoint (5*3 tip) -> 7 motor angle (radian).
    Inspire 와 달리 0~1 normalize 가 아니라 raw joint angle.

vr_input == "controller":
    좌/우 trigger 의 rising-edge 마다 grasp <-> release 토글.
    thumb_yaw, thumb_bend (CLI 사전 설정, 0..1) 는 thumb_0, thumb_1 joint
    의 lower~upper 한계에 선형 매핑.
    thumb_2 / middle_0,1 / index_0,1 은 grasp 시 lower limit, release 시 0.
"""
import os
import time
import threading
import traceback
from enum import IntEnum
from multiprocessing import Array, Lock

import numpy as np

# unitree_sdk2py (외부 git clone 으로 설치) — DEX3 는 unitree_hg msg 사용.
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

from hand_control.hand_retargeting import HandRetargeting, HandType

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import RECORD_MODE_LAYOUT, QUEST_CONTROLLER

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


Dex3_Num_Motors = 7

kTopicDex3LeftCommand  = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState    = "rt/dex3/left/state"
kTopicDex3RightState   = "rt/dex3/right/state"

_TRIGGER_THRESH = 0.5

# DEX3 left/right URDF joint limits (rad) — 좌우 동일.
# 인덱스는 *hardware* (Dex3_*_Left_JointIndex / Right_JointIndex) 순서.
# Left  order: [Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1]
# Right order: [Thumb0, Thumb1, Thumb2, Index0,  Index1,  Middle0, Middle1]
# fingers 의 굽힘 부호는 좌우 동일 — 음수 방향이 닫힘.
_THUMB0_LIMIT = (-1.04719755,  1.04719755)   # yaw
_THUMB1_LIMIT = (-0.72431163,  0.920)        # bend
_THUMB2_OPEN, _THUMB2_CLOSED   = 0.0,  1.74532925
_M0_OPEN,     _M0_CLOSED       = 0.0, -1.57079632
_M1_OPEN,     _M1_CLOSED       = 0.0, -1.74532925
_I0_OPEN,     _I0_CLOSED       = 0.0, -1.57079632
_I1_OPEN,     _I1_CLOSED       = 0.0, -1.74532925


def _lerp_unit(u: float, lo: float, hi: float) -> float:
    """u in [0, 1] -> lo + u*(hi-lo)."""
    u = float(np.clip(u, 0.0, 1.0))
    return lo + u * (hi - lo)


def _grip_q_left(grasp: bool, thumb_yaw: float, thumb_bend: float) -> np.ndarray:
    """Left-hand 7-vector in hardware order (Thumb0,1,2, Middle0,1, Index0,1)."""
    return np.array([
        _lerp_unit(thumb_yaw,  *_THUMB0_LIMIT),
        _lerp_unit(thumb_bend, *_THUMB1_LIMIT),
        _THUMB2_CLOSED if grasp else _THUMB2_OPEN,
        _M0_CLOSED if grasp else _M0_OPEN,
        _M1_CLOSED if grasp else _M1_OPEN,
        _I0_CLOSED if grasp else _I0_OPEN,
        _I1_CLOSED if grasp else _I1_OPEN,
    ], dtype=np.float64)


def _grip_q_right(grasp: bool, thumb_yaw: float, thumb_bend: float) -> np.ndarray:
    """Right-hand 7-vector in hardware order (Thumb0,1,2, Index0,1, Middle0,1)."""
    return np.array([
        _lerp_unit(thumb_yaw,  *_THUMB0_LIMIT),
        _lerp_unit(thumb_bend, *_THUMB1_LIMIT),
        _THUMB2_CLOSED if grasp else _THUMB2_OPEN,
        _I0_CLOSED if grasp else _I0_OPEN,
        _I1_CLOSED if grasp else _I1_OPEN,
        _M0_CLOSED if grasp else _M0_OPEN,
        _M1_CLOSED if grasp else _M1_OPEN,
    ], dtype=np.float64)


class _RIS_Mode:
    """motor_mode 비트필드 인코더 (id + status + timeout)."""
    def __init__(self, id=0, status=0x01, timeout=0):
        self.id      = id      & 0x0F
        self.status  = status  & 0x07
        self.timeout = timeout & 0x01

    def to_uint8(self) -> int:
        m  = 0
        m |= (self.id      & 0x0F)
        m |= (self.status  & 0x07) << 4
        m |= (self.timeout & 0x01) << 7
        return m


class Dex3_Controller:
    """Unitree DEX3 양손 컨트롤러.

    Inspire_Controller 와 동일한 외부 인터페이스 (worker_hand_ctrl 에서 swap 가능).
    """

    def __init__(self, shm_name, shared_lock,
                 left_hand_array, right_hand_array,
                 dual_hand_data_lock=None,
                 dual_hand_state_array=None,
                 dual_hand_action_array=None,
                 fps=100.0, Unit_Test=False,
                 vr_input="hand", thumb_bend=0.5, thumb_yaw=0.5):
        logger_mp.info("Initialize Dex3_Controller...")

        import yaml
        cfg = yaml.safe_load(open("utils/lan_config.yaml"))
        ChannelFactoryInitialize(0, cfg["network_interface"])

        self.fps        = fps
        self.Unit_Test  = Unit_Test
        self.vr_input   = vr_input
        self.thumb_bend = float(np.clip(thumb_bend, 0.0, 1.0))
        self.thumb_yaw  = float(np.clip(thumb_yaw,  0.0, 1.0))

        self.record_mode_shm = SharedMemoryManager(
            RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_name["record_mode_shm"],
        )
        self.quest_controller_shm = None
        if self.vr_input == "controller":
            self.quest_controller_shm = SharedMemoryManager(
                QUEST_CONTROLLER, shared_lock["quest_controller_lock"],
                shm_name["quest_controller_shm"],
            )

        # DexPilot retargeting (양손, 같은 yml)
        self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)

        # DDS publishers/subscribers (xr_teleoperate 패턴 그대로)
        self.LeftHandCmb_publisher  = ChannelPublisher(kTopicDex3LeftCommand,  HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()
        self.LeftHandState_subscriber  = ChannelSubscriber(kTopicDex3LeftState,  HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # init cmd msg + motor_mode
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            self.left_msg.motor_cmd[id].mode = _RIS_Mode(id=id, status=0x01).to_uint8()
            self.left_msg.motor_cmd[id].q   = 0.0
            self.left_msg.motor_cmd[id].dq  = 0.0
            self.left_msg.motor_cmd[id].tau = 0.0
            self.left_msg.motor_cmd[id].kp  = 1.5
            self.left_msg.motor_cmd[id].kd  = 0.2

        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            self.right_msg.motor_cmd[id].mode = _RIS_Mode(id=id, status=0x01).to_uint8()
            self.right_msg.motor_cmd[id].q   = 0.0
            self.right_msg.motor_cmd[id].dq  = 0.0
            self.right_msg.motor_cmd[id].tau = 0.0
            self.right_msg.motor_cmd[id].kp  = 1.5
            self.right_msg.motor_cmd[id].kd  = 0.2

        # State subscribe threads
        self.subscribe_Lstate_thread = threading.Thread(
            target=self._subscribe_hand_state,
            args=(self.LeftHandState_subscriber, self.left_hand_state_array, Dex3_1_Left_JointIndex),
            daemon=True,
        )
        self.subscribe_Rstate_thread = threading.Thread(
            target=self._subscribe_hand_state,
            args=(self.RightHandState_subscriber, self.right_hand_state_array, Dex3_1_Right_JointIndex),
            daemon=True,
        )
        self.subscribe_Lstate_thread.start()
        self.subscribe_Rstate_thread.start()

        # Wait for first state msg (xr_teleoperate 와 동일 패턴)
        while True:
            if any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.info("[Dex3_Controller] Waiting to subscribe dds...")

        hand_control_thread = threading.Thread(
            target=self.control_process,
            args=(left_hand_array, right_hand_array,
                  self.left_hand_state_array, self.right_hand_state_array,
                  dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array),
            daemon=True,
        )
        hand_control_thread.start()
        logger_mp.info("Initialize Dex3_Controller OK!\n")

    # ------------------------------------------------------------------
    def _subscribe_hand_state(self, subscriber, state_array, joint_index_enum):
        while True:
            msg = subscriber.Read()
            if msg is not None:
                for idx, jid in enumerate(joint_index_enum):
                    state_array[idx] = msg.motor_state[jid].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """left/right 7-vector(rad, hardware order) -> DDS publish."""
        for idx, jid in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[jid].q = float(left_q_target[idx])
        for idx, jid in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[jid].q = float(right_q_target[idx])
        try:
            self.LeftHandCmb_publisher.Write(self.left_msg)
            self.RightHandCmb_publisher.Write(self.right_msg)
        except Exception as e:
            logger_mp.error(f"[Dex3] DDS publish failed: {e}")

    # ------------------------------------------------------------------
    def control_process(self, left_hand_array, right_hand_array,
                        left_hand_state_array, right_hand_state_array,
                        dual_hand_data_lock=None,
                        dual_hand_state_array=None, dual_hand_action_array=None):
        self.running = True

        left_q_target  = np.zeros(Dex3_Num_Motors, dtype=np.float64)
        right_q_target = np.zeros(Dex3_Num_Motors, dtype=np.float64)

        grasp_l = False
        grasp_r = False
        prev_trig_l = False
        prev_trig_r = False

        try:
            while self.running:
                mode_data = self.record_mode_shm.read_data()
                home   = bool(mode_data["home"])
                replay = bool(mode_data["replay"])
                deploy = bool(mode_data["deploy"])

                start_time = time.time()

                state_data = np.concatenate((
                    np.array(left_hand_state_array[:]),
                    np.array(right_hand_state_array[:]),
                ))  # length 14, raw radian (Inspire 와 다름 — 그대로 보관)

                # ===========================================================
                # vr_input == "controller": trigger rising-edge toggle
                # ===========================================================
                if self.vr_input == "controller" and self.quest_controller_shm is not None:
                    cd = self.quest_controller_shm.read_data()
                    trig_l = float(cd["left_trigger"])  >= _TRIGGER_THRESH
                    trig_r = float(cd["right_trigger"]) >= _TRIGGER_THRESH

                    if trig_l and not prev_trig_l:
                        grasp_l = not grasp_l
                        logger_mp.info(f"[Dex3] LEFT trigger toggle -> {'GRASP' if grasp_l else 'RELEASE'}")
                    if trig_r and not prev_trig_r:
                        grasp_r = not grasp_r
                        logger_mp.info(f"[Dex3] RIGHT trigger toggle -> {'GRASP' if grasp_r else 'RELEASE'}")
                    prev_trig_l, prev_trig_r = trig_l, trig_r

                    # home 명령 시 release 강제 + prev_trig 동기화 (lockout).
                    # worker_g1_ik 의 cosine recovery 끝나면 home flag 가 자동 clear 됨.
                    if home:
                        grasp_l = False
                        grasp_r = False
                        prev_trig_l = trig_l
                        prev_trig_r = trig_r

                    left_q_target  = _grip_q_left(grasp_l,  self.thumb_yaw, self.thumb_bend)
                    right_q_target = _grip_q_right(grasp_r, self.thumb_yaw, self.thumb_bend)

                # ===========================================================
                # vr_input == "hand": DexPilot retargeting — 현재 NOT WIRED.
                #
                # xr_teleoperate `robot_hand_unitree.Dex3_1_Controller`(L186-191)는
                # *25개* hand landmark 전체에서 `left_indices[1,:] - left_indices[0,:]`
                # 차분 벡터(6쌍)를 만들어 retarget() 에 넘긴다. 우리
                # open_television/television.py 는 현재 5개 손가락 tip 만 SHM 으로
                # 노출하므로 DEX3 DexPilot solver 가 요구하는 입력 shape 과 맞지
                # 않는다. controller-mode 가 사용자 요구사항의 핵심이므로 hand-mode
                # 는 안전한 release 자세를 publish 한다.
                #
                # 향후 확성 단계:
                #   1) television.py 에 raw 25*3 landmark Array 추가
                #   2) hand_retargeting 에 left_indices/right_indices =
                #      retargeting.optimizer.target_link_human_indices 노출
                #   3) 여기서 ref = landmarks[indices[1]] - landmarks[indices[0]]
                #      차분 벡터를 만든 후 retarget()
                # ===========================================================
                else:
                    left_q_target  = _grip_q_left(False, self.thumb_yaw, self.thumb_bend)
                    right_q_target = _grip_q_right(False, self.thumb_yaw, self.thumb_bend)

                action_data = np.concatenate((left_q_target, right_q_target))  # 14, radian
                if dual_hand_state_array is not None and dual_hand_action_array is not None and dual_hand_data_lock is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:]  = state_data
                        dual_hand_action_array[:] = action_data

                if not replay and not deploy:
                    self.ctrl_dual_hand(left_q_target, right_q_target)

                elapsed = time.time() - start_time
                time.sleep(max(0.0, 1.0 / self.fps - elapsed))

        except KeyboardInterrupt:
            logger_mp.error("KeyboardInterrupt, exiting Dex3 control.")
        except Exception as e:
            logger_mp.error(f"[Dex3] main loop error: {e}")
            traceback.print_exc()
        finally:
            logger_mp.info("Dex3_Controller has been closed.")
            try: self.record_mode_shm.worker_close()
            except Exception: pass
            if self.quest_controller_shm is not None:
                try: self.quest_controller_shm.worker_close()
                except Exception: pass


# IntEnum 정의는 xr_teleoperate `robot_hand_unitree.Dex3_1_*_JointIndex` 그대로.
class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0  = 0
    kLeftHandThumb1  = 1
    kLeftHandThumb2  = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0  = 5
    kLeftHandIndex1  = 6


class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0  = 0
    kRightHandThumb1  = 1
    kRightHandThumb2  = 2
    kRightHandIndex0  = 3
    kRightHandIndex1  = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6
