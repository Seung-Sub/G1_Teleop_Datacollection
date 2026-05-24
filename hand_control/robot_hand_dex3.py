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

# ── 파지 속도 / 안전 제어 파라미터 ────────────────────────────────────────
# rate-limit 방식: 내부 명령목표(_cmd_q)를 매 프레임 _STEP_MAX 만큼 최종목표로
# 전진시키되, _cmd_q 가 *현재 측정 위치*보다 _MAX_POS_ERR 이상 앞서지 못하게 한다.
#   - 빈손: 손가락이 자유로워 현재 위치가 명령을 잘 따라옴 → _cmd_q 가 _STEP_MAX
#           씩 빠르게 전진 → 빠른 파지/펴기.
#   - 막힘: 물체에 막혀 현재 위치가 안 따라옴 → _cmd_q 가 현재+_MAX_POS_ERR 에서
#           멈춤 → 위치오차 = _MAX_POS_ERR 로 상한 → tau = kp·err 포화 방지 (안전).
# 과거의 "매 프레임 현재±e 로 목표를 clip" 방식은 빈손에서도 목표가 현재에 묶여
# 모터 응답속도(느림)에 종속됐다. rate-limit 은 빈손 속도와 막힘 안전을 *분리*한다.
_STEP_MAX    = 0.28    # 프레임당 최대 목표 전진량 (rad). ~100Hz × 0.18 ≈ 18rad/s 빠름.
_MAX_POS_ERR = 0.30    # 명령목표가 현재보다 앞설 수 있는 최대 오차 (rad). 막힘 시 토크 상한.
#   tau_max ≈ kp · _MAX_POS_ERR. kp=2.0, err=0.30 → 0.6(정규화). 로그상 안 죽은
#   tau(~30만)가 과거 오차로 발생했음을 감안해 보수적으로 설정. 더 강하게 쥐려면
#   _MAX_POS_ERR ↑, 더 안전하게는 ↓. 속도는 _STEP_MAX 로 독립 조절.

# DEX3 left/right URDF joint limits (rad).
# 인덱스는 *hardware* (Dex3_*_Left_JointIndex / Right_JointIndex) 순서.
# Left  order: [Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1]
# Right order: [Thumb0, Thumb1, Thumb2, Index0,  Index1,  Middle0, Middle1]
#
# ⚠️ 좌우 거울대칭: thumb2/index/middle 의 *닫힘 부호가 좌우 반대* (실측 확정).
#   - 왼손 closed:  Thumb2=+, Index/Middle=-   (스펙 범위의 음수쪽 또는 양수쪽 끝)
#   - 오른손 closed: Thumb2=-, Index/Middle=+
#
# ⚠️ 관절 가동범위 (DEX3-1 공식 스펙, deg → rad):
#   Thumb0(yaw)  : -60~ 60  = -1.047~+1.047
#   Thumb1(bend) : -35~ 60  = -0.611~+1.047
#   Thumb2       :   0~100  =  0~+1.745
#   Index0       :   0~ 90  =  0~+1.571   ← 90° 한계! (과거 ±1.745 명령은 한계 초과)
#   Index1       :   0~100  =  0~+1.745
#   Middle0      :   0~ 90  =  0~+1.571   ← 90° 한계!
#   Middle1      :   0~100  =  0~+1.745
# closed 목표는 한계의 97%(_LIMIT_MARGIN) 로 둔다. 한계 정확히(100%)에 두면 빈손
# 에서도 손가락이 기계 한계에 박혀 위치오차가 상시 남아 토크가 지속 발생 → 한계
# 살짝 안쪽에서 멈추게 해 빈손 시 박힘/토크 잔류를 방지.
_LIMIT_MARGIN = 0.97

_J_THUMB2 = 1.74532925   # 100°
_J_INDEX0 = 1.57079632   # 90°
_J_INDEX1 = 1.74532925   # 100°
_J_MID0   = 1.57079632   # 90°
_J_MID1   = 1.74532925   # 100°

_THUMB0_LIMIT = (-1.04719755,  1.04719755)   # yaw -60~60 (좌우 동일)
_THUMB1_LIMIT = (-0.61086524,  1.04719755)   # bend -35~60 왼손 (스펙 범위)
# 오른손 thumb bend 는 거울대칭. 부호 반전 범위.
_THUMB1_LIMIT_R = (0.61086524, -1.04719755)

# 왼손 (Thumb2 양수쪽, Index/Middle 음수쪽 닫힘). open=0.
_THUMB2_OPEN, _THUMB2_CLOSED   = 0.0,  _J_THUMB2 * _LIMIT_MARGIN
_M0_OPEN,     _M0_CLOSED       = 0.0, -_J_MID0   * _LIMIT_MARGIN
_M1_OPEN,     _M1_CLOSED       = 0.0, -_J_MID1   * _LIMIT_MARGIN
_I0_OPEN,     _I0_CLOSED       = 0.0, -_J_INDEX0 * _LIMIT_MARGIN
_I1_OPEN,     _I1_CLOSED       = 0.0, -_J_INDEX1 * _LIMIT_MARGIN

# 오른손 (거울대칭: Thumb2 음수쪽, Index/Middle 양수쪽 닫힘). open=0.
_THUMB2_OPEN_R, _THUMB2_CLOSED_R = 0.0, -_J_THUMB2 * _LIMIT_MARGIN
_M0_OPEN_R,     _M0_CLOSED_R     = 0.0,  _J_MID0   * _LIMIT_MARGIN
_M1_OPEN_R,     _M1_CLOSED_R     = 0.0,  _J_MID1   * _LIMIT_MARGIN
_I0_OPEN_R,     _I0_CLOSED_R     = 0.0,  _J_INDEX0 * _LIMIT_MARGIN
_I1_OPEN_R,     _I1_CLOSED_R     = 0.0,  _J_INDEX1 * _LIMIT_MARGIN


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
    """Right-hand 7-vector in hardware order (Thumb0,1,2, Index0,1, Middle0,1).

    오른손은 거울대칭이라 thumb2/index/middle 의 닫힘 부호가 왼손과 반대 (*_R_* 상수).
    thumb0(yaw)/thumb1(bend) 의 lerp 범위는 좌우 동일.
    """
    return np.array([
        _lerp_unit(thumb_yaw,  *_THUMB0_LIMIT),
        _lerp_unit(thumb_bend, *_THUMB1_LIMIT_R),
        _THUMB2_CLOSED_R if grasp else _THUMB2_OPEN_R,
        _I0_CLOSED_R if grasp else _I0_OPEN_R,
        _I1_CLOSED_R if grasp else _I1_OPEN_R,
        _M0_CLOSED_R if grasp else _M0_OPEN_R,
        _M1_CLOSED_R if grasp else _M1_OPEN_R,
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
                 vr_input="hand", thumb_bend=0.5, thumb_yaw=0.5,
                 collect_tactile=False):
        logger_mp.info("Initialize Dex3_Controller...")
        # Phase K8 (P1-5): 압력센서 (press_sensor_state) 수집 on/off.
        # off (default) — 기존 동작 100% 유지. on — _subscribe_hand_state 가 메시지의
        # press_sensor_state sequence length 만 첫 번째 메시지에서 로깅. SHM 저장은
        # sequence length 확정 후 후속 작업.
        self.collect_tactile = bool(collect_tactile)
        self._tactile_logged = {'l': False, 'r': False}

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

        # State arrays & recv_ts 를 *subscriber Init 보다 먼저* 생성한다.
        # (콜백 핸들러가 이들을 참조하므로 등록 시점에 반드시 존재해야 함.)
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)
        # ctrl_dual_hand 의 위치오차 클램프가 현재 state 를 읽기 위한 참조.
        self._left_state_ref  = self.left_hand_state_array
        self._right_state_ref = self.right_hand_state_array
        # rate-limit 내부 명령목표 (이전 프레임 명령). None 이면 첫 호출 시 현재 state 로 초기화.
        self._cmd_q_left  = None
        self._cmd_q_right = None
        # Phase K3 (P0-1.3): DDS 수신 시각 (host perf_counter_ns). 좌/우 별도.
        # 콜백 핸들러가 msg 수신 직후 갱신, control_process 가 read.
        self.left_state_recv_ts  = 0
        self.right_state_recv_ts = 0

        # DDS publishers — cmd 발행용.
        self.LeftHandCmb_publisher  = ChannelPublisher(kTopicDex3LeftCommand,  HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        # DDS subscribers — *콜백(handler) 방식* 으로 등록.
        #   과거: Init() 후 별도 스레드에서 subscriber.Read() 폴링 → DDS discovery
        #   매칭 지연 시 Read() 가 계속 None 을 반환해 init 이 간헐적으로 무한 hang
        #   되는 race 가 있었음 (단독 스크립트는 participant 1개라 즉시 매칭되어 항상
        #   성공했지만, main 처럼 여러 participant 가 동시에 discovery 경쟁하면 hand
        #   subscriber 매칭이 늦어짐). xr_teleoperate 원본 및 verify_dex3_cmd.py 와
        #   동일하게 콜백(Init(handler, queuelen)) 으로 바꿔 도착 즉시 처리한다.
        self.LeftHandState_subscriber  = ChannelSubscriber(kTopicDex3LeftState,  HandState_)
        self.LeftHandState_subscriber.Init(
            lambda msg: self._on_hand_state(msg, self.left_hand_state_array,
                                            Dex3_1_Left_JointIndex, 'l'), 1)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init(
            lambda msg: self._on_hand_state(msg, self.right_hand_state_array,
                                            Dex3_1_Right_JointIndex, 'r'), 1)

        # init cmd msg + motor_mode
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            self.left_msg.motor_cmd[id].mode = _RIS_Mode(id=id, status=0x01).to_uint8()
            self.left_msg.motor_cmd[id].q   = 0.0
            self.left_msg.motor_cmd[id].dq  = 0.0
            self.left_msg.motor_cmd[id].tau = 0.0
            self.left_msg.motor_cmd[id].kp  = 2.0  # 1.0->2.0: rate-limit 으로 빠른 파지 (오차상한 e_max 로 토크 안전)
            self.left_msg.motor_cmd[id].kd  = 0.2

        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            self.right_msg.motor_cmd[id].mode = _RIS_Mode(id=id, status=0x01).to_uint8()
            self.right_msg.motor_cmd[id].q   = 0.0
            self.right_msg.motor_cmd[id].dq  = 0.0
            self.right_msg.motor_cmd[id].tau = 0.0
            self.right_msg.motor_cmd[id].kp  = 2.0  # 1.0->2.0: rate-limit 으로 빠른 파지 (오차상한 e_max 로 토크 안전)
            self.right_msg.motor_cmd[id].kd  = 0.2

        # 콜백 방식이므로 별도 폴링 스레드는 불필요 (DDS listener 가 내부 스레드에서
        # 도착 즉시 _on_hand_state 를 호출).

        # Wait for first state msg — *양손 모두* recv_ts > 0 이 될 때까지.
        #   과거엔 any(right_hand_state_array) 만 검사 → (a) 손가락이 정확히 중립이면
        #   q≈0 이라 any() 가 False 로 머물 수 있고, (b) 왼손은 아예 확인 안 해 왼손
        #   subscriber 가 매칭 전이어도 init 완료 처리되어 왼손 state 가 0 으로 남는
        #   문제가 있었음. 콜백 기반 recv_ts 로 "실제 메시지 도착" 을 정확히 판정한다.
        _wait_start = time.time()
        _wait_warned_at = 0.0
        _wait_log_interval = 2.0
        while True:
            if self.left_state_recv_ts > 0 and self.right_state_recv_ts > 0:
                break
            now = time.time()
            elapsed = now - _wait_start
            if now - _wait_warned_at >= _wait_log_interval:
                _wait_warned_at = now
                missing = []
                if self.left_state_recv_ts  <= 0: missing.append("left")
                if self.right_state_recv_ts <= 0: missing.append("right")
                print(f"[Dex3_Controller] Waiting for hand state DDS ({elapsed:.1f}s)... "
                      f"미수신: {missing}. rt/dex3/{{left,right}}/state publish 가 0Hz 이면 "
                      f"DEX3 보드/케이블/펌웨어 점검 필요.", flush=True)
            time.sleep(0.05)

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
    def _on_hand_state(self, msg, state_array, joint_index_enum, side):
        """DDS 콜백 핸들러 — 메시지 1건 도착 시 호출됨 (listener 내부 스레드).
        과거 _subscribe_hand_state 의 while/Read()/sleep 폴링을 콜백으로 대체."""
        if msg is None:
            return
        # Phase K3: 수신 직후 host 시각 캡처. control_process / worker_hand_ctrl
        # 가 이 ts 를 obs_hand_ts 로 사용해 latency 정확도 향상.
        recv_ts = time.perf_counter_ns()
        for idx, jid in enumerate(joint_index_enum):
            state_array[idx] = msg.motor_state[jid].q
        if side == 'l':
            self.left_state_recv_ts  = recv_ts
        else:
            self.right_state_recv_ts = recv_ts
        # Phase K8 (P1-5): tactile=on 시 press_sensor_state sequence length
        # 만 첫 메시지에서 1회 로깅.
        if self.collect_tactile and not self._tactile_logged.get(side, False):
            try:
                seq = getattr(msg, 'press_sensor_state', None)
                if seq is None:
                    logger_mp.warning(f"[Dex3:{side}] tactile=on but HandState has no press_sensor_state attr")
                else:
                    n_objs = len(seq)
                    if n_objs > 0:
                        p_len = len(getattr(seq[0], 'pressure', []))
                    else:
                        p_len = 0
                    logger_mp.info(
                        f"[Dex3:{side}] tactile press_sensor_state: "
                        f"{n_objs} objects, each pressure[{p_len}]. "
                        f"Total tactile values per hand = {n_objs * p_len}."
                    )
            except Exception as e:
                logger_mp.warning(f"[Dex3:{side}] tactile length 로깅 실패: {e}")
            self._tactile_logged[side] = True

    def get_hand_state_recv_ts(self) -> int:
        """좌/우 중 더 오래된 recv_ts 반환 (worker_hand_ctrl 가 obs_hand_ts 로 사용).

        더 오래된 쪽을 채택하는 이유: state_array (좌+우) 둘 다 valid 이려면 두 쪽 모두
        publish 된 시점이 필요. 최신값 둘 중 *늦은 게 도착했을 시각* = max(l, r) 보다도,
        실은 state_data = concat(left, right) 가 잘 정렬되어 있다는 보장의 lower bound
        는 min(l, r) (이전 sample 이라도 양쪽 다 도착했음). 보수적 선택.
        """
        l = int(self.left_state_recv_ts)
        r = int(self.right_state_recv_ts)
        if l <= 0 or r <= 0:
            return max(l, r)  # 아직 한쪽만 도착한 경우엔 그쪽 ts 라도 반환
        return min(l, r)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """left/right 7-vector(rad, hardware order) -> DDS publish.

        rate-limit + 오차상한 방식 (빠른 파지 + 막힘 안전 분리):
          1) 내부 명령목표 _cmd_q 를 최종목표(left/right_q_target) 쪽으로 매 프레임
             최대 _STEP_MAX 만큼 전진 (빠른 속도 확보).
          2) _cmd_q 가 *현재 측정 위치*보다 _MAX_POS_ERR 이상 앞서지 못하게 상한
             (물체에 막히면 현재가 안 따라오므로 _cmd_q 도 멈춤 → tau 포화 방지).
        빈손이면 현재 위치가 명령을 잘 따라와 _cmd_q 가 _STEP_MAX 씩 빠르게 전진하고,
        막히면 오차가 _MAX_POS_ERR 로 묶여 안전하다. (과거 현재±e clip 방식은 빈손
        속도까지 모터 응답에 묶여 느렸음 → rate-limit 으로 분리.)
        """
        cur_l = np.array(self._left_state_ref[:],  dtype=np.float64)
        cur_r = np.array(self._right_state_ref[:], dtype=np.float64)
        tgt_l = np.asarray(left_q_target,  dtype=np.float64)
        tgt_r = np.asarray(right_q_target, dtype=np.float64)

        # 첫 호출 시 내부 명령목표를 현재 위치로 초기화 (급격한 점프 방지).
        if self._cmd_q_left is None:
            self._cmd_q_left  = cur_l.copy()
        if self._cmd_q_right is None:
            self._cmd_q_right = cur_r.copy()

        def _advance(cmd, tgt, cur):
            # 1) 최종목표 쪽으로 최대 _STEP_MAX 전진
            delta = np.clip(tgt - cmd, -_STEP_MAX, _STEP_MAX)
            cmd = cmd + delta
            # 2) 현재 위치 기준 ±_MAX_POS_ERR 로 상한 (막힘 시 토크 포화 방지)
            cmd = np.clip(cmd, cur - _MAX_POS_ERR, cur + _MAX_POS_ERR)
            return cmd

        self._cmd_q_left  = _advance(self._cmd_q_left,  tgt_l, cur_l)
        self._cmd_q_right = _advance(self._cmd_q_right, tgt_r, cur_r)

        for idx, jid in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[jid].q = float(self._cmd_q_left[idx])
        for idx, jid in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[jid].q = float(self._cmd_q_right[idx])
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
