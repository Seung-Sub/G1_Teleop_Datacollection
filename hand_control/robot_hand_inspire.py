"""Inspire RH56E2 (FTP / RH56DFTP) 양손 컨트롤러 — DEX3_Controller 와 동일 인터페이스.

토폴로지 (검증됨):
    핸드는 Modbus-TCP (R=192.168.123.210:6000, L=192.168.123.211:6000).
    worker_hand_dds 의 ModbusDataHandler 가 DDS<->Modbus 브리지:
      rt/inspire_hand/ctrl/{l,r} 구독 -> Modbus write,
      Modbus read -> rt/inspire_hand/state|touch/{l,r} publish.
    => 이 컨트롤러는 DDS 만 사용 (ctrl publish + state subscribe).

안전 제어 (스펙 §5, DEX3 와 다른 접근):
    Inspire 는 펌웨어 폐루프 선형 액추에이터라 DEX3 의 rate-limit(위치제어 워크어라운드)을
    쓰지 않는다. 대신 ctrl 메시지의 force_set + speed_set + mode 비트(angle+force+speed)를
    써서 펌웨어가 직접 과압/과부하를 막게 한다.
      - speed_set = full      -> 빠른 파지 (1000 ≈ 무부하 full travel 800ms)
      - force_set = grip_force -> 파지력 상한. force_act 도달 시 펌웨어가 그 손가락 정지(STATUS=3).
                                 빈손이면 force_act 가 안 차서 끝까지 닫히고(STATUS=2),
                                 물체가 들어오면 force_set 에서 firm 하게 holding.
      - 펌웨어 CURRENT_LIMIT(저장값) 이 하드 백스톱 -> 모터 cutoff 방지.
    추가로 state 의 status/err/temperature 를 읽어 fault DOF 는 추가 가압을 멈추고 로깅한다.

mode 비트마스크 (inspire_sdk.py:116-130): 0b0001 angle / 0b0010 pos / 0b0100 force / 0b1000 speed.
    => 0b1101 = angle+force+speed 동시.

vr_input == "controller":
    좌/우 trigger rising-edge 마다 grasp <-> release 토글 (DEX3 와 동일 UX).
    엄지(idx4 bend, idx5 yaw)는 CLI 사전설정(thumb_bend/thumb_yaw)으로 *항상* 자세 지정.
    pinky/ring/middle/index(idx0..3) 중 grasp_fingers 에 포함된 것만 토글로 open/close.

vr_input == "hand":
    기존 dex_retargeting 경로 (VR hand keypoint -> 6 normalized q). 변경 없음.
"""
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
from sharedmemory.shm_schema import RECORD_MODE_LAYOUT, QUEST_CONTROLLER

# Trigger 토글 임계값 (controller 모드)
_TRIGGER_THRESH = 0.5

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

inspire_tip_indices = [4, 9, 14, 19, 24]
Inspire_Num_Motors = 6
kTopicInspireCommand_L = "rt/inspire_hand/ctrl/l"
kTopicInspireCommand_R = "rt/inspire_hand/ctrl/r"
kTopicInspireState_L = "rt/inspire_hand/state/l"
kTopicInspireState_R = "rt/inspire_hand/state/r"

# 안전/제어 상수 -----------------------------------------------------------------
_MODE_AFS      = 0b1101     # angle + force + speed 동시 (mode 비트마스크)
_ANGLE_MAX     = 1000       # angle_set/angle_act 범위 (1000=open, 0=closed)
_FORCE_MAX     = 1000       # force_set 범위 (g)
_SPEED_MAX     = 1000       # speed_set 범위
_OPEN_Q        = 1.0        # 정규화 open
_FAULT_STATUS  = (5, 6, 7)  # 5=전류보호 6=락드로터 7=고장 (스펙 §4.6)
_TEMP_HOLD_C   = 75         # 이 온도 이상 DOF 는 추가 가압 중단
_FAULT_LOG_INTERVAL = 2.0   # fault 로그 rate-limit (초)

# 손가락 이름 -> DOF idx (엄지는 thumb_bend/thumb_yaw 로 항상 자세 지정하므로 제외)
_FINGER_NAME_TO_IDX = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}


def _parse_grasp_fingers(spec):
    """'index,middle' 또는 리스트 -> {0,1,2,3} subset. 빈/None -> 전체 4지."""
    if spec is None:
        return set(_FINGER_NAME_TO_IDX.values())
    if isinstance(spec, str):
        names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    else:
        names = [str(s).strip().lower() for s in spec]
    if not names:
        return set(_FINGER_NAME_TO_IDX.values())
    idxs = set()
    for n in names:
        if n in _FINGER_NAME_TO_IDX:
            idxs.add(_FINGER_NAME_TO_IDX[n])
        elif n == "thumb":
            # 엄지는 항상 자세 지정 — grasp_fingers 무시. (명시적 경고)
            logger_mp.warning("[Inspire] 'thumb' in --grasp-fingers ignored "
                              "(엄지는 --thumb-bend/--thumb-yaw 로 항상 자세 지정).")
        else:
            logger_mp.warning(f"[Inspire] unknown finger name in --grasp-fingers: '{n}' (무시).")
    return idxs or set(_FINGER_NAME_TO_IDX.values())


class Inspire_Controller:
    def __init__(self, shm_name, shared_lock, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None, \
                 dual_hand_action_array = None, fps = 100.0, Unit_Test = False,
                 vr_input = "hand", thumb_bend = 0.5, thumb_yaw = 0.5,
                 grasp_fingers = "pinky,ring,middle,index", close_depth = 1.0,
                 grip_force = 800, grip_speed = 1000):
        """vr_input:
            - "hand": dex_retargeting from VR hand keypoints (default).
            - "controller": Quest3 controller trigger toggles grasp/release.

        controller-mode 손가락 모드 (CLI):
            grasp_fingers: 파지 시 닫히는 손가락 subset ('pinky,ring,middle,index' 부분집합).
                           엄지는 thumb_bend/thumb_yaw 로 항상 자세 지정.
            close_depth:   파지 깊이 0..1 (1.0=완전 폐쇄, 0.x=부분).
            grip_force:    force_set 0..1000 (g). 파지력 상한 = 과부하 1차 차단.
            grip_speed:    speed_set 0..1000 (1000=full ≈ 800ms).
        """
        logger_mp.info("Initialize Inspire_Controller...")

        import yaml
        cfg = yaml.safe_load(open("utils/lan_config.yaml"))
        ChannelFactoryInitialize(0, cfg["network_interface"])
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.vr_input = vr_input
        self.thumb_bend = float(np.clip(thumb_bend, 0.0, 1.0))
        self.thumb_yaw  = float(np.clip(thumb_yaw,  0.0, 1.0))

        # controller-mode 손가락 모드 설정
        self.grasp_finger_idx = _parse_grasp_fingers(grasp_fingers)
        self.close_depth      = float(np.clip(close_depth, 0.0, 1.0))
        self.grip_force       = int(np.clip(int(grip_force), 0, _FORCE_MAX))
        self.grip_speed       = int(np.clip(int(grip_speed), 0, _SPEED_MAX))
        logger_mp.info(
            f"[Inspire] grasp_fingers idx={sorted(self.grasp_finger_idx)} "
            f"close_depth={self.close_depth} grip_force={self.grip_force} grip_speed={self.grip_speed} "
            f"thumb_bend={self.thumb_bend} thumb_yaw={self.thumb_yaw}"
        )

        self.record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_name["record_mode_shm"])
        # Controller 모드일 때만 attach
        self.quest_controller_shm = None
        if self.vr_input == "controller":
            self.quest_controller_shm = SharedMemoryManager(
                QUEST_CONTROLLER, shared_lock["quest_controller_lock"],
                shm_name["quest_controller_shm"],
            )

        self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)

        # State arrays — 콜백 핸들러가 참조하므로 subscriber Init 보다 먼저 생성.
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)   # angle_act (0..1000)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        # ctrl_dual_hand safe-hold 가 읽을 현재 angle_act 참조.
        self._left_state_ref  = self.left_hand_state_array
        self._right_state_ref = self.right_hand_state_array
        # DDS 수신 시각 (host perf_counter_ns). 좌/우 별도. DEX3 와 대칭.
        self.left_state_recv_ts  = 0
        self.right_state_recv_ts = 0
        # 안전 telemetry (콜백이 갱신, control loop / ctrl_dual_hand 가 read). list[6].
        self._l_status = [0] * Inspire_Num_Motors
        self._r_status = [0] * Inspire_Num_Motors
        self._l_err    = [0] * Inspire_Num_Motors
        self._r_err    = [0] * Inspire_Num_Motors
        self._l_temp   = [0] * Inspire_Num_Motors
        self._r_temp   = [0] * Inspire_Num_Motors
        self._l_force  = [0] * Inspire_Num_Motors
        self._r_force  = [0] * Inspire_Num_Motors
        self._l_curr   = [0] * Inspire_Num_Motors
        self._r_curr   = [0] * Inspire_Num_Motors
        # fault 로그 rate-limit + 이전 상태 (변화 시에만 로깅).
        self._fault_last_log = {'l': 0.0, 'r': 0.0}
        self._prev_status    = {'l': None, 'r': None}

        # cmd 객체 — force/speed/mode 는 정적이라 init 시 1회 세팅, angle_set 만 매 publish 갱신.
        self.cmd_L = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.cmd_R = inspire_hand_defaut.get_inspire_hand_ctrl()
        for cmd in (self.cmd_L, self.cmd_R):
            cmd.force_set = [self.grip_force] * Inspire_Num_Motors
            cmd.speed_set = [self.grip_speed] * Inspire_Num_Motors
            cmd.mode      = _MODE_AFS

        # Publishers
        self.HandCmb_publisher_L = ChannelPublisher(kTopicInspireCommand_L, inspire_dds.inspire_hand_ctrl)
        self.HandCmb_publisher_L.Init()
        self.HandCmb_publisher_R = ChannelPublisher(kTopicInspireCommand_R, inspire_dds.inspire_hand_ctrl)
        self.HandCmb_publisher_R.Init()

        # Subscribers — *콜백(handler) 방식*. (DEX3 와 동일: 폴링 Read() 는 multi-participant
        #   DDS discovery 경쟁 시 init hang race 가 있어 콜백으로 도착 즉시 처리.)
        self.HandState_subscriber_L = ChannelSubscriber(kTopicInspireState_L, inspire_dds.inspire_hand_state)
        self.HandState_subscriber_L.Init(lambda msg: self._on_hand_state(msg, self.left_hand_state_array, 'l'), 1)
        self.HandState_subscriber_R = ChannelSubscriber(kTopicInspireState_R, inspire_dds.inspire_hand_state)
        self.HandState_subscriber_R.Init(lambda msg: self._on_hand_state(msg, self.right_hand_state_array, 'r'), 1)

        # 양손 모두 첫 state 수신까지 대기 (과거엔 right 만 확인 -> 왼손 0 잔류 가능).
        _wait_start = time.time()
        _wait_warned_at = 0.0
        while True:
            if self.left_state_recv_ts > 0 and self.right_state_recv_ts > 0:
                break
            now = time.time()
            if now - _wait_warned_at >= 2.0:
                _wait_warned_at = now
                missing = []
                if self.left_state_recv_ts  <= 0: missing.append("left")
                if self.right_state_recv_ts <= 0: missing.append("right")
                logger_mp.info(
                    f"[Inspire_Controller] Waiting for hand state DDS ({now - _wait_start:.1f}s)... "
                    f"미수신: {missing}. rt/inspire_hand/state/{{l,r}} publish(=worker_hand_dds 브리지) "
                    f"가 0Hz 이면 Modbus-TCP(.210/.211) 연결/핸드 전원 점검."
                )
            time.sleep(0.05)

        hand_control_thread = threading.Thread(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_thread.daemon = True
        hand_control_thread.start()

        logger_mp.info("Initialize Inspire_Controller OK!\n")

    # ------------------------------------------------------------------
    def _on_hand_state(self, msg, state_array, side='l'):
        """DDS 콜백 — state 메시지 1건 도착 시 호출 (listener 내부 스레드)."""
        if msg is None:
            return
        recv_ts = time.perf_counter_ns()
        for idx in range(Inspire_Num_Motors):
            state_array[idx] = msg.angle_act[idx]
        # 안전 telemetry 저장 (snapshot copy).
        status = list(msg.status)
        err    = list(msg.err)
        temp   = list(msg.temperature)
        if side == 'l':
            self._l_status, self._l_err, self._l_temp = status, err, temp
            self._l_force, self._l_curr = list(msg.force_act), list(msg.current)
            self.left_state_recv_ts  = recv_ts
        else:
            self._r_status, self._r_err, self._r_temp = status, err, temp
            self._r_force, self._r_curr = list(msg.force_act), list(msg.current)
            self.right_state_recv_ts = recv_ts
        self._maybe_log_fault(side, status, err, temp)

    def _maybe_log_fault(self, side, status, err, temp):
        """fault(STATUS 5/6/7 또는 err≠0 또는 과온) 를 변화 시 + rate-limit 로 로깅."""
        faulted = any(s in _FAULT_STATUS for s in status) or any(e != 0 for e in err) \
                  or any(t >= _TEMP_HOLD_C for t in temp)
        if not faulted:
            self._prev_status[side] = tuple(status)
            return
        now = time.time()
        changed = (self._prev_status[side] != tuple(status))
        if changed or (now - self._fault_last_log[side] >= _FAULT_LOG_INTERVAL):
            self._fault_last_log[side] = now
            self._prev_status[side]    = tuple(status)
            try:
                err_desc = [inspire_hand_defaut.get_error_description(e) if e else "" for e in err]
            except Exception:
                err_desc = err
            logger_mp.warning(
                f"[Inspire:{side}] FAULT status={status} err={err_desc} temp={temp} "
                f"-> 해당 DOF 추가 가압 중단(safe-hold)."
            )

    def get_hand_state_recv_ts(self) -> int:
        """좌/우 중 더 오래된 recv_ts (worker_hand_ctrl 가 obs_hand_ts 로 사용). DEX3 와 동일 의미."""
        l = int(self.left_state_recv_ts)
        r = int(self.right_state_recv_ts)
        if l <= 0 or r <= 0:
            return max(l, r)
        return min(l, r)

    # ------------------------------------------------------------------
    def _build_angle_set(self, q_target, side):
        """정규화 q(0..1, 1=open) -> angle_set(int, 0..1000) + per-DOF safe-hold.

        fault(STATUS 5/6/7 / err≠0 / 과온) DOF 는 목표를 현재 angle_act 로 대체해
        추가 가압을 멈춘다(펌웨어가 실제 cutoff 를 막고, 이 hold 는 반복 가압을 억제).
        """
        q = np.asarray(q_target, dtype=np.float64).copy()
        if side == 'l':
            cur_raw, status, err, temp = list(self._left_state_ref[:]),  self._l_status, self._l_err, self._l_temp
        else:
            cur_raw, status, err, temp = list(self._right_state_ref[:]), self._r_status, self._r_err, self._r_temp
        for idx in range(Inspire_Num_Motors):
            if status[idx] in _FAULT_STATUS or err[idx] != 0 or temp[idx] >= _TEMP_HOLD_C:
                q[idx] = cur_raw[idx] / float(_ANGLE_MAX)   # hold current
        angle_set = [int(np.clip(round(v * _ANGLE_MAX), 0, _ANGLE_MAX)) for v in q]
        return angle_set

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """정규화 q(0..1) -> DDS publish (angle+force+speed, safe-hold 적용)."""
        self.cmd_L.angle_set = self._build_angle_set(left_q_target,  'l')
        self.cmd_R.angle_set = self._build_angle_set(right_q_target, 'r')
        # force/speed/mode 는 init 에서 세팅됨 (정적). 방어적으로 mode 재확인.
        self.cmd_L.mode = _MODE_AFS
        self.cmd_R.mode = _MODE_AFS
        try:
            self.HandCmb_publisher_L.Write(self.cmd_L)
            self.HandCmb_publisher_R.Write(self.cmd_R)
        except Exception as e:
            logger_mp.error(f"[Inspire] DDS publish failed: {e}")

    # ------------------------------------------------------------------
    def _grip_q(self, grasp):
        """controller-mode 6-vector 생성 (양손 동일 convention, 0..1, 1=open).

        idx0..3 = pinky/ring/middle/index, idx4=thumb_bend, idx5=thumb_yaw.
        엄지는 항상 thumb_bend/thumb_yaw. grasp 시 grasp_fingers 포함 idx 만 closed.
        """
        closed = max(0.0, 1.0 - self.close_depth)
        q = [_OPEN_Q, _OPEN_Q, _OPEN_Q, _OPEN_Q]
        if grasp:
            for i in self.grasp_finger_idx:
                q[i] = closed
        q.append(self.thumb_bend)
        q.append(self.thumb_yaw)
        return np.array(q, dtype=np.float64)

    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        self.running = True
        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # Controller-mode toggle state (per-side latched grasp; True == closed).
        grasp_l = False
        grasp_r = False
        prev_trig_l = False
        prev_trig_r = False

        try:
            while self.running:

                mode_data = self.record_mode_shm.read_data()
                home = mode_data["home"]
                replay = mode_data["replay"]
                deploy = mode_data["deploy"]

                start_time = time.time()

                # angle_act(0..1000) -> 0..1 정규화 (1=open, 0=closed). milli-radian 아님.
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))
                state_data = state_data / float(_ANGLE_MAX)

                # =====================================================
                # vr_input == "controller": trigger toggle -> grip q
                # =====================================================
                if self.vr_input == "controller" and self.quest_controller_shm is not None:
                    cd = self.quest_controller_shm.read_data()
                    trig_l = float(cd["left_trigger"])  >= _TRIGGER_THRESH
                    trig_r = float(cd["right_trigger"]) >= _TRIGGER_THRESH

                    # Rising-edge toggle: 한 번 누를 때마다 grasp <-> release
                    if trig_l and not prev_trig_l:
                        grasp_l = not grasp_l
                        logger_mp.info(f"[Inspire] LEFT trigger toggle -> {'GRASP' if grasp_l else 'RELEASE'}")
                    if trig_r and not prev_trig_r:
                        grasp_r = not grasp_r
                        logger_mp.info(f"[Inspire] RIGHT trigger toggle -> {'GRASP' if grasp_r else 'RELEASE'}")
                    prev_trig_l = trig_l
                    prev_trig_r = trig_r

                    # home 명령 시 모두 release 로 강제 + prev_trig 동기화로 recovery 중/직후
                    # stale rising-edge 가 새 toggle 로 인식되지 않도록 lockout. (recovery 끝나면
                    # worker_g1_ik 가 home flag 를 clear -> 다음 사이클부터 정상 토글 재개.)
                    if home:
                        grasp_l = False
                        grasp_r = False
                        prev_trig_l = trig_l
                        prev_trig_r = trig_r

                    left_q_target  = self._grip_q(grasp_l)
                    right_q_target = self._grip_q(grasp_r)

                # =====================================================
                # vr_input == "hand": existing dex_retargeting path
                # =====================================================
                else:
                    left_hand_mat  = np.array(left_hand_array[:]).reshape(5, 3).copy()
                    right_hand_mat = np.array(right_hand_array[:]).reshape(5, 3).copy()

                    if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[4] == np.array([-0.8, 0.3, 0.15])):
                        ref_left_value  = left_hand_mat
                        ref_right_value = right_hand_mat

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
                    else:
                        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
                        right_q_target = np.full(Inspire_Num_Motors, 1.0)

                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array is not None and dual_hand_action_array is not None and dual_hand_data_lock is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:]  = state_data
                        dual_hand_action_array[:] = action_data

                if not replay and not deploy:
                    self.ctrl_dual_hand(left_q_target, right_q_target)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time   = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger_mp.error("KeyboardInterrupt, exiting program...")
        except Exception as e:
            logger_mp.error(f"[Main Error] {e}")
            traceback.print_exc()

        finally:
            logger_mp.info("Inspire_Controller has been closed.")
            self.record_mode_shm.worker_close()
            if self.quest_controller_shm is not None:
                try:
                    self.quest_controller_shm.worker_close()
                except Exception:
                    pass

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
