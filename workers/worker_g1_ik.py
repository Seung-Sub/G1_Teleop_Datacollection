import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.state import State, EventsSnapshot, next_state
from utils.rate import Rate
from utils.mat_tool import fast_mat_inv, cosine_ease, se3_interp
from utils.modality_layout import layout_from_modality_json, split_state_vec

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    TELEVISION, RECORD_MODE_LAYOUT, WORKER_FREQ, RECORD_EPISODE_LAYOUT,
    RECORD_TASK_LAYOUT, ROBOT_ACTION, ROBOT_OBS, QUEST_CONTROLLER,
)

from g1_control.g1_ik import G1_29_ArmIK

import pandas as pd

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


# ---- Controller-mode tunables -----------------------------------------------
GRIP_THRESH       = 0.5     # squeeze >= GRIP_THRESH 이면 grip 누름으로 인식
BUTTON_THRESH     = 0.5
WAIST_LIMITS      = np.array([1.05, 0.6, 0.6])   # |yaw|, |roll|, |pitch| (rad) safety clamp
WAIST_GAIN        = np.array([1.0, 1.0, 1.0])    # HMD delta -> waist target 매핑 게인

# Head ready-pose (controller 모드 head 고정값). DXL tick [2048, 1934] 에 해당.
# (1934 - 2048) / (4096 / 2π) = -0.17486 rad. yaw=0(center), pitch=-0.175(약 -10° 아래로 기울임).
# 이 값은 카메라 mount install 자세에 맞춰진 hardware-specific ready 자세.
HEAD_READY_RAD    = np.array([0.0, -0.17486])

# Recovery (Right-A 트리거) cosine ease 동안 controller 입력 전체 lockout.
# 50Hz × 3.0s = 150 step. ease 끝나면 anchor 모두 reset 후 controller 입력 재개.
RECOVERY_DURATION_SEC = 3.0


def _yaw_align_from_head(head_mat):
    """grip engage 시점의 HMD pose 에서 yaw 정렬 회전(3x3) 을 계산.

    head_mat 는 robot-convention(z-up, x-front, y-left) 기저. 사용자가 향한 수평
    yaw 의 *역회전* 을 곱하면 사용자 정면이 로봇 +x(정면) 로 정렬된다.

    yaw 추출은 euler 'zyx' 분해의 첫 성분(yaw) 을 사용한다. 검증 결과 순수
    yaw+pitch 회전에서 euler-zyx yaw 는 pitch 크기와 무관하게 정확하다 (forward 축
    수평투영 방식은 pitch 가 크면 yaw 를 왜곡 — pitch=-60° 에서 30°→49° 오차).
    HMD 를 목에 걸면 pitch 는 크지만 roll 은 작으므로 euler-zyx 가 안정적.

    반환: R_align (3,3) — p_aligned = R_align @ p_world. world frame yaw 오차 제거.
    """
    Rh = head_mat[:3, :3]
    try:
        yaw = float(R.from_matrix(Rh).as_euler('zyx')[0])
    except Exception:
        # degenerate (비정상 행렬) → forward 수평투영 폴백
        fwd = Rh[:, 0]
        yaw = float(np.arctan2(fwd[1], fwd[0]))
    if not np.isfinite(yaw):
        return np.eye(3)
    c, s = np.cos(-yaw), np.sin(-yaw)      # 역회전
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _events_snapshot(shared_event, g1_initialized: bool) -> EventsSnapshot:
    return EventsSnapshot(
        shutdown = shared_event['shutdown'].is_set(),
        emergency = shared_event['emergency'].is_set(),
        g1_ready = g1_initialized ,
        hand_ready = shared_event['set_hand'].is_set(),
        start = shared_event['set_start'].is_set(),
        select_pressed = False
    )

def worker_g1_ik(shared_event, shm_name, shared_lock, vr_input="hand", waist_mode="hmd"):
    """vr_input: 'hand' | 'controller'.
    waist_mode: 'hmd' (HMD R_delta → waist 매핑) | 'fixed' (init waist q 고정).
    """

    #Set SharedMemory
    television_shm       = SharedMemoryManager(TELEVISION,             shared_lock["television_lock"],       shm_name["television_shm"])
    freq_shm             = SharedMemoryManager(WORKER_FREQ,            shared_lock["freq_lock"],             shm_name["freq_shm"])
    record_mode_shm      = SharedMemoryManager(RECORD_MODE_LAYOUT,     shared_lock["record_lock"],           shm_name["record_mode_shm"])
    record_episode_shm   = SharedMemoryManager(RECORD_EPISODE_LAYOUT,  shared_lock["record_lock"],           shm_name["record_episode_shm"])
    record_task_shm      = SharedMemoryManager(RECORD_TASK_LAYOUT,     shared_lock["record_lock"],           shm_name["record_task_shm"])
    robot_action_shm     = SharedMemoryManager(ROBOT_ACTION,           shared_lock["robot_action_lock"],     shm_name["robot_action_shm"])
    robot_obs_shm        = SharedMemoryManager(ROBOT_OBS,              shared_lock["robot_obs_lock"],        shm_name["robot_obs_shm"])
    quest_controller_shm = SharedMemoryManager(QUEST_CONTROLLER,       shared_lock["quest_controller_lock"], shm_name["quest_controller_shm"])

    # 60Hz 주기로 실행 (50→60: 저장 정렬축 60Hz 와 일치시켜 action 업샘플 제거.
    # IK solve 측정 avg~3.2ms/max~3.6ms 라 60Hz(16.7ms 예산) 여유 충분. VR 소스
    # 60fps 와도 매칭. recovery(3s)는 perf_counter 경과시간 기반이라 hz 무관.)
    rate = Rate(60.0)

    # G1/Hand 컨트롤러를 아직 생성하지 않은 상태를 추적하는 플래그
    g1_initialized   = False

    # 각각 컨트롤러 객체를 저장할 변수 (None으로 초기화)
    g1_ik_solver = None

    set_home = False

    replay_demo_init = False
    replay_actions = None   # action 배열 (N×31)
    replay_g1_data   = None   # 핸드 관련 action 배열 (N×12)

    replay_length = 0       # 전체 프레임 수
    replay_frame_idx = 0    # 현재 재생 중인 프레임 인덱스

    REPLAY_HZ = 20.0
    _replay_period = 1.0 / REPLAY_HZ
    _replay_next_t = None            # 다음 프레임을 넘길 벽시각
    _replay_start_wall = None        # 재생 시작 벽시각
    _replay_rel_ts = None            # (선택) 데이터셋 타임스탬프(상대시간) 배열
    _last_written_idx = -1           # 진행률/로그 업데이트 중복 방지


    base_left_pos = base_right_pos = base_head_pos = None
    T_L_init = T_R_init = T_H_init = None

    # ---- Controller-mode (vr_input == 'controller') 상태 ----
    # SE(3) anchor / target. grip release 시 freeze, 다음 grip 시 다시 anchor 캡처.
    T_ee_target_l = None;  T_ee_target_r = None
    T_ctrl_anchor_l = None; T_ctrl_anchor_r = None
    T_ee_anchor_l = None;   T_ee_anchor_r = None
    prev_grip_l = False;    prev_grip_r = False
    # Waist clutch: HMD pose 변화량을 G1 waist [yaw, roll, pitch] 에 매핑.
    waist_anchor_head = None
    waist_anchor_q    = None
    target_waist_q    = None
    # IK 결과 NaN guard: solve_ik 가 (VR pose NaN / 솔버 실패 시) NaN sol_q 를 낼 수
    # 있다. NaN action 이 robot_action_shm + parquet 에 저장되면 데이터가 오염되므로,
    # 마지막 *정상(finite)* sol_q/sol_tauff 를 보관했다가 NaN 발생 시 그대로 유지한다.
    # (로봇도 NaN 명령 대신 직전 자세 유지 → 안전.) mat_update 의 finite guard 가 1차
    # 방어, 이게 2차 방어선.
    _last_good_sol_q     = None
    _last_good_sol_tauff = None
    # Head-yaw 정렬 (개선안 ①): grip engage 시점의 HMD yaw 를 1회 캡처해
    # 컨트롤러 변위/회전을 "사용자 정면 = 로봇 정면" 으로 정렬한다. 이로써 사용자가
    # 어느 방향으로 서서 VR 세션을 시작했든(=VR world frame 이 어디로 회전됐든)
    # 직관적 조작이 가능. 좌우 클러치가 공유 (양손 동시 engage 시 먼저 잡힌 yaw 유지).
    R_yaw_align = None   # (3,3) — world->aligned 회전. None 이면 미설정(정렬 비활성).
    # Ready 버튼 edge detection
    prev_right_a_btn = False
    # Recovery state (cosine ease + controller 입력 lockout)
    _recovery_active   = False
    _recovery_start_t  = None
    _recovery_T_l_from = None
    _recovery_T_r_from = None
    _recovery_waist_from = None

    # Phase L4 (Part 2 P1-5): IK solve 시간 계측. rolling window 의 50 cycle 누적
    # 후 avg/p95/max ms 를 freq_shm 에 publish. 50Hz 예산 (20ms) 초과 빈도 정량화용.
    from collections import deque as _deque
    _ik_solve_ms_window = _deque(maxlen=50)
    _ik_publish_every = 25  # cycle (≈0.5초)
    _ik_cycle_count = 0
    def _publish_ik_stats(samples):
        if not samples:
            return
        arr = np.asarray(samples, dtype=np.float64)
        try:
            freq_shm.write_data(
                ik_solve_ms_avg=float(arr.mean()),
                ik_solve_ms_p95=float(np.percentile(arr, 95)),
                ik_solve_ms_max=float(arr.max()),
            )
        except Exception:
            pass

    def _guard_sol(sol_q, sol_tauff):
        """IK 결과가 finite 면 last_good 갱신 후 반환, NaN/inf 면 last_good 으로 대체.
        last_good 도 없으면(에피소드 첫 NaN) None 반환 → 호출부에서 write skip."""
        nonlocal _last_good_sol_q, _last_good_sol_tauff
        if sol_q is not None and np.all(np.isfinite(sol_q)) and \
           sol_tauff is not None and np.all(np.isfinite(sol_tauff)):
            _last_good_sol_q     = np.asarray(sol_q).copy()
            _last_good_sol_tauff = np.asarray(sol_tauff).copy()
            return sol_q, sol_tauff, True
        # NaN/inf — 직전 정상값 유지 (로봇/데이터 보호)
        if _last_good_sol_q is not None:
            logger_mp.warning("[G1_IK] solve_ik 결과 NaN/inf — 직전 정상 sol 유지 (action 오염 방지).")
            return _last_good_sol_q, _last_good_sol_tauff, False
        logger_mp.warning("[G1_IK] solve_ik 결과 NaN/inf 이고 last_good 없음 — 이 cycle write skip.")
        return None, None, False

    first_loop = True

    state = State.WAIT_CONNECT
    logger_mp.info(f"[G1_IK] FSM start: WAIT_CONNECT (vr_input={vr_input})")

    try:
        while True:
            # 상태 전이
            state_next = next_state(state, _events_snapshot(shared_event, g1_initialized))
            if state_next != state:
                logger_mp.info(f"[G1_IK] {state.name} -> {state_next.name}")
            state = state_next

            if state is State.EXIT:
                break

            if state is State.WAIT_CONNECT:
                # set_g1이 올라오면 IK 초기화 1회 시도
                if shared_event['set_g1'].is_set() and not g1_initialized:
                    try:
                        # IK 솔버 초기화
                        g1_ik_solver = G1_29_ArmIK(False, True)
                        g1_initialized = True
                        logger_mp.info("[G1_IK] IK solver initialized.")
                    except Exception as e:
                        logger_mp.exception("[G1_IK] IK init failed: %s", e)
                        shared_event['emergency'].set()
                time.sleep(0.01)

            elif state is State.WAIT_START:
                # START 대기
                time.sleep(0.01)

            # 실제 IK 계산 수행
            elif state is State.RUN:
                # 1) 주파수 보고
                freq_shm.write_data(g1_freq=rate.tick_hz())

                # 2) 모드 읽기
                mode = record_mode_shm.read_data()
                home, replay, deploy = mode["home"], mode["replay"], mode["deploy"]

                if first_loop:
                    vr_data = television_shm.read_data()
                    left_wrist, right_wrist, head_pose = vr_data["left_wrist_mat"], vr_data["right_wrist_mat"], vr_data["head_rmat"]
                    base_left_pos, base_right_pos, base_head_pos = left_wrist[:3,3].copy(), right_wrist[:3,3].copy(), head_pose[:3,3].copy()

                    T_L_init, T_R_init, T_H_init = g1_ik_solver.init_pose()

                    set_home = True
                    first_loop = False


                if shared_event['set_start'].is_set() and not replay and not deploy:
                    vr_data = television_shm.read_data()

                    head_pose   = vr_data["head_rmat"]
                    left_wrist  = vr_data["left_wrist_mat"]
                    right_wrist = vr_data["right_wrist_mat"]

                    robot_obs = robot_obs_shm.read_data()
                    current_waist_q = robot_obs["obs_waist"]
                    current_head    = robot_obs["obs_head"]
                    current_arm_q   = robot_obs["obs_arm"]

                    # ===========================================================
                    # vr_input == "controller": recovery + clutch + waist + ready
                    # ===========================================================
                    if vr_input == "controller":
                        ctrl_data = quest_controller_shm.read_data()

                        # ---- Recovery state machine (controller 입력 lockout) --
                        # home rising-edge 감지: 현재 ee target / waist target 을
                        # _from 으로 캡처하고 ease 시작. ease 동안 grip/trigger/
                        # 다른 버튼은 모두 무시. ease 끝나면 anchor 전부 reset.
                        if home and not _recovery_active:
                            if T_L_init is None or T_R_init is None or T_H_init is None:
                                T_L_init, T_R_init, T_H_init = g1_ik_solver.init_pose()
                            if T_ee_target_l is None: T_ee_target_l = T_L_init.copy()
                            if T_ee_target_r is None: T_ee_target_r = T_R_init.copy()
                            _recovery_T_l_from = T_ee_target_l.copy()
                            _recovery_T_r_from = T_ee_target_r.copy()
                            _recovery_waist_from = (
                                target_waist_q.copy() if target_waist_q is not None
                                else current_waist_q.copy()
                            )
                            _recovery_active  = True
                            _recovery_start_t = time.perf_counter()
                            logger_mp.info(
                                "[G1_IK] Recovery START — 3s cosine ease to ready pose, controller LOCKOUT."
                            )

                        if _recovery_active:
                            elapsed = time.perf_counter() - _recovery_start_t
                            alpha   = elapsed / RECOVERY_DURATION_SEC
                            ease    = cosine_ease(alpha)
                            T_ee_target_l  = se3_interp(_recovery_T_l_from, T_L_init, ease)
                            T_ee_target_r  = se3_interp(_recovery_T_r_from, T_R_init, ease)
                            target_waist_q = (1.0 - ease) * _recovery_waist_from + ease * np.zeros(3)

                            rel_head_pose = T_H_init if T_H_init is not None else np.eye(4)
                            seed_waist = target_waist_q
                            current_lr_arm_qdml = np.concatenate((seed_waist, current_head, current_arm_q))
                            _ik_t0 = time.perf_counter_ns()
                            sol_q, sol_tauff, _, _ = g1_ik_solver.solve_ik(
                                T_ee_target_l, T_ee_target_r, rel_head_pose, current_lr_arm_qdml,
                            )
                            _ik_solve_ms_window.append((time.perf_counter_ns() - _ik_t0) / 1e6)
                            _ik_cycle_count += 1
                            if _ik_cycle_count % _ik_publish_every == 0:
                                _publish_ik_stats(_ik_solve_ms_window)
                            sol_q, sol_tauff, _ok = _guard_sol(sol_q, sol_tauff)  # NaN guard
                            if sol_q is not None:
                                robot_action_shm.write_data(
                                    action_body_ts   =np.int64(time.perf_counter_ns()),
                                    action_waist     =target_waist_q,
                                    action_waist_tauff=np.zeros(3),
                                    action_head      =HEAD_READY_RAD,
                                    action_arm       =sol_q[5:],
                                    action_arm_tauff =sol_tauff[5:],
                                )

                            if alpha >= 1.0:
                                # ease 종료 — anchor 전체 reset + home flag clear
                                _recovery_active   = False
                                T_ctrl_anchor_l = T_ctrl_anchor_r = None
                                T_ee_anchor_l   = T_ee_anchor_r   = None
                                waist_anchor_head = None
                                waist_anchor_q    = None
                                R_yaw_align       = None
                                # Right-A 는 현재값으로 sync — 사용자가 버튼을 계속 누르고 있어도
                                # 즉시 새 recovery 가 트리거되지 않도록 함.
                                prev_right_a_btn = float(ctrl_data["right_buttons"][0]) >= BUTTON_THRESH
                                # Grip 은 False 로 리셋 — 사용자가 grip 을 잡고 있는 채로 recovery
                                # 가 끝나면, 다음 cycle 에서 grip_l/r=True && prev=False 가 rising-edge
                                # 로 인식되어 새 ready-pose 기준의 anchor 가 자연스럽게 캡처된다.
                                # (current 로 sync 하면 anchor=None 상태에서 clutch 가 동작해 NPE.)
                                prev_grip_l = False
                                prev_grip_r = False
                                rm = record_mode_shm.read_data()
                                rm["home"] = False
                                record_mode_shm.write_data(**rm)
                                logger_mp.info("[G1_IK] Recovery COMPLETE — controller input re-enabled.")

                            rate.sleep()
                            continue

                        # ---- 정상 controller-mode 로직 (recovery 비활성) -------
                        T_ctrl_l_now = ctrl_data["left_ctrl_mat"]
                        T_ctrl_r_now = ctrl_data["right_ctrl_mat"]

                        grip_l = float(ctrl_data["left_squeeze"])  >= GRIP_THRESH
                        grip_r = float(ctrl_data["right_squeeze"]) >= GRIP_THRESH

                        # ---- ready button (right A) edge detection -------------
                        # set_start 는 clear 하지 않는다 — recovery 동안 worker_g1_ctrl
                        # 이 RUN state 유지해야 ease action 이 G1 으로 전달됨.
                        right_a = float(ctrl_data["right_buttons"][0]) >= BUTTON_THRESH
                        if right_a and not prev_right_a_btn:
                            logger_mp.info("[G1_IK] Right-A pressed -> HOME (smooth 3s recovery).")
                            rm = record_mode_shm.read_data()
                            rm["home"] = True
                            record_mode_shm.write_data(**rm)
                        prev_right_a_btn = right_a

                        # ---- left arm clutch -----------------------------------
                        if T_ee_target_l is None:
                            T_ee_target_l = T_L_init.copy() if T_L_init is not None else np.eye(4)
                        if grip_l:
                            if not prev_grip_l:
                                T_ctrl_anchor_l = T_ctrl_l_now.copy()
                                T_ee_anchor_l   = T_ee_target_l.copy()
                                # Head-yaw 정렬 (개선안 ①): 양손 모두 grip 떼진 상태에서
                                # 새로 잡힐 때만 head yaw 를 1회 캡처. 한쪽이 이미 잡고
                                # 있으면(R_yaw_align != None) 기존 정렬 유지 → 양손 일관.
                                if R_yaw_align is None:
                                    R_yaw_align = _yaw_align_from_head(head_pose)
                                logger_mp.info("[G1_IK] LEFT grip ENGAGE")
                            # Clutch delta: 위치/회전 분리 + head-yaw 정렬.
                            #  - 위치: world 평행이동 차분에 R_yaw_align 적용해 사용자
                            #          정면 기준으로 정렬 (p_t = p_t^a + R_align·(p_h - p_h^a)).
                            #  - 회전: 컨트롤러 상대회전도 R_align 으로 정렬한 뒤 EE 에 합성.
                            R_rel_l = T_ctrl_l_now[:3, :3] @ T_ctrl_anchor_l[:3, :3].T
                            R_rel_l = R_yaw_align @ R_rel_l @ R_yaw_align.T
                            dp_l    = R_yaw_align @ (T_ctrl_l_now[:3, 3] - T_ctrl_anchor_l[:3, 3])
                            T_ee_target_l = np.eye(4)
                            T_ee_target_l[:3, :3] = R_rel_l @ T_ee_anchor_l[:3, :3]
                            T_ee_target_l[:3,  3] = T_ee_anchor_l[:3, 3] + dp_l
                        else:
                            if prev_grip_l:
                                logger_mp.info("[G1_IK] LEFT grip RELEASE -> freeze EE target")
                            # freeze: T_ee_target_l 그대로 유지
                        prev_grip_l = grip_l

                        # ---- right arm clutch ----------------------------------
                        if T_ee_target_r is None:
                            T_ee_target_r = T_R_init.copy() if T_R_init is not None else np.eye(4)
                        if grip_r:
                            if not prev_grip_r:
                                T_ctrl_anchor_r = T_ctrl_r_now.copy()
                                T_ee_anchor_r   = T_ee_target_r.copy()
                                if R_yaw_align is None:
                                    R_yaw_align = _yaw_align_from_head(head_pose)
                                logger_mp.info("[G1_IK] RIGHT grip ENGAGE")
                            # 위치/회전 분리 + head-yaw 정렬 (왼팔과 동일, 주석은 왼팔 참고)
                            R_rel_r = T_ctrl_r_now[:3, :3] @ T_ctrl_anchor_r[:3, :3].T
                            R_rel_r = R_yaw_align @ R_rel_r @ R_yaw_align.T
                            dp_r    = R_yaw_align @ (T_ctrl_r_now[:3, 3] - T_ctrl_anchor_r[:3, 3])
                            T_ee_target_r = np.eye(4)
                            T_ee_target_r[:3, :3] = R_rel_r @ T_ee_anchor_r[:3, :3]
                            T_ee_target_r[:3,  3] = T_ee_anchor_r[:3, 3] + dp_r
                        else:
                            if prev_grip_r:
                                logger_mp.info("[G1_IK] RIGHT grip RELEASE -> freeze EE target")
                        prev_grip_r = grip_r

                        # 양손 모두 grip 떼지면 yaw 정렬 리셋 → 다음 engage 때 새 yaw 캡처.
                        if (not grip_l) and (not grip_r):
                            R_yaw_align = None

                        # ---- waist clutch (HMD delta -> waist target) ----------
                        # waist_mode='fixed': HMD 매핑 비활성, target = init waist q (0 벡터) 고정.
                        # G1 의 waist 모터는 PID 유지 위해 계속 publish 됨 (안전).
                        if waist_mode == 'fixed':
                            target_waist_q = np.zeros(3, dtype=np.float64)
                            waist_anchor_head = None
                            waist_anchor_q    = None
                        elif grip_l or grip_r:
                            # 정책: 둘 중 하나의 grip 이라도 잡혀 있는 동안 anchor 유지.
                            # 둘 다 떼면 anchor=None (다음 grip engage 시 새 anchor).
                            # anchor q 는 *last target* 사용 (arm clutch 와 동일한 의미) —
                            # current obs 를 쓰면 PID lag 만큼의 1-tick 백워드 jump 가 가능.
                            if waist_anchor_head is None:
                                waist_anchor_head = head_pose.copy()
                                waist_anchor_q = (
                                    target_waist_q.copy() if target_waist_q is not None
                                    else current_waist_q.copy()
                                )
                            R_now    = head_pose[:3, :3]
                            R_anchor = waist_anchor_head[:3, :3]
                            R_delta  = R_now @ R_anchor.T
                            # 'zyx' euler -> [yaw, pitch, roll] (radians)
                            yaw_d, pitch_d, roll_d = R.from_matrix(R_delta).as_euler('zyx')
                            # G1 waist joint order: [yaw, roll, pitch]
                            delta_waist = np.array([yaw_d, roll_d, pitch_d]) * WAIST_GAIN
                            tw = waist_anchor_q + delta_waist
                            target_waist_q = np.clip(tw, -WAIST_LIMITS, WAIST_LIMITS)
                        else:
                            waist_anchor_head = None
                            waist_anchor_q    = None
                            # grip 미누름 시 마지막 target 을 유지 (freeze).
                            if target_waist_q is None:
                                target_waist_q = current_waist_q.copy()

                        # ---- IK 풀이 -------------------------------------------
                        # head 는 기본 init 자세 고정 (HMD 자세 != 로봇 머리; 머리 모션은 별도 결정)
                        rel_head_pose = T_H_init if T_H_init is not None else np.eye(4)
                        # IK seed 의 waist 부분에 target_waist_q 를 넣어 일관성 유도
                        seed_waist = target_waist_q
                        current_lr_arm_qdml = np.concatenate((seed_waist, current_head, current_arm_q))
                        _ik_t0 = time.perf_counter_ns()
                        sol_q, sol_tauff, _, _ = g1_ik_solver.solve_ik(
                            T_ee_target_l, T_ee_target_r, rel_head_pose, current_lr_arm_qdml,
                        )
                        _ik_solve_ms_window.append((time.perf_counter_ns() - _ik_t0) / 1e6)
                        _ik_cycle_count += 1
                        if _ik_cycle_count % _ik_publish_every == 0:
                            _publish_ik_stats(_ik_solve_ms_window)

                        # NaN guard: 결과가 NaN/inf 면 직전 정상 sol 유지 (action 오염 방지).
                        sol_q, sol_tauff, _ok = _guard_sol(sol_q, sol_tauff)

                        # waist 는 우리가 직접 명령(IK 결과 sol_q[:3] 무시),
                        # head 도 init 고정(0)이므로 IK 결과 대신 0 사용해도 무방하지만
                        # IK 가 head dof 를 안 건드린다는 보장이 없으므로 그대로 두면 0 근처 유지.
                        if sol_q is not None:
                            robot_action_shm.write_data(
                                action_body_ts   =np.int64(time.perf_counter_ns()),
                                action_waist     =target_waist_q,
                                action_waist_tauff=np.zeros(3),
                                action_head      =HEAD_READY_RAD,               # controller 모드: 머리는 ready-pose([2048,1934])로 고정
                                action_arm       =sol_q[5:],
                                action_arm_tauff =sol_tauff[5:],
                            )

                    # ===========================================================
                    # vr_input == "hand": 기존 home-rebase + 절대좌표 fallback
                    # ===========================================================
                    else:
                        if home:
                            # hand-mode home: VR base / IK init / set_home rebase 즉시 적용.
                            # (controller-mode 의 cosine recovery 와 달리 hand-mode 는
                            # 사용자가 손을 자연스럽게 새 위치에서 시작하면 되므로 instant rebase.)
                            base_left_pos  = left_wrist[:3, 3].copy()
                            base_right_pos = right_wrist[:3, 3].copy()
                            base_head_pos  = head_pose[:3, 3].copy()
                            T_L_init, T_R_init, T_H_init = g1_ik_solver.init_pose()
                            set_home = True
                            rm = record_mode_shm.read_data()
                            rm["home"] = False
                            record_mode_shm.write_data(**rm)

                        ik_head_pose = head_pose.copy()
                        ik_head_pose[2, 3] -= 0.6  # head_rmat translation z 보정 (기존 동작 유지)

                        if set_home and base_left_pos is not None:
                            R_L, p_L = left_wrist[:3, :3],  left_wrist[:3, 3]
                            R_R, p_R = right_wrist[:3, :3], right_wrist[:3, 3]
                            R_H, p_H = head_pose[:3, :3],   head_pose[:3, 3]

                            delta_p_L = p_L - base_left_pos
                            delta_p_R = p_R - base_right_pos
                            delta_p_H = p_H - base_head_pos

                            goal_p_L = T_L_init[:3, 3] + delta_p_L
                            goal_p_R = T_R_init[:3, 3] + delta_p_R
                            goal_p_H = T_H_init[:3, 3] + delta_p_H

                            rel_left_wrist  = np.eye(4)
                            rel_left_wrist[:3, :3] = R_L
                            rel_left_wrist[:3,  3] = goal_p_L

                            rel_right_wrist = np.eye(4)
                            rel_right_wrist[:3, :3] = R_R
                            rel_right_wrist[:3,  3] = goal_p_R

                            rel_head_pose   = np.eye(4)
                            rel_head_pose[:3, :3] = R_H
                            rel_head_pose[:3,  3] = goal_p_H

                            current_lr_arm_qdml = np.concatenate((current_waist_q, current_head, current_arm_q))
                            _ik_t0 = time.perf_counter_ns()
                            sol_q, sol_tauff, _, _ = g1_ik_solver.solve_ik(
                                rel_left_wrist, rel_right_wrist, rel_head_pose, current_lr_arm_qdml,
                            )
                            _ik_solve_ms_window.append((time.perf_counter_ns() - _ik_t0) / 1e6)
                            _ik_cycle_count += 1
                            if _ik_cycle_count % _ik_publish_every == 0:
                                _publish_ik_stats(_ik_solve_ms_window)
                        else:
                            current_lr_arm_qdml = np.concatenate((current_waist_q, current_head, current_arm_q))
                            _ik_t0 = time.perf_counter_ns()
                            sol_q, sol_tauff, _, _ = g1_ik_solver.solve_ik(
                                left_wrist, right_wrist, ik_head_pose, current_lr_arm_qdml,
                            )
                            _ik_solve_ms_window.append((time.perf_counter_ns() - _ik_t0) / 1e6)
                            _ik_cycle_count += 1
                            if _ik_cycle_count % _ik_publish_every == 0:
                                _publish_ik_stats(_ik_solve_ms_window)

                        sol_q, sol_tauff, _ok = _guard_sol(sol_q, sol_tauff)  # NaN guard
                        if sol_q is not None:
                            robot_action_shm.write_data(
                                action_body_ts   =np.int64(time.perf_counter_ns()),
                                action_waist     =sol_q[:3],
                                action_waist_tauff=sol_tauff[:3],
                                action_head      =sol_q[3:5],
                                action_arm       =sol_q[5:],
                                action_arm_tauff =sol_tauff[5:],
                            )

                if replay and not replay_demo_init :
                    logger_mp.info(f"[replay init]")

                    replay_idx = int(record_episode_shm.read_data()["replay_idx"].item())
                    task_name = record_task_shm.read_data()["task_name"].item().strip()

                    # 2) chunk 계산 (예: CHUNK_SIZE가 1000이라면)
                    CHUNK_SIZE = 1000
                    chunk_id   = replay_idx // CHUNK_SIZE

                    parquet_path   = os.path.join(
                        "record", task_name,
                        "data", f"chunk-{chunk_id:03d}",
                        f"episode_{replay_idx:06d}.parquet"
                    )
                    logger_mp.info(f"[REPLAY INIT] Loading {parquet_path}")

                    # ── 존재하지 않는 에피소드 방어 ──────────────────────────────
                    # 사용자가 GUI replay 입력창에 수집된 범위를 벗어난 번호(예: 0~2 만
                    # 수집했는데 3)를 넣으면 parquet 파일이 없어 pd.read_parquet 가
                    # FileNotFoundError 를 던지고, 그게 worker_g1_ik / worker_hand_ctrl
                    # 프로세스를 통째로 죽여 수집 세션 전체가 중단됐다(arm/hand 제어 불가).
                    # 파일이 없으면 크래시 대신: 에러 로그 + replay 플래그 클리어 +
                    # set_start 유지(teleop 계속 가능) 로 우아하게 복귀한다.
                    if not os.path.isfile(parquet_path):
                        logger_mp.error(
                            f"[REPLAY] 파일 없음: {parquet_path} — replay_idx={replay_idx} "
                            f"가 수집된 에피소드 범위를 벗어났습니다. replay 취소(teleop 유지)."
                        )
                        rm = record_mode_shm.read_data()
                        rm["replay"] = False
                        rm["done"]   = False
                        record_mode_shm.write_data(**rm)
                        replay_demo_init = False
                        # set_start 는 유지 → teleop 계속 가능. 다음 cycle 에서 teleop 블록 재개.
                        rate.sleep()
                        continue

                    try:
                        df = pd.read_parquet(parquet_path, engine="pyarrow")
                    except Exception as e:
                        logger_mp.error(
                            f"[REPLAY] parquet 로드 실패: {parquet_path} ({e}). replay 취소(teleop 유지)."
                        )
                        rm = record_mode_shm.read_data()
                        rm["replay"] = False
                        rm["done"]   = False
                        record_mode_shm.write_data(**rm)
                        replay_demo_init = False
                        rate.sleep()
                        continue

                    replay_actions   = np.stack(df["action"].to_numpy()).astype(np.float64)
                    replay_length    = replay_actions.shape[0]

                    # ---- 동적 layout 파싱 (수집 인자 무관 자동 대응) ----------------
                    # 저장 측(record_collectors)은 modality_layout.build_state_layout 순서로
                    # action_vec 을 concat 한다. 그 결과가 meta/modality.json 에 그대로 기록됨.
                    # → replay 도 *같은 modality.json* 을 읽어 layout 을 복원하면 어떤 토글
                    #   (waist/head on·off, dex3/inspire) 이든 정확히 분해된다. 하드코딩 X.
                    modality_path = os.path.join("record", task_name, "meta", "modality.json")
                    replay_layout = None
                    try:
                        import json as _json
                        with open(modality_path, "r") as _f:
                            _m = _json.load(_f)
                        replay_layout = layout_from_modality_json(_m)
                        logger_mp.info(f"[REPLAY INIT] layout from modality.json: {replay_layout}")
                    except Exception as e:
                        logger_mp.warning(
                            f"[REPLAY INIT] modality.json 로드 실패({e}). "
                            f"19D(waist3+head2+arm14) 고정 fallback 사용."
                        )

                    # body action (waist3 + head2 + arm14 = 19D) 을 layout 기반으로 재구성.
                    # waist/head 가 off 인 데이터면 그 부분은 0 으로 채운다(로봇은 해당 관절을
                    # 별도 제어/고정하므로 0 action 이 안전한 중립값).
                    if replay_layout is not None:
                        names = [nm for nm, _ in replay_layout]
                        replay_body_waist = np.zeros((replay_length, 3), dtype=np.float64)
                        replay_body_head  = np.zeros((replay_length, 2), dtype=np.float64)
                        replay_body_larm  = None
                        replay_body_rarm  = None
                        for k in range(replay_length):
                            parts = split_state_vec(replay_actions[k], replay_layout)
                            if k == 0:
                                # left_arm/right_arm 은 항상 존재(layout 규칙). 차원 확인.
                                la_dim = parts['left_arm'].size
                                ra_dim = parts['right_arm'].size
                                replay_body_larm = np.zeros((replay_length, la_dim))
                                replay_body_rarm = np.zeros((replay_length, ra_dim))
                            if 'waist' in parts: replay_body_waist[k] = parts['waist']
                            if 'head'  in parts: replay_body_head[k]  = parts['head']
                            replay_body_larm[k] = parts['left_arm']
                            replay_body_rarm[k] = parts['right_arm']
                        # arm14 = left_arm(7) + right_arm(7)
                        replay_body_arm = np.concatenate([replay_body_larm, replay_body_rarm], axis=1)
                        # g1 body action 19D = waist3 + head2 + arm14 (worker_g1_ctrl 가 받는 형식)
                        replay_g1_data = np.concatenate(
                            [replay_body_waist, replay_body_head, replay_body_arm], axis=1
                        )
                    else:
                        # fallback: 옛 19D 고정 가정 (33D 레이아웃 데이터 호환)
                        replay_g1_data = replay_actions[:, :19]

                    replay_demo_init = True
                    replay_frame_idx = 0
                    logger_mp.info(f"[REPLAY INIT] Loaded {replay_length} frames; g1_data shape = {replay_g1_data.shape}")

                    # (선택) 데이터에 'timestamp' 컬럼이 있으면 실제 간격을 그대로 사용
                    if "timestamp" in df.columns:
                        ts = df["timestamp"].to_numpy().astype(np.float64)
                        ts = ts - ts[0]                    # 0초부터 시작하는 상대시간
                        _replay_rel_ts = ts
                        _replay_start_wall = time.perf_counter()
                    else:
                        _replay_rel_ts = None
                        _replay_start_wall = None

                    # 고정 20Hz 재생용 타이머 (첫 프레임은 즉시 송출)
                    _replay_next_t = time.perf_counter()
                    _last_written_idx = -1



                if replay and replay_demo_init:

                    now = time.perf_counter()

                    if _replay_rel_ts is not None:
                        # 3-1) 기록 타임스탬프 기반 (가변 간격): 현재 경과시간에 해당하는 인덱스를 선택
                        elapsed = now - _replay_start_wall
                        target_idx = int(np.searchsorted(_replay_rel_ts, elapsed, side="right")) - 1
                        target_idx = max(0, min(target_idx, replay_length - 1))
                        replay_frame_idx = target_idx
                    else:
                        # 3-2) 고정 20Hz: 다음 예약 시각이 되면 한 프레임만 전진
                        if now >= _replay_next_t:
                            if replay_frame_idx < replay_length - 1:
                                replay_frame_idx += 1
                            # 다음 예약 시각 업데이트(드리프트 최소화: 지났으면 여러 번 보정)
                            while _replay_next_t <= now:
                                _replay_next_t += _replay_period
                        # 아직 시간이 안 됐으면 같은 프레임 유지
                    
                    # 현재(혹은 유지) 프레임을 취득
                    idx = replay_frame_idx
                    sol_q = replay_g1_data[idx]

                    robot_action_shm.write_data(
                        action_body_ts=np.int64(time.perf_counter_ns()),
                        action_waist=sol_q[:3],
                        action_arm=sol_q[5:],
                        action_head=sol_q[3:5]
                    )

                    # 진행률/로그는 인덱스가 변했을 때만 업데이트(로그 스팸 방지)
                    if idx != _last_written_idx:
                        logging_percentages = int(idx * 100 / replay_length)
                        record_episode_shm.write_data(logging_progress=logging_percentages)
                        # logger_mp.info(f" percentages : {logging_percentages} replay_frame_idx : {idx} replay_length : {replay_length}")
                        _last_written_idx = idx

                    # 종료 조건
                    if replay_frame_idx >= replay_length - 1:
                        # 타임스탬프 모드면 마지막 타임스탬프를 지난 뒤 종료, 아니면 즉시 종료
                        if (_replay_rel_ts is None) or (now - _replay_start_wall >= _replay_rel_ts[-1]):
                            record_mode_data = record_mode_shm.read_data()
                            record_mode_data["replay"] = False
                            record_mode_data["start"]  = False
                            # ── 통제권 복귀 + 자동 recovery 수정 ──────────────────
                            # 과거: shared_event['set_start'].clear() 로 RUN 을 벗어나
                            #   (RUN→PAUSE→WAIT_START) g1_ctrl/g1_ik 가 robot_action_shm
                            #   read/write 를 모두 멈췄다 → replay 후 controller 로 arm 을
                            #   다시 조작하려면 GUI Start 를 눌러야만 했다.
                            # 수정 핵심 2가지:
                            #  (1) set_start 를 유지 → RUN 유지 → replay=False 이므로 위의
                            #      teleop 블록(set_start.is_set() and not replay and not
                            #      deploy)이 자동 재개된다. GUI on_replay_episode 가 replay
                            #      시작 시 set_start.set() 하므로(ui L836) 흐름상 정합.
                            #  (2) home=True 로 설정 → 다음 cycle 에서 recovery state machine
                            #      (위 'if home and not _recovery_active')이 트리거되어 arm 이
                            #      replay 마지막 자세에서 ready-pose 로 3초 cosine ease 로
                            #      부드럽게 복귀한다. recovery 종료 시 anchor/grip 가 모두
                            #      리셋되고 home=False 로 클리어되므로(위 ease 종료 블록),
                            #      사용자가 grip 을 다시 잡는 순간 새 anchor 가 캡처되어
                            #      곧바로 teleop 으로 이어서 데이터 수집을 재개할 수 있다.
                            #      (replay 끝 자세에서 급격히 튀지 않도록 하는 안전장치.)
                            record_mode_data["home"] = True
                            record_mode_shm.write_data(**record_mode_data)
                            replay_demo_init = False
                            # set_start 는 유지 (clear 하지 않음).

                            logger_mp.info("[REPLAY DONE] Finished all frames "
                                           "→ set_start 유지 + home recovery 트리거 "
                                           "(ready-pose 복귀 후 grip 으로 teleop 재개).")

                # deploy 모드: 이 워커는 IK 를 수행하지 않고, evaluate.py(외부 conda env)
                # 가 ROBOT_ACTION SHM 에 action 을 직접 publish 한다. 여기서는 단순히
                # rate.sleep 만 수행하고 home 등의 SHM 토글은 그쪽(평가 워커/ UI) 책임.
                # (과거 주석처리된 home flag reset 코드는 evaluate.py 도입 후 의미 없어 제거.)

                rate.sleep()

            elif state is State.PAUSE:
                time.sleep(0.01)

    finally:
        logger_mp.info("[G1_IK] 종료 및 SHM 정리")
        television_shm.worker_close(); robot_action_shm.worker_close(); robot_obs_shm.worker_close()
        record_mode_shm.worker_close(); freq_shm.worker_close()
        record_episode_shm.worker_close(); record_task_shm.worker_close()
        quest_controller_shm.worker_close()