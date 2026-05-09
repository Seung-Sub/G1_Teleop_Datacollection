import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.state import State, EventsSnapshot, next_state
from utils.rate import Rate
from utils.mat_tool import fast_mat_inv

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


def _events_snapshot(shared_event, g1_initialized: bool) -> EventsSnapshot:
    return EventsSnapshot(
        shutdown = shared_event['shutdown'].is_set(),
        emergency = shared_event['emergency'].is_set(),
        g1_ready = g1_initialized ,
        hand_ready = shared_event['set_hand'].is_set(),
        start = shared_event['set_start'].is_set(),
        select_pressed = False
    )

def worker_g1_ik(shared_event, shm_name, shared_lock, vr_input="hand"):
    """vr_input: 'hand' (current behaviour) or 'controller' (Phase 1-C clutch IK)."""

    #Set SharedMemory
    television_shm       = SharedMemoryManager(TELEVISION,             shared_lock["television_lock"],       shm_name["television_shm"])
    freq_shm             = SharedMemoryManager(WORKER_FREQ,            shared_lock["freq_lock"],             shm_name["freq_shm"])
    record_mode_shm      = SharedMemoryManager(RECORD_MODE_LAYOUT,     shared_lock["record_lock"],           shm_name["record_mode_shm"])
    record_episode_shm   = SharedMemoryManager(RECORD_EPISODE_LAYOUT,  shared_lock["record_lock"],           shm_name["record_episode_shm"])
    record_task_shm      = SharedMemoryManager(RECORD_TASK_LAYOUT,     shared_lock["record_lock"],           shm_name["record_task_shm"])
    robot_action_shm     = SharedMemoryManager(ROBOT_ACTION,           shared_lock["robot_action_lock"],     shm_name["robot_action_shm"])
    robot_obs_shm        = SharedMemoryManager(ROBOT_OBS,              shared_lock["robot_obs_lock"],        shm_name["robot_obs_shm"])
    quest_controller_shm = SharedMemoryManager(QUEST_CONTROLLER,       shared_lock["quest_controller_lock"], shm_name["quest_controller_shm"])

    # 50Hz 주기로 실행 
    rate = Rate(50.0)

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
    # Ready 버튼 edge detection
    prev_right_a_btn = False

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

                    if home:
                        # 양쪽 모드 공통: VR/EE/head anchor 재설정
                        base_left_pos  = left_wrist[:3, 3].copy()
                        base_right_pos = right_wrist[:3, 3].copy()
                        base_head_pos  = head_pose[:3, 3].copy()
                        T_L_init, T_R_init, T_H_init = g1_ik_solver.init_pose()
                        set_home = True
                        # controller-mode 상태 초기화
                        T_ee_target_l = T_L_init.copy()
                        T_ee_target_r = T_R_init.copy()
                        T_ctrl_anchor_l = T_ctrl_anchor_r = None
                        T_ee_anchor_l = T_ee_anchor_r   = None
                        waist_anchor_head = None
                        waist_anchor_q    = None
                        target_waist_q    = current_waist_q.copy()
                        prev_grip_l = prev_grip_r = False
                        prev_right_a_btn = False

                        record_mode_data = record_mode_shm.read_data()
                        record_mode_data["home"] = False
                        record_mode_shm.write_data(**record_mode_data)

                    # ===========================================================
                    # vr_input == "controller": clutch (grip-hold) + waist + ready
                    # ===========================================================
                    if vr_input == "controller":
                        ctrl_data = quest_controller_shm.read_data()

                        T_ctrl_l_now = ctrl_data["left_ctrl_mat"]
                        T_ctrl_r_now = ctrl_data["right_ctrl_mat"]

                        grip_l = float(ctrl_data["left_squeeze"])  >= GRIP_THRESH
                        grip_r = float(ctrl_data["right_squeeze"]) >= GRIP_THRESH

                        # ---- ready button (right A) edge detection -------------
                        right_a = float(ctrl_data["right_buttons"][0]) >= BUTTON_THRESH
                        if right_a and not prev_right_a_btn:
                            logger_mp.info("[G1_IK] Right-A pressed -> trigger HOME (ready pose).")
                            rm = record_mode_shm.read_data()
                            rm["home"]  = True
                            rm["start"] = False
                            record_mode_shm.write_data(**rm)
                            shared_event['set_start'].clear()
                        prev_right_a_btn = right_a

                        # ---- left arm clutch -----------------------------------
                        if T_ee_target_l is None:
                            T_ee_target_l = T_L_init.copy() if T_L_init is not None else np.eye(4)
                        if grip_l:
                            if not prev_grip_l:
                                T_ctrl_anchor_l = T_ctrl_l_now.copy()
                                T_ee_anchor_l   = T_ee_target_l.copy()
                                logger_mp.info("[G1_IK] LEFT grip ENGAGE")
                            delta_l = T_ctrl_l_now @ fast_mat_inv(T_ctrl_anchor_l)
                            T_ee_target_l = delta_l @ T_ee_anchor_l
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
                                logger_mp.info("[G1_IK] RIGHT grip ENGAGE")
                            delta_r = T_ctrl_r_now @ fast_mat_inv(T_ctrl_anchor_r)
                            T_ee_target_r = delta_r @ T_ee_anchor_r
                        else:
                            if prev_grip_r:
                                logger_mp.info("[G1_IK] RIGHT grip RELEASE -> freeze EE target")
                        prev_grip_r = grip_r

                        # ---- waist clutch (HMD delta -> waist target) ----------
                        # grip 한쪽 이상 누른 동안만 HMD 변화를 waist 에 반영.
                        if grip_l or grip_r:
                            if waist_anchor_head is None:
                                waist_anchor_head = head_pose.copy()
                                waist_anchor_q    = current_waist_q.copy()
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
                            # grip 미누름 시 현재 waist q 를 target 으로 freeze
                            if target_waist_q is None:
                                target_waist_q = current_waist_q.copy()

                        # ---- IK 풀이 -------------------------------------------
                        # head 는 기본 init 자세 고정 (HMD 자세 != 로봇 머리; 머리 모션은 별도 결정)
                        rel_head_pose = T_H_init if T_H_init is not None else np.eye(4)
                        # IK seed 의 waist 부분에 target_waist_q 를 넣어 일관성 유도
                        seed_waist = target_waist_q
                        current_lr_arm_qdml = np.concatenate((seed_waist, current_head, current_arm_q))
                        sol_q, sol_tauff, _, _ = g1_ik_solver.solve_ik(
                            T_ee_target_l, T_ee_target_r, rel_head_pose, current_lr_arm_qdml,
                        )

                        # waist 는 우리가 직접 명령(IK 결과 sol_q[:3] 무시),
                        # head 도 init 고정(0)이므로 IK 결과 대신 0 사용해도 무방하지만
                        # IK 가 head dof 를 안 건드린다는 보장이 없으므로 그대로 두면 0 근처 유지.
                        robot_action_shm.write_data(
                            action_waist     =target_waist_q,
                            action_waist_tauff=np.zeros(3),
                            action_head      =np.zeros(2),                 # controller 모드: 머리 정면 고정
                            action_arm       =sol_q[5:],
                            action_arm_tauff =sol_tauff[5:],
                        )

                    # ===========================================================
                    # vr_input == "hand": 기존 home-rebase + 절대좌표 fallback
                    # ===========================================================
                    else:
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
                            sol_q, sol_tauff, _, _ = g1_ik_solver.solve_ik(
                                rel_left_wrist, rel_right_wrist, rel_head_pose, current_lr_arm_qdml,
                            )
                        else:
                            current_lr_arm_qdml = np.concatenate((current_waist_q, current_head, current_arm_q))
                            sol_q, sol_tauff, _, _ = g1_ik_solver.solve_ik(
                                left_wrist, right_wrist, ik_head_pose, current_lr_arm_qdml,
                            )

                        robot_action_shm.write_data(
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

                    df = pd.read_parquet(parquet_path, engine="pyarrow")

                    replay_actions   = np.stack(df["action"].to_numpy()).astype(np.float64)
                    replay_length    = replay_actions.shape[0]

                    replay_g1_data = replay_actions[:, :19]   # (N,12)

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
                            record_mode_data["start"] = False
                            record_mode_shm.write_data(**record_mode_data)
                            replay_demo_init = False
                            shared_event['set_start'].clear()

                            logger_mp.info("[REPLAY DONE] Finished all frames")

                if deploy : 
                    if home:
                        # record_mode_data = record_mode_shm.read_data()
                        # record_mode_data["deploy"] = False
                        # record_mode_data["home"] = False
                        # record_mode_data["start"] = False
                        # record_mode_shm.write_data(**record_mode_data)
                        # # shared_event['set_start'].clear()
                        continue


                rate.sleep()

            elif state is State.PAUSE:
                time.sleep(0.01)

    finally:
        logger_mp.info("[G1_IK] 종료 및 SHM 정리")
        television_shm.worker_close(); robot_action_shm.worker_close(); robot_obs_shm.worker_close()
        record_mode_shm.worker_close(); freq_shm.worker_close()
        record_episode_shm.worker_close(); record_task_shm.worker_close()
        quest_controller_shm.worker_close()