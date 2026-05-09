import os
import time
import numpy as np

from utils.state import State, EventsSnapshot, next_state
from utils.rate import Rate

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import TELEVISION, RECORD_MODE_LAYOUT, WORKER_FREQ, RECORD_EPISODE_LAYOUT, RECORD_TASK_LAYOUT, ROBOT_ACTION, ROBOT_OBS

from g1_control.g1_ik import G1_29_ArmIK

import pandas as pd

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


def _events_snapshot(shared_event, g1_initialized: bool) -> EventsSnapshot:
    return EventsSnapshot(
        shutdown = shared_event['shutdown'].is_set(),
        emergency = shared_event['emergency'].is_set(),
        g1_ready = g1_initialized ,
        hand_ready = shared_event['set_hand'].is_set(),
        start = shared_event['set_start'].is_set(),
        select_pressed = False
    )

def worker_g1_ik(shared_event, shm_name, shared_lock, ctrl_mode):

    #Set SharedMemory
    television_shm = SharedMemoryManager(TELEVISION, shared_lock["television_lock"], shm_name["television_shm"])
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])
    record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_name["record_mode_shm"])
    record_episode_shm = SharedMemoryManager(RECORD_EPISODE_LAYOUT, shared_lock["record_lock"], shm_name["record_episode_shm"])
    record_task_shm = SharedMemoryManager(RECORD_TASK_LAYOUT, shared_lock["record_lock"], shm_name["record_task_shm"])
    robot_action_shm = SharedMemoryManager(ROBOT_ACTION, shared_lock["robot_action_lock"], shm_name["robot_action_shm"])
    robot_obs_shm = SharedMemoryManager(ROBOT_OBS, shared_lock["robot_obs_lock"], shm_name["robot_obs_shm"])

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

    first_loop = True

    state = State.WAIT_CONNECT
    logger_mp.info("[G1_IK] FSM start: WAIT_CONNECT")

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
                    
                    head_pose        = vr_data["head_rmat"]
                    left_wrist       = vr_data["left_wrist_mat"]
                    right_wrist      = vr_data["right_wrist_mat"]
                    
                    ik_head_pose = head_pose.copy()

                    translation = ik_head_pose[0:3, 3]

                    # 전역 Z축으로 −0.6 이동
                    translation[2] -= 0.6

                    # 업데이트
                    ik_head_pose[0:3, 3] = translation

                    robot_obs = robot_obs_shm.read_data()

                    # 현재 로봇 상태 읽기 
                    current_waist_q  = robot_obs["obs_waist"]
                    current_head = robot_obs["obs_head"]
                    current_arm_q  = robot_obs["obs_arm"]

                    if home:
                        # 현재 VR 위치를 기준점으로 저장
                        base_left_pos  = left_wrist[:3, 3].copy()
                        base_right_pos = right_wrist[:3, 3].copy()
                        base_head_pos  = head_pose[:3, 3].copy()

                        # 로봇의 초기 자세도 저장
                        T_L_init, T_R_init, T_H_init = g1_ik_solver.init_pose()

                        set_home = True

                        record_mode_data = record_mode_shm.read_data()
                        record_mode_data["home"] = False
                        record_mode_shm.write_data(**record_mode_data)

                
                    if set_home and base_left_pos is not None:
                        # VR 현재 위치 
                        R_L, p_L = left_wrist[:3, :3],  left_wrist[:3, 3]
                        R_R, p_R = right_wrist[:3, :3], right_wrist[:3, 3]
                        R_H, p_H = head_pose[:3, :3],   head_pose[:3, 3]

                        # 4) 기준 위치 대비 translation delta
                        delta_p_L = p_L - base_left_pos
                        delta_p_R = p_R - base_right_pos
                        delta_p_H = p_H - base_head_pos

                        # 5) 로봇 홈 EE 위치에 delta를 더해 goal translation 생성
                        goal_p_L = T_L_init[:3, 3] + delta_p_L
                        goal_p_R = T_R_init[:3, 3] + delta_p_R
                        goal_p_H = T_H_init[:3, 3] + delta_p_H

                        # 6) 최종 target transform 조립 (orientation은 원본 human R 사용)
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
                        sol_q, sol_tauff, pelvis_height, torso_quat = g1_ik_solver.solve_ik(rel_left_wrist, rel_right_wrist, rel_head_pose, current_lr_arm_qdml)

                    else:
                        # 절대 좌표 IK: VR 원본 데이터 직접 사용
                        current_lr_arm_qdml = np.concatenate((current_waist_q, current_head, current_arm_q))
                        sol_q, sol_tauff, pelvis_height, torso_quat = g1_ik_solver.solve_ik(left_wrist, right_wrist, ik_head_pose, current_lr_arm_qdml)

                    # IK를 푼 결과를 action에 전달
                    robot_action_shm.write_data(
                        action_waist=sol_q[:3],               # 허리 3개
                        action_waist_tauff=sol_tauff[:3],
                        action_head=sol_q[3:5],               # 머리 2개
                        action_arm=sol_q[5:],                 # 팔 14개
                        action_arm_tauff=sol_tauff[5:],
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