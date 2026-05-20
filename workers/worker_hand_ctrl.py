import os
import time
import numpy as np

from utils.state import State, EventsSnapshot, next_state
from utils.rate import Rate

# hand 컨트롤러는 hand 인자에 따라 lazy import — DEX3 는 unitree_hg DDS msg
# 모듈을 필요로 하므로 Inspire 만 쓰는 환경에서도 import 가 깨지지 않게.
from multiprocessing import shared_memory, Array, Lock
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import TELEVISION, WORKER_FREQ, RECORD_MODE_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_TASK_LAYOUT, ROBOT_ACTION, ROBOT_OBS
import traceback
import pandas as pd

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


def _events_snapshot(shared_event, hand_initialized: bool) -> EventsSnapshot:
    return EventsSnapshot(
        shutdown = shared_event['shutdown'].is_set(),
        emergency = shared_event['emergency'].is_set(),
        g1_ready = shared_event['set_g1'].is_set(),
        hand_ready = hand_initialized,
        start = shared_event['set_start'].is_set(),
        select_pressed = False
    )

def worker_hand_ctrl(shared_event, shm_name, shared_lock,
                     hand="inspire",
                     vr_input="hand", thumb_bend=0.5, thumb_yaw=0.5,
                     tactile='off'):
    """hand: 'inspire' (6+6 DOF, Inspire DDS) or 'dex3' (7+7 DOF, unitree_hg DDS).
    vr_input='hand' uses dex_retargeting from VR keypoints; vr_input='controller'
    uses Quest3 controller trigger as a grasp/release toggle, with thumb_bend/thumb_yaw
    (0..1) selecting a pre-set thumb pose.
    tactile: 'off' (default) | 'on'. Phase K8: on 시 DEX3 가 HandState.press_sensor_state
    length 를 로깅. 실제 SHM 저장은 SDK 실 device sequence length 확정 후 후속 작업."""
    #Set SharedMemory
    television_shm = SharedMemoryManager(TELEVISION, shared_lock["television_lock"], shm_name["television_shm"])
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])
    record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_name["record_mode_shm"])
    record_episode_shm = SharedMemoryManager(RECORD_EPISODE_LAYOUT, shared_lock["record_lock"], shm_name["record_episode_shm"])
    record_task_shm = SharedMemoryManager(RECORD_TASK_LAYOUT, shared_lock["record_lock"], shm_name["record_task_shm"])
    robot_action_shm = SharedMemoryManager(ROBOT_ACTION, shared_lock["robot_action_lock"], shm_name["robot_action_shm"])
    robot_obs_shm = SharedMemoryManager(ROBOT_OBS, shared_lock["robot_obs_lock"], shm_name["robot_obs_shm"])   

    rate = Rate(50.0)

    # G1/Hand 컨트롤러를 아직 생성하지 않은 상태를 추적하는 플래그
    hand_initialized = False

    # 각각 컨트롤러 객체를 저장할 변수 (None으로 초기화)
    hand_ctrl    = None

    replay_demo_init = False
    replay_actions = None   # action 배열 (N×31)
    replay_hand_data   = None   # 핸드 관련 action 배열 (N×12)

    replay_length = 0       # 전체 프레임 수
    replay_frame_idx = 0    # 현재 재생 중인 프레임 인덱스

    REPLAY_HZ = 20.0
    _replay_period = 1.0 / REPLAY_HZ
    _replay_next_t = None           # 다음 프레임을 전진시킬 절대시각
    _replay_start_wall = None       # (선택) 타임스탬프 재생용 시작 벽시각
    _replay_rel_ts = None           # (선택) 데이터셋 상대 타임스탬프 배열
    _last_written_idx = -1          # (선택) 진행률/로그 중복 방지

    state = State.WAIT_CONNECT
    logger_mp.info("[Hand_Ctrl] FSM start: WAIT_CONNECT")

    try:
        while True:
            # 상태 전이
            state_next = next_state(state, _events_snapshot(shared_event, hand_initialized))
            if state_next != state:
                logger_mp.info(f"[Hand_Ctrl] {state.name} -> {state_next.name}")
            state = state_next

            if state is State.EXIT:
                break

            if state is State.WAIT_CONNECT:
                # set_g1이 올라오면 IK 초기화 1회 시도
                if shared_event['set_hand'].is_set() and not hand_initialized:
                    try:
                        left_hand_array  = Array('d', 5 * 3, lock=True)
                        right_hand_array = Array('d', 5 * 3, lock=True)
                        dual_hand_data_lock = Lock()

                        if hand == "inspire":
                            from hand_control.robot_hand_inspire import Inspire_Controller
                            # Inspire: 양손 6+6 = 12 motors, normalized 0..1 q
                            dual_hand_state_array  = Array('d', 12, lock=False)
                            dual_hand_action_array = Array('d', 12, lock=False)
                            hand_ctrl = Inspire_Controller(
                                shm_name, shared_lock, left_hand_array, right_hand_array,
                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array,
                                fps=100.0, Unit_Test=False,
                                vr_input=vr_input, thumb_bend=thumb_bend, thumb_yaw=thumb_yaw,
                            )
                        elif hand == "dex3":
                            from hand_control.robot_hand_dex3 import Dex3_Controller
                            # DEX3: 양손 7+7 = 14 motors, raw radian q
                            dual_hand_state_array  = Array('d', 14, lock=False)
                            dual_hand_action_array = Array('d', 14, lock=False)
                            hand_ctrl = Dex3_Controller(
                                shm_name, shared_lock, left_hand_array, right_hand_array,
                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array,
                                fps=100.0, Unit_Test=False,
                                vr_input=vr_input, thumb_bend=thumb_bend, thumb_yaw=thumb_yaw,
                                collect_tactile=(tactile == 'on'),  # Phase K8
                            )
                        else:
                            raise ValueError(f"Unsupported hand hardware: {hand}")

                        hand_initialized = True
                        logger_mp.info(f"[Hand_Ctrl] Hand controller initialized (hand={hand}, vr_input={vr_input}).")
                    except Exception as e:
                        logger_mp.exception("[Hand_Ctrl] Hand init failed: %s", e)
                        shared_event['emergency'].set()
                time.sleep(0.01)

            elif state is State.WAIT_START:
                # START 대기
                time.sleep(0.01)

            elif state is State.RUN:

                mode = record_mode_shm.read_data()
                home, replay, deploy = mode["home"], mode["replay"], mode["deploy"]

                freq_shm.write_data(hand_freq=rate.tick_hz())

                if shared_event['set_start'].is_set() and not replay and not deploy:
                    data = television_shm.read_data()

                    left_hand       = data["left_hand"]
                    right_hand      = data["right_hand"]

                    left_hand_array[:]  = left_hand.flatten()
                    right_hand_array[:] = right_hand.flatten()

                    # obs_hand / action_hand SHM 은 14D. 양손 DOF 가 작은 경우 (Inspire 12D) pad.
                    obs14    = np.zeros(14, dtype=np.float64)
                    act14    = np.zeros(14, dtype=np.float64)
                    n        = len(dual_hand_state_array)
                    obs14[:n] = dual_hand_state_array[:]
                    act14[:n] = dual_hand_action_array[:]
                    # Phase K4 (P0-1.4): obs 는 DDS 수신 시각, action 은 publish 시각.
                    # 둘이 다른 시점/출처라서 같은 ts 를 공유하면 안 됨.
                    try:
                        ts_obs_hand = int(hand_ctrl.get_hand_state_recv_ts())
                    except AttributeError:
                        ts_obs_hand = time.perf_counter_ns()  # 구버전 controller fallback
                    if ts_obs_hand <= 0:
                        # 아직 첫 state 수신 전 — write skip (가짜 0 ts 방지)
                        pass
                    else:
                        ts_act_hand = time.perf_counter_ns()
                        robot_obs_shm.write_data(obs_hand=obs14, obs_hand_ts=np.int64(ts_obs_hand))
                        robot_action_shm.write_data(action_hand=act14, action_hand_ts=np.int64(ts_act_hand))


                if replay and not replay_demo_init :
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

                    # 4) DataFrame 로드
                    df = pd.read_parquet(parquet_path, engine="pyarrow")
                    # 5) NumPy 배열로 변환
                    #    - observation/state도 필요하면 같은 식으로 df["observation.state"]
                    replay_actions   = np.stack(df["action"].to_numpy()).astype(np.float64)
                    replay_length    = replay_actions.shape[0]
                    # 6) 핸드 데이터만 분리 (뒤 12개)
                    #    replay_actions shape = (N, 19 + 12)
                    replay_hand_data = replay_actions[:, 19:]   # (N,12)

                    replay_demo_init = True
                    replay_frame_idx = 0
                    logger_mp.info(f"[REPLAY INIT] Loaded {replay_length} frames; hand_data shape = {replay_hand_data.shape}")
                    if "timestamp" in df.columns:
                        ts = df["timestamp"].to_numpy().astype(np.float64)
                        ts = ts - ts[0]                 # 0초 기준 상대시간
                        _replay_rel_ts = ts
                        _replay_start_wall = time.perf_counter()
                    else:
                        _replay_rel_ts = None
                        _replay_start_wall = None

                    # 고정 20Hz 재생을 위한 다음 예약 시각 (첫 프레임은 즉시 송출)
                    _replay_next_t = time.perf_counter()
                    _last_written_idx = -1


                if replay and replay_demo_init:
                    now = time.perf_counter()

                    # --- 프레임 인덱스 결정: 타임스탬프 기반 또는 고정 20Hz ---
                    if _replay_rel_ts is not None:
                        # 기록된 타임스탬프 기반 (가변 간격)
                        elapsed = now - _replay_start_wall
                        target_idx = int(np.searchsorted(_replay_rel_ts, elapsed, side="right")) - 1
                        target_idx = max(0, min(target_idx, replay_length - 1))
                        replay_frame_idx = target_idx
                    else:
                        # 고정 20Hz: 예약 시각이 지났을 때만 한 프레임 전진
                        if now >= _replay_next_t:
                            if replay_frame_idx < replay_length - 1:
                                replay_frame_idx += 1
                            # 지연 누적분 보정(드리프트 억제)
                            while _replay_next_t <= now:
                                _replay_next_t += _replay_period
                        # 아직 시간이 안 됐으면 같은 프레임 유지

                    idx = replay_frame_idx
                    hand_vec = replay_hand_data[idx]  # length 12 (inspire) or 14 (dex3)

                    # hand 종류별 분할.
                    if hand == "inspire":
                        left_q_target  = hand_vec[:6]
                        right_q_target = hand_vec[6:12]
                    else:  # dex3
                        left_q_target  = hand_vec[:7]
                        right_q_target = hand_vec[7:14]

                    hand_ctrl.ctrl_dual_hand(left_q_target=left_q_target,
                                            right_q_target=right_q_target)

                    # 프레임 전진은 위의 타이머/타임스탬프 분기에서만 수행.
                    # (과거 코드에 있던 unconditional `replay_frame_idx += 1` 은 body(worker_g1_ik)
                    #  20Hz vs hand 50Hz 비동기 버그의 원인이라 제거.)


                if deploy:
                    # deploy 모드: evaluate.py(외부 inference) 가 robot_action_shm.action_hand
                    # 에 양손 q 를 직접 publish. hand 종류에 따라 슬라이싱이 다르다.
                    robot_dict = robot_action_shm.read_data()
                    hand_action = robot_dict["action_hand"]
                    if hand == "inspire":
                        # Inspire 는 0..1 normalized q
                        hand_action = np.clip(hand_action, a_min=None, a_max=1.0)
                        left_q_target  = hand_action[:6]
                        right_q_target = hand_action[6:12]
                    else:  # dex3
                        left_q_target  = hand_action[:7]
                        right_q_target = hand_action[7:14]
                    hand_ctrl.ctrl_dual_hand(left_q_target=left_q_target,
                                            right_q_target=right_q_target)

                # 매 사이클 obs_hand SHM 갱신 — DDS 수신 시각 (Phase K4) 으로 ts.
                obs14 = np.zeros(14, dtype=np.float64)
                n     = len(dual_hand_state_array)
                obs14[:n] = dual_hand_state_array[:]
                try:
                    ts_obs_hand = int(hand_ctrl.get_hand_state_recv_ts())
                except AttributeError:
                    ts_obs_hand = time.perf_counter_ns()
                if ts_obs_hand > 0:
                    robot_obs_shm.write_data(
                        obs_hand=obs14,
                        obs_hand_ts=np.int64(ts_obs_hand),
                    )
            
                rate.sleep()

            elif state is State.PAUSE:
                time.sleep(0.01)


    finally:

        logger_mp.info("[Hand_Ctrl] 종료 신호 수신. 종료합니다.")
        television_shm.worker_close()
        freq_shm.worker_close()
        record_mode_shm.worker_close()
        record_episode_shm.worker_close()
        record_task_shm.worker_close()
        robot_obs_shm.worker_close()
        robot_action_shm.worker_close()