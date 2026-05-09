# worker_record.py

import os

from utils.rate import Rate
from utils import ParquetSink, VideoSink

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    CAMERA, RECORD_TASK_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_MODE_LAYOUT,
    WORKER_FREQ, ROBOT_ACTION, ROBOT_OBS, WORKSPACE_MASK,
    TELEOP_CONFIG, CAMERA_MAPPING_INV,
)

import numpy as np
import cv2

import glob  

import logging_mp
logger_mp = logging_mp.get_logger(__name__)



from enum import Enum
class State(Enum):
    WAIT_FOR_SET=1; IDLE=2; RECORDING=3

def read_mode_snapshot(shm):
    m = shm.read_data()
    return bool(m["start"]), bool(m["reset"]), bool(m["replay"]), m

def worker_record(shared_event, shm_name, shared_lock):
    """
    SharedMemory(RECORD_TASK_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_MODE_LAYOUT)를 이용해
    다음과 같은 상태 기계(State Machine)로 동작합니다.

    1) WAIT_FOR_SET:
       - UI에서 SET을 눌러 task_name이 "0" 또는 빈 문자열이 아닌 값으로 바뀔 때까지 대기.
       - num_episodes, episode_len은 0이어도 무시됨.

    2) IDLE:
       - START나 RESET 같은 입력을 기다리는 대기 모드.
       - mode_data["start"]가 True로 세팅되면 한 에피소드 RECORDING으로 진입. (단발성)
       - mode_data["reset"]을 눌러도 IDLE에서는 플래그만 False로 리셋.
       - replay 플래그는 어떤 상태에서든 최우선 처리.

    3) RECORDING:
       - 매 주기(50Hz)마다 "로그 저장"(여기서는 콘솔 출력) → frame_count 증가.
       - episode_len * 50 프레임이 쌓이면 해당 에피소드 저장 완료 → ep_idx += 1 → IDLE로 복귀.
       - 중간에 mode_data["reset"]이 True가 되면 "현재 에피소드만 취소" → IDLE로 복귀 (다음 START 기다림).
       - replay 요청이 있으면 즉시 로그만 남기고 플래그를 False로 리셋.

    replay 동작은 "현재는 로그만 찍고, 플래그를 False로 리셋"합니다. 실제 파일 삭제/재생은 추후 구현.
    """

    # 공유 메모리 객체들을 리스트로 관리하여 정리 용이하게 함
    shm_objects = []

    def create_shm(name, layout, lock_key):
        """공유 메모리 객체 생성 및 리스트에 추가"""
        try:
            shm = SharedMemoryManager(layout, shared_lock[lock_key], shm_name[name])
            shm_objects.append(shm)
            return shm
        except Exception as e:
            logger_mp.error(f"[Record] 공유 메모리 생성 실패 ({name}): {e}")
            return None

    # 기본 공유 메모리 객체들 생성
    record_task_shm    = create_shm("record_task_shm",    RECORD_TASK_LAYOUT,    "record_lock")
    record_episode_shm = create_shm("record_episode_shm", RECORD_EPISODE_LAYOUT, "record_lock")
    record_mode_shm    = create_shm("record_mode_shm",    RECORD_MODE_LAYOUT,    "record_lock")
    camera_shm         = create_shm("camera_shm",         CAMERA,                "camera_lock")
    robot_action_shm   = create_shm("robot_action_shm",   ROBOT_ACTION,          "robot_action_lock")
    robot_obs_shm      = create_shm("robot_obs_shm",      ROBOT_OBS,             "robot_obs_lock")
    freq_shm           = create_shm("freq_shm",           WORKER_FREQ,           "freq_lock")
    workspace_mask_shm = create_shm("workspace_mask_shm", WORKSPACE_MASK,        "workspace_mask_lock")
    teleop_config_shm  = create_shm("teleop_config_shm",  TELEOP_CONFIG,         "record_lock")

    # 카메라 종류는 main.py 가 시작 시 한 번 기록한 값을 참고만 한다 (분기 X 대신 view 토글).
    camera_type_str = "zed"  # default
    if teleop_config_shm is not None:
        try:
            cfg = teleop_config_shm.read_data()
            camera_type_str = CAMERA_MAPPING_INV.get(int(cfg["camera_type"].item()), "zed")
        except Exception as e:
            logger_mp.warning(f"[Record] teleop_config 읽기 실패: {e}")
    use_zed       = (camera_type_str == "zed")
    use_realsense = (camera_type_str == "realsense")
    logger_mp.info(f"[Record] camera_type={camera_type_str} (use_zed={use_zed}, use_realsense={use_realsense})")

    freq = 20.0
    rate = Rate(freq)


    # ── State Machine 변수 초기화 ────────────────────────────
    state = State.WAIT_FOR_SET
    done_once = False

    ep_idx = 0            # 현재 녹화할 에피소드 인덱스
    frame_count = 0       # 한 에피소드에서 저장된 프레임 수
    total_frames = 0      # 한 에피소드당 저장할 총 프레임 수 (episode_len * 50)

    # ── 에피소드 레코딩 시각적 데이터·로봇 데이터 accumulators (추가) ──
    frames_img_left    = None   # 각 프레임 이미지 배열을 append할 리스트 (None으로 초기화)
    frames_img_right    = None   # 각 프레임 이미지 배열을 append할 리스트 (None으로 초기화)
    frames_img_realsense = None
    frames_qpos   = None   # 각 프레임 qpos를 append할 리스트
    frames_action = None   # 각 프레임 action을 append할 리스트
    frames_hand_qpos = None   
    frames_hand_action = None   

    parquet_sink = ParquetSink(logger_mp)
    video_sink   = VideoSink(logger_mp, fps=freq)


    logger_mp.info("[INIT] 상태: WAIT_FOR_SET 대기 중")

    while not shared_event["shutdown"].is_set():

        try:
            try:
                task_data = record_task_shm.read_data()
                episode_data = record_episode_shm.read_data()
                start, reset, replay, mode_raw = read_mode_snapshot(record_mode_shm)
            except Exception:
                continue

            # =======================================================
            # 0) replay 요청은 어떤 상태이든 최우선 처리
            #    - mode_data["replay"]는 np.bool_ → .item() → Python bool
            # =======================================================
            

            if replay:
                # TODO: 실제 재생/삭제 구현 시 여기
                # record_mode_shm.write_data(replay=False)  # 소비
                state = State.IDLE
                continue
            
            # =======================================================
            # 1) 상태: WAIT_FOR_SET
            #    - UI에서 SET할 때 task_name이 "0" 또는 빈 문자열이 아니라면 IDLE로 전환
            # =======================================================

            if state is State.WAIT_FOR_SET:
                task_name = task_data["task_name"].item().strip()
                if task_name and task_name != "0":
                    # ep_idx 결정은 메타파일/SHM로 대체 권장
                    state = State.IDLE

                    data_base = os.path.join("record", task_name, "data")
                    if os.path.isdir(data_base):
                        files = glob.glob(os.path.join(data_base, "chunk-*", "episode_*.parquet"))
                        if files:
                            nums = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files]
                            ep_idx = max(nums) + 1
                        else:
                            ep_idx = 0
                    else:
                        ep_idx = 0

                    base_folder = "record"

                    task_folder = os.path.join(base_folder, task_name)
                    if not os.path.isdir(task_folder):
                        os.makedirs(task_folder, exist_ok=True)

                    logger_mp.info(f"[IDLE] Task 설정 완료: '{task_name}'. 레코드 대기 모드 진입")
                continue  # SET이 되지 않았으면 계속 대기

            # =======================================================
            # 2) 상태: IDLE (레코드 대기 모드)
            #    - mode_data["reset"]이 True: 플래그만 False로 리셋하고 즉시 다음 루프로 continue
            #    - mode_data["start"]이 True: 한 에피소드 RECORDING으로 진입 (단발성)
            #    - num_episodes, episode_len은 episode_data에서 np.int32 → .item() → Python int
            # =======================================================
            if state is State.IDLE:
                # (2-1) RESET 요청이 들어왔을 때: IDLE → IDLE (플래그만 리셋, 루프 continue)
                if reset:
                    record_mode_shm.write_data(reset=False, start=False)
                    logger_mp.info("[IDLE] RESET 신호 감지: IDLE 상태 유지")
                    continue  # 같은 사이클에서 start 플래그 검사하지 않도록 다음 루프로 건너뜀

                # (2-2) START 요청: 한 에피소드 녹화 진입
                if start:

                    num_episodes = int(episode_data["num_episodes"].item())
                    episode_len  = int(episode_data["episode_len"].item())
                    if ep_idx >= num_episodes or episode_len <= 0:
                        record_mode_shm.write_data(reset=False, start=False)
                        continue

                    frame_count = 0
                    total_frames = episode_len * int(freq)
                    record_mode_shm.write_data(start=False, reset=False, done=False)

                    task_name = task_data["task_name"].item().strip()
                    parquet_sink.start_episode(task_name, ep_idx)
                    video_sink.start_episode(task_name, ep_idx)

                    state = State.RECORDING

                    logger_mp.info(
                        f"[RECORDING] Episode {ep_idx} 녹화 시작: "
                        f"{episode_len}s → {total_frames}프레임 저장 예정"
                    )

                continue  # IDLE 블록 끝

            # =======================================================
            # 3) 상태: RECORDING (한 에피소드 저장 모드)
            #    - mode_data["reset"]이 True이면 “현재 에피소드만 취소” → IDLE
            #    - 아니면 매 주기마다 로그 저장 → frame_count += 1
            #    - frame_count >= total_frames 이면, Episode 저장 완료 → ep_idx += 1 → IDLE
            # =======================================================

            if state is State.RECORDING:
                # (3-1) RESET 요청: 현재 에피소드만 취소 → IDLE
                if reset:
                    record_mode_shm.write_data(reset=False)
                    state = State.IDLE
                    logger_mp.info(f"[RECORDING] RESET 감지: Episode {ep_idx} 녹화 취소 후 IDLE 복귀")
                    continue

                if start:
                    logger_mp.info(f"[RECORDING] START during recording: early stop at frame {frame_count}")
                    record_mode_shm.write_data(start=False)
                    # 전체 프레임 수를 현재 모은 프레임으로 맞춰, 저장 블록 진입 예약
                    total_frames = frame_count

                # (3-2) 실제 데이터 저장
                try:
                    image_dict = camera_shm.read_data()
                    robot_obs = robot_obs_shm.read_data()
                    robot_action = robot_action_shm.read_data()
                except Exception:
                    logger_mp.exception("[RECORDING] SHM read 실패, 프레임 스킵")
                    continue

                img_left      = image_dict.get("camera_left", None)
                img_right     = image_dict.get("camera_right", None)
                img_realsense = image_dict.get("realsense", None)
                # camera_type 에 따라 사용 안 하는 view 는 None 으로 — video_sink 가 mp4 안 만든다.
                if not use_zed:
                    img_left  = None
                    img_right = None
                if not use_realsense:
                    img_realsense = None

                obs_waist = robot_obs["obs_waist"]
                obs_head  = robot_obs["obs_head"]
                obs_arm   = robot_obs["obs_arm"]
                qpos      = np.concatenate((obs_waist, obs_head, obs_arm))

                action_waist = robot_action["action_waist"]
                action_head  = robot_action["action_head"]
                action_arm   = robot_action["action_arm"]
                action       = np.concatenate((action_waist, action_head, action_arm))

                # Inspire 양손 (12개 관절: 왼손6 + 오른손6)
                obs_hand    = robot_obs["obs_hand"]
                action_hand = robot_action["action_hand"]
                hand_qpos   = obs_hand
                hand_action = action_hand
                hand_kinesthetic = np.zeros(12, dtype=np.float32)

                state_vec        = np.concatenate([qpos, hand_qpos])
                state_vec_sensor = hand_kinesthetic
                action_vec       = np.concatenate([action, hand_action])

                parquet_sink.append(state_vec, state_vec_sensor, action_vec, t_sec=frame_count/float(freq))
                # 원본 이미지만 저장 (realsense, zed_left, zed_right)
                video_sink.append(img_left, img_right, img_realsense)

                frame_count += 1

                if total_frames > 0:
                    record_episode_shm.write_data(logging_progress=int(frame_count*100/total_frames))

                if frame_count >= total_frames:

                    parquet_sink.close_episode()
                    video_sink.close_episode()

                    # 4) 다음 에피소드 준비
                    logger_mp.info(f"[RECORDING] Episode {ep_idx} 저장 완료 (Parquet + MP4)")
                    record_episode_shm.write_data(episode_index=ep_idx+1)
                    record_mode_shm.write_data(done=True)
                    ep_idx += 1

                    if ep_idx < num_episodes:
                        state = State.IDLE
                        remaining = num_episodes - ep_idx
                        logger_mp.info(f"[IDLE] Episode {ep_idx-1} 완료. 다음 에피소드 대기 모드 진입 (남은: {remaining})")
                    else:
                        state = State.IDLE
                        logger_mp.info("[IDLE] 모든 에피소드 완료됨. 추가 녹화 없음")
                continue  
        finally:
            hz = rate.tick_hz()                  # 지난 사이클 기준 실제 주파수 계산
            freq_shm.write_data(record_freq=hz)  # 주파수 보고는 여기서만
            rate.sleep()                         # 20Hz 보장



    # while 루프 종료 (shutdown 신호 수신)
    logger_mp.info("[Record Worker] 종료 신호 수신. 정상 종료합니다.")

    # 모든 공유 메모리 정리 (shm_objects 리스트에 있는 모든 객체)
    for shm in shm_objects:
        try:
            if shm and hasattr(shm, 'worker_close'):
                shm.worker_close()
        except Exception as e:
            logger_mp.warning(f"[Record Worker] SHM 정리 실패: {e}")

    logger_mp.info(f"[Record Worker] {len(shm_objects)}개 공유 메모리 객체 정리 완료")