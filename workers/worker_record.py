# worker_record.py

import os

from utils.rate import Rate
from utils import ParquetSink, VideoSink

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA, RECORD_TASK_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_MODE_LAYOUT, CURRENT_MODE_LAYOUT, WORKER_FREQ, ROBOT_ACTION, ROBOT_OBS, ROBOT_AMO_INPUT, KISTAR_HAND_RECEIVED, KISTAR_HAND_ACTION, WORKSPACE_MASK, MODE_MAPPING, MODE_MAPPING_INV

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

def get_current_mode(current_mode_shm):
    """공유 메모리에서 현재 모드 정보를 읽어옴 (정수 → 문자열 변환)"""
    try:
        mode_data = current_mode_shm.read_data()
        mode_int = int(mode_data["mode"].item())
        current_mode = MODE_MAPPING_INV.get(mode_int, 'teleop')
        
        return current_mode
    except Exception as e:
        logger_mp.warning(f"[Record] 모드 정보 읽기 실패: {e}, 기본값 'teleop' 사용")
        return "teleop"

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
    record_task_shm = create_shm("record_task_shm", RECORD_TASK_LAYOUT, "record_lock")
    record_episode_shm = create_shm("record_episode_shm", RECORD_EPISODE_LAYOUT, "record_lock")
    record_mode_shm = create_shm("record_mode_shm", RECORD_MODE_LAYOUT, "record_lock")
    current_mode_shm = create_shm("current_mode_shm", CURRENT_MODE_LAYOUT, "record_lock")
    camera_shm = create_shm("camera_shm", CAMERA, "camera_lock")
    robot_action_shm = create_shm("robot_action_shm", ROBOT_ACTION, "robot_action_lock")
    robot_obs_shm = create_shm("robot_obs_shm", ROBOT_OBS, "robot_obs_lock")
    freq_shm = create_shm("freq_shm", WORKER_FREQ, "freq_lock")
    robot_amo_input_shm = create_shm("robot_amo_input_shm", ROBOT_AMO_INPUT, "robot_amo_input_lock")
    workspace_mask_shm = create_shm("workspace_mask_shm", WORKSPACE_MASK, "workspace_mask_lock")

    # KISTAR 손 데이터 공유 메모리 (kistar_teleop, kistar_only 모드에서만 사용)
    kistar_hand_received_shm = None
    kistar_hand_action_shm = None
    if current_mode_shm:
        logger_mp.info("[Reocord Worker] Read Current Mode!!")
        current_mode = get_current_mode(current_mode_shm)
        logger_mp.info("[Reocord Worker] Read Current Finish!!")
        if current_mode in ['kistar_teleop', 'kistar_only', 'kistar_inspire_teleop'] and "kistar_hand_received_shm" in shm_name:
            try:
                kistar_hand_received_shm = SharedMemoryManager(
                    KISTAR_HAND_RECEIVED,
                    shared_lock["kistar_hand_received_lock"],
                    shm_name["kistar_hand_received_shm"]
                )
                shm_objects.append(kistar_hand_received_shm)  # 정리 리스트에 추가
                
                # KISTAR hand action 공유 메모리 추가
                if "kistar_hand_action_shm" in shm_name:
                    kistar_hand_action_shm = SharedMemoryManager(
                        KISTAR_HAND_ACTION,
                        shared_lock["kistar_hand_action_lock"],
                        shm_name["kistar_hand_action_shm"]
                    )
                    shm_objects.append(kistar_hand_action_shm)  # 정리 리스트에 추가
                
                logger_mp.info(f"[Record] KISTAR 손 공유 메모리 초기화 완료 (모드: {current_mode})")
            except Exception as e:
                logger_mp.warning(f"[Record] KISTAR 손 공유 메모리 초기화 실패: {e}")
                kistar_hand_received_shm = None
                kistar_hand_action_shm = None

    
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
                    amo_input = robot_amo_input_shm.read_data()

                    # KISTAR 손 데이터 읽기 (kistar_teleop, kistar_only 모드에서만)
                    kistar_data = None
                    kistar_action_data = None
                    if kistar_hand_received_shm is not None:
                        try:
                            kistar_data = kistar_hand_received_shm.read_data()
                        except Exception as e:
                            logger_mp.warning(f"[RECORDING] KISTAR 손 observation 읽기 실패: {e}")
                            kistar_data = None
                    
                    # KISTAR hand action 데이터 읽기
                    if kistar_hand_action_shm is not None:
                        try:
                            kistar_action_data = kistar_hand_action_shm.read_data()
                        except Exception as e:
                            logger_mp.warning(f"[RECORDING] KISTAR 손 action 읽기 실패: {e}")
                            kistar_action_data = None

                except Exception :
                    logger_mp.exception("[RECORDING] SHM read 실패, 프레임 스킵")
                    continue

                img_left    = image_dict.get("camera_left", None)
                img_right   = image_dict.get("camera_right", None)
                img_realsense   = image_dict.get("realsense", None)

                pelvis_pose = amo_input["pelvis_pose"]
                torso_quat = amo_input["torso_quat"]
                vel_command = amo_input["vel_command"]


                obs_leg = robot_obs["obs_leg"]
                obs_waist = robot_obs["obs_waist"]
                obs_head = robot_obs["obs_head"]
                obs_arm = robot_obs["obs_arm"]
                qpos = np.concatenate((obs_waist,obs_head,obs_arm))

                action_waist = robot_action["action_waist"]
                action_head = robot_action["action_head"]
                action_arm = robot_action["action_arm"]
                action = np.concatenate((action_waist,action_head,action_arm))

                # 모드별 손 데이터 처리 
                # logger_mp.info(f"current mode: {current_mode}")
                if current_mode in ['kistar_teleop', 'kistar_only', 'kistar_inspire_teleop']:
                    # KISTAR 손 모드: KISTAR 데이터 사용 (16개 관절)
                    # Observation: ASUS NUC로부터 받은 현재 상태
                    if kistar_data is not None:
                        hand_qpos = kistar_data["hand_q_pos"]  # 16개 (observation)
                        
                        # print("hand_qpos_observation: ",hand_qpos/np.pi*180)

                    else:
                        logger_mp.warning("[RECORDING] KISTAR 손 observation 없음, 기본값 사용")
                        hand_qpos = np.zeros(16, dtype=np.float32)

                    if current_mode=='kistar_inspire_teleop':
                        logger_mp.info("[Worker Record] kistar inspire teleoperation working !!")
                        obs_hand = robot_obs["obs_hand"]
                        left_hand = obs_hand[:6]
                        
                        print("kistar_qpos_set: ",hand_qpos)
                        print("inspire_hand_set: ",left_hand)
                        hand_qpos=np.concatenate([left_hand, hand_qpos])

                        print("hand_pose set: ",hand_qpos)
                    
                    # Action: 전송된 제어 명령
                    if kistar_action_data is not None:
                        hand_action = kistar_action_data["hand_action"]  # 16개 (action)
                    else:
                        # action이 없으면 observation 사용 (fallback)
                        if kistar_data is not None:
                            hand_action = kistar_data["hand_q_pos"]
                        else:
                            logger_mp.warning("[RECORDING] KISTAR 손 action 없음, 기본값 사용")
                            hand_action = np.zeros(16, dtype=np.float32)                    

                    if current_mode=='kistar_inspire_teleop':
                        action_hand = robot_action["action_hand"]
                        left_hand_action = action_hand[:6]

                        hand_action=np.concatenate([left_hand_action, hand_action])


                else:
                    # 일반 모드: Inspire 손 데이터 사용 (12개 관절)
                    # logger_mp.info("[Worker Record] inspire only teleoperation working !!")
                    obs_hand = robot_obs["obs_hand"]
                    action_hand = robot_action["action_hand"]
                    hand_qpos = obs_hand
                    hand_action = action_hand

                state_vec  = np.concatenate([qpos, hand_qpos])
                action_vec = np.concatenate([action, hand_action])

                parquet_sink.append(state_vec, action_vec, t_sec=frame_count/float(freq))
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