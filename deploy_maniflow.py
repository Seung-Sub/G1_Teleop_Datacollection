import multiprocessing
from workers.worker_deploy_policy_maniflow import Gr00t_Inference # change name for maniflow deployment
from workers.worker_plot_maniflow import worker_plot
import argparse
import time

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA, GR00T_TASK_LAYOUT, RECORD_MODE_LAYOUT, ROBOT_ACTION, ROBOT_OBS, KISTAR_HAND_RECEIVED, KISTAR_HAND_ACTION, MASK_CONTROL_LAYOUT, WORKSPACE_MASK, DEPTH_MAP


def gr00t_worker(shared_event, shm_names, shared_lock,
                policy, mode, action_method, hand_mode, decay, window_size,
                slow_hz, fast_hz):
    inf = Gr00t_Inference(
        shm_name=shm_names,
        shared_lock=shared_lock,
        shared_event=shared_event,
        policy = policy,
        mode=mode,
        action_method=action_method,
        hand_mode=hand_mode,
        decay=decay,
        window_size=window_size,
        slow_hz=slow_hz,
        fast_hz=fast_hz,
    )
    shared_event["shutdown"].wait()
    inf.stop()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  # Windows일 경우 필수

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",choices=["gr00t", "act", "maniflow"],default="gr00t",help="Which policy set to run") # (NEW 0209): add maniflow as a policy choice
    parser.add_argument(
        "--mode",
        choices=["gr00t", "gr00t_zed", "gr00t_kistar", "gr00t_kistar_inspire"],
        default="gr00t",
        help="Which worker set to run"
    )
    parser.add_argument("--action_method",default="base",choices=["base","tem", "maf", "lipo"],help="")
    parser.add_argument("--decay",      type=float, default=0.3, help="지수 가중치 계수")
    parser.add_argument("--window_size",type=int,   default=5,   help="이동 평균 윈도우 크기")    
    parser.add_argument(
        "--hand_model",
        choices=["full", "reduced", "reduced_v3", "reduced_pca", "kistar_only"],
        default="full",
        help="Hand DOF mode"
    )

    args = parser.parse_args()
    policy = args.policy
    mode = args.mode
    hand_mode = args.hand_model
    action_method = args.action_method
    decay = args.decay
    window_size = args.window_size
    slow_hz = 20.0
    fast_hz =50.0

    shared_event = {
        "shutdown": multiprocessing.Event()
    }

    shared_lock = {
        "camera_lock": multiprocessing.Lock(),
        "gr00t_lock": multiprocessing.Lock(),
        "record_lock": multiprocessing.Lock(),
        "robot_action_lock": multiprocessing.Lock(),
        "robot_obs_lock": multiprocessing.Lock(),
        "kistar_hand_received_lock": multiprocessing.Lock(),  # KISTAR 손 추가
        "kistar_hand_action_lock": multiprocessing.Lock(),  # KISTAR 손 추가
        "workspace_mask_lock": multiprocessing.Lock(), # 마스크 lock 추가
        "depth_map_lock": multiprocessing.Lock(), # 뎁스맵 lock 추가
    }

    shm_names = {
        "camera_shm":           "camera_shm",
        "television_shm":       "television_shm",
        "record_task_shm":      "record_task_shm",
        "record_episode_shm":   "record_episode_shm",
        "record_mode_shm":      "record_mode_shm",
        "freq_shm":             "freq_shm",
        "pelvis_shm":           "pelvis_shm",
        "visual_shm":           "visual_shm",
        "gr00t_shm":            "gr00t_shm",
        "robot_obs_shm":       "robot_obs_shm",
        "robot_action_shm":       "robot_action_shm",
        "kistar_hand_received_shm": "kistar_hand_received_shm",  
        "kistar_hand_action_shm": "kistar_hand_action_shm",   
        "mask_control_shm":     "mask_control_shm",  # 마스크 제어 추가
        "workspace_mask_shm":   "workspace_mask_shm", # 마스크 데이터 추가
        "depth_map_shm":        "depth_map_shm",      # 뎁스맵 데이터 추가
    }

    camera_shm = SharedMemoryManager(CAMERA, shared_lock["camera_lock"], shm_names["camera_shm"])
    gr00t_task_shm = SharedMemoryManager(GR00T_TASK_LAYOUT, shared_lock["gr00t_lock"],shm_names["gr00t_shm"])
    record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_names["record_mode_shm"])
    robot_obs_shm = SharedMemoryManager(ROBOT_OBS, shared_lock["robot_obs_lock"], shm_names["robot_obs_shm"])
    robot_action_shm = SharedMemoryManager(ROBOT_ACTION, shared_lock["robot_action_lock"], shm_names["robot_action_shm"])
    kistar_hand_shm = SharedMemoryManager(KISTAR_HAND_RECEIVED, shared_lock["kistar_hand_received_lock"], shm_names["kistar_hand_received_shm"])
    kistar_hand_action_shm = SharedMemoryManager(KISTAR_HAND_ACTION, shared_lock["kistar_hand_action_lock"], shm_names["kistar_hand_action_shm"])
    mask_control_shm = SharedMemoryManager(MASK_CONTROL_LAYOUT, shared_lock["record_lock"], shm_names["mask_control_shm"])
    workspace_mask_shm = SharedMemoryManager(WORKSPACE_MASK, shared_lock["workspace_mask_lock"], shm_names["workspace_mask_shm"])
    depth_map_shm = SharedMemoryManager(DEPTH_MAP, shared_lock["depth_map_lock"], shm_names["depth_map_shm"])
    
    gr00t_process = multiprocessing.Process(
        target=gr00t_worker,
        args=(shared_event, shm_names, shared_lock,policy, mode, action_method, hand_mode, decay, window_size, slow_hz, fast_hz)
    )

    gr00t_process.start()
    
    try:
        gr00t_process.join() 
    except KeyboardInterrupt:
        shared_event["shutdown"].set()
        gr00t_process.join()

# # 2026-02-10 run on PC with:

# export CUDA_VISIBLE_DEVICES=0
# export MANIFLOW_CKPT=/media/ansur/ANSURLAB/2026-02-11_maniflow_deploy/checkpoints/2026-02-05_apple_stereo_full_ddp_aug2/epoch=0006-val_loss=0.084252.ckpt
# export MANIFLOW_STEPS=2
# export MANIFLOW_USE_EMA=1
# export MANIFLOW_COMPILE=0
# conda activate g1-mf-deploy
# cd /home/ansur/ssd2tb/Kapex-Vanilla-VLA/G1_Teleoperation
# python deploy_maniflow.py \
#   --mode gr00t_kistar_inspire \
#   --policy maniflow \
#   --action_method tem \
#   --hand_model reduced_v3
