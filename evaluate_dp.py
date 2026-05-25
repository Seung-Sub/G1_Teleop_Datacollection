"""evaluate_dp.py — run a trained Diffusion Policy against the live SHM bus.

GR00T 의 evaluate.py 와 동일 구조. main.py(teleop) 가 worker_g1_ctrl / worker_g1_ik /
worker_hand_ctrl / camera worker 를 띄워 SHM 을 소유한 상태에서, 별도 conda env(DP 학습
환경: torch/hydra/diffusers + diffusion_policy)에서 실행해 SHM 에 attach.

GR00T(evaluate.py) 대비 차이:
  - DP_Inference 사용 (Gr00t_Inference 대신).
  - --model-path = DP .ckpt 경로 (workspace checkpoint).
  - --slow-hz 기본 10 (DP 학습 60→10 다운샘플). fast 60 (arm 제어, 동일).
  - language/embodiment/data-config 인자 없음 (DP 단일 task).
  - --camera-key-map 으로 role→camera_N 매핑 (shape_meta 와 일치, 기본 ego/wrist_l/wrist_r).

Workflow:
    Terminal 1:  conda activate teleop
                 python main.py --hand dex3 --camera realsense \\
                     --vr-input controller --waist fixed --head off --lower-body hoist
    Terminal 2:  conda activate umi   # 또는 DP 학습 환경
                 python evaluate_dp.py --mode gr00t_rs_multi \\
                     --model-path /path/to/checkpoints/latest.ckpt
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import sys
import time

from workers.worker_deploy_dp import DP_Inference

# SHM names must match main.py's SHM_CONFIG exactly (attach, not create).
SHM_NAMES = {
    "robot_obs_shm":        "robot_obs_shm",
    "robot_action_shm":     "robot_action_shm",
    "gr00t_shm":            "gr00t_shm",
    "record_mode_shm":      "record_mode_shm",
    "teleop_config_shm":    "teleop_config_shm",
    "workspace_mask_shm":   "workspace_mask_shm",
    "camera_shm":           "camera_shm",        # legacy ZED single SHM
    "rs_ego_shm":           "rs_ego_shm",
    "rs_wrist_l_shm":       "rs_wrist_l_shm",
    "rs_wrist_r_shm":       "rs_wrist_r_shm",
}


def _build_locks():
    return {
        "camera_lock":         mp.Lock(),
        "rs_ego_lock":         mp.Lock(),
        "rs_wrist_l_lock":     mp.Lock(),
        "rs_wrist_r_lock":     mp.Lock(),
        "robot_obs_lock":      mp.Lock(),
        "robot_action_lock":   mp.Lock(),
        "gr00t_lock":          mp.Lock(),
        "record_lock":         mp.Lock(),
        "workspace_mask_lock": mp.Lock(),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Deploy a trained Diffusion Policy on G1 via SHM.")
    p.add_argument("--mode", choices=["gr00t_rs_multi", "gr00t_zed", "gr00t"],
                   default="gr00t_rs_multi",
                   help="gr00t_rs_multi = RealSense 멀티뷰 (ego/wrist_l/wrist_r), "
                        "gr00t_zed = ZED stereo. (DP 도 동일 카메라 SHM 사용.)")
    p.add_argument("--model-path", required=True,
                   help="DP checkpoint(.ckpt) 경로 (예: data/outputs/.../checkpoints/latest.ckpt)")
    p.add_argument("--action-method", choices=["base", "maf", "tem"], default="tem")
    p.add_argument("--decay",         type=float, default=0.3, help="TEM decay 계수")
    p.add_argument("--window-size",   type=int,   default=5,   help="MAF/TEM 윈도우 길이")
    p.add_argument("--slow-hz",       type=float, default=10.0,
                   help="추론 주기. DP action chunk step 이 10Hz(학습 60→10 다운샘플) 타임스텝.")
    p.add_argument("--fast-hz",       type=float, default=60.0,
                   help="action 실행/업샘플 주기. arm 제어 루프(worker_g1_ctrl ACT_HZ=60)와 일치.")
    p.add_argument("--device", default="cuda",
                   help="DP policy device (예: cuda, cuda:0, cpu).")
    p.add_argument("--binocular", dest="binocular", action="store_true", default=True,
                   help="(ZED 전용) stereo 양안 활성")
    p.add_argument("--no-binocular", dest="binocular", action="store_false")
    p.add_argument("--masking", action="store_true")
    p.add_argument("--lag-compensate", dest="lag_compensate", action="store_true", default=True,
                   help="추론 지연만큼 chunk 앞부분 trim (default on)")
    p.add_argument("--no-lag-compensate", dest="lag_compensate", action="store_false")
    p.add_argument("--lag-log-every", type=int, default=50)
    p.add_argument("--obs-ts-policy", choices=["min", "max"], default="min")
    p.add_argument("--modality-json", dest="modality_json", default=None,
                   help="(선택) record/<task>/meta/modality.json. DP 는 state 28D 를 직접 "
                        "구성하므로 필수 아님 — hand_type 은 teleop_config_shm 에서 결정.")
    return p.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = _parse_args()

    shared_event = {"shutdown": mp.Event()}
    shared_lock  = _build_locks()

    inf = DP_Inference(
        shm_name      =SHM_NAMES,
        shared_lock   =shared_lock,
        shared_event  =shared_event,
        mode          =args.mode,
        model_path    =args.model_path,
        action_method =args.action_method,
        decay         =args.decay,
        window_size   =args.window_size,
        slow_hz       =args.slow_hz,
        fast_hz       =args.fast_hz,
        device        =args.device,
        binocular     =args.binocular,
        masking       =args.masking,
        lag_compensate=args.lag_compensate,
        lag_log_every =args.lag_log_every,
        obs_ts_policy =args.obs_ts_policy,
        modality_json_path=args.modality_json,
    )

    print(f"[evaluate_dp] Running. UI must set Deploy=True to start policy loading.")
    print(f"[evaluate_dp] mode={args.mode} action_method={args.action_method} "
          f"slow={args.slow_hz}Hz fast={args.fast_hz}Hz device={args.device} "
          f"lag_compensate={args.lag_compensate} obs_ts_policy={args.obs_ts_policy}")
    try:
        while not shared_event["shutdown"].is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        shared_event["shutdown"].set()
    finally:
        inf.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
