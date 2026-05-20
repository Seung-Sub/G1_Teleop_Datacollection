"""evaluate.py — run a trained external policy against the live SHM bus.

Runs in a *separate* conda environment from main.py (e.g. `gr00t`) where the
policy library (Isaac-GR00T) is pip-installed. main.py must already be
running with matching --hand/--cameras-config so the worker_g1_ctrl /
worker_g1_ik / worker_hand_ctrl / camera workers own the SHM segments.

Phase L (Part 2) 변경:
  - mode 기본값을 `gr00t_rs_multi` 로 (RealSense 멀티뷰, DEX3+RS3뷰 운용 정합).
  - hand_type 자동 결정 (teleop_config_shm 의 hand_type 읽음) — CLI 미지정.
  - SHM_NAMES 에 rs_ego_shm / rs_wrist_l_shm / rs_wrist_r_shm + teleop_config_shm
    포함. main.py 가 owner-create 한 것만 deploy 측이 attach (없으면 skip).
  - --obs-ts-policy {min,max} CLI 추가. default min (stale 기준 = lag 보상 안전).

Workflow:
    Terminal 1:  conda activate teleop
                 python main.py --hand dex3 --cameras-config utils/cameras.yaml \\
                                --vr-input controller --waist fixed --head off
    Terminal 2:  conda activate gr00t
                 python evaluate.py --mode gr00t_rs_multi \\
                     --model-path /path/to/checkpoint-XXXXX \\
                     --data-config-key <KEY>   # DEX3+RS3뷰 학습 시 등록한 키
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import sys
import time

from workers.worker_deploy_policy import Gr00t_Inference

# SHM names must match main.py's SHM_CONFIG exactly (we *attach*, we do not
# create — main.py is the owner).
SHM_NAMES = {
    # Robot / mode / language instruction / mask
    "robot_obs_shm":        "robot_obs_shm",
    "robot_action_shm":     "robot_action_shm",
    "gr00t_shm":            "gr00t_shm",
    "record_mode_shm":      "record_mode_shm",
    "teleop_config_shm":    "teleop_config_shm",
    "workspace_mask_shm":   "workspace_mask_shm",
    # Camera SHMs — main.py 가 cameras.yaml 기반으로 owner-create. deploy 는 attach.
    # 없으면 Gr00t_Inference 가 자동 skip.
    "camera_shm":           "camera_shm",        # legacy ZED single SHM
    "rs_ego_shm":           "rs_ego_shm",
    "rs_wrist_l_shm":       "rs_wrist_l_shm",
    "rs_wrist_r_shm":       "rs_wrist_r_shm",
}


def _build_locks():
    """Local lock dict — process-local. SHM 자체는 OS-wide 공유라 mutex 의미는
    process-private. Phase L1: camera role 별 lock 추가."""
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
    p = argparse.ArgumentParser(description="External GR00T policy evaluator")
    p.add_argument("--mode", choices=["gr00t_rs_multi", "gr00t_zed", "gr00t"],
                   default="gr00t_rs_multi",
                   help="gr00t_rs_multi = RealSense 멀티뷰 (ego/wrist_l/wrist_r), "
                        "gr00t_zed = ZED stereo, 'gr00t' = alias for gr00t_rs_multi (legacy).")
    p.add_argument("--model-path",    required=True,
                   help="GR00T checkpoint 디렉토리 (예: /path/to/checkpoint-100000)")
    p.add_argument("--data-config-key", default="unitree_g1",
                   help="gr00t.experiment.data_config.DATA_CONFIG_MAP 의 키. "
                        "사용자 운용 (DEX3+RS3뷰) 시 학습 측에 신규 DataConfig 등록 필요. "
                        "기본값은 legacy unitree_g1 — 운용 시 학습한 키로 override 할 것.")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--action-method", choices=["base", "maf", "tem"], default="tem")
    p.add_argument("--decay",         type=float, default=0.3,
                   help="TEM decay 계수")
    p.add_argument("--window-size",   type=int,   default=5,
                   help="MAF/TEM 윈도우 길이")
    p.add_argument("--slow-hz",       type=float, default=20.0)
    p.add_argument("--fast-hz",       type=float, default=50.0)
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--binocular", dest="binocular", action="store_true", default=True,
                   help="(ZED 전용) stereo 양안 활성")
    p.add_argument("--no-binocular", dest="binocular", action="store_false",
                   help="ZED single-view")
    p.add_argument("--masking", action="store_true",
                   help="workspace_mask_shm 을 frame 에 적용 (legacy ZED)")
    # Phase E — chunk lag compensation
    p.add_argument("--lag-compensate", dest="lag_compensate", action="store_true", default=True,
                   help="Trim chunk start by measured (t_publish - t_obs) lag (default on)")
    p.add_argument("--no-lag-compensate", dest="lag_compensate", action="store_false",
                   help="Disable inference-lag chunk trim (debug)")
    p.add_argument("--lag-log-every", type=int, default=50,
                   help="Log avg/max lag every N chunks")
    # Phase L2 — obs ts 정책
    p.add_argument("--obs-ts-policy", choices=["min", "max"], default="min",
                   help="obs_ts_ns 결정 (모든 modality ts 후보 중). "
                        "min = 가장 stale (의미상 정확, lag 보상 안전), "
                        "max = 가장 최근 (legacy 동작).")
    # Phase M4 — modality.json 명시 (학습 데이터셋의 meta 파일)
    p.add_argument("--modality-json", dest="modality_json", default=None,
                   help="record/<task>/meta/modality.json 경로. 명시되면 layout 을 "
                        "이 파일에서 읽어 obs/action dict 구성 (학습=배포 자동 정합). "
                        "미명시 시 TELEOP_CONFIG SHM 의 토글로 build_state_layout 호출.")
    return p.parse_args()


def main():
    mp.set_start_method("spawn", force=True)

    args = _parse_args()

    shared_event = {"shutdown": mp.Event()}
    shared_lock  = _build_locks()

    inf = Gr00t_Inference(
        shm_name      =SHM_NAMES,
        shared_lock   =shared_lock,
        shared_event  =shared_event,
        mode          =args.mode,
        model_path    =args.model_path,
        data_config_key=args.data_config_key,
        embodiment_tag=args.embodiment_tag,
        action_method =args.action_method,
        decay         =args.decay,
        window_size   =args.window_size,
        slow_hz       =args.slow_hz,
        fast_hz       =args.fast_hz,
        denoising_steps=args.denoising_steps,
        binocular     =args.binocular,
        masking       =args.masking,
        lag_compensate=args.lag_compensate,
        lag_log_every =args.lag_log_every,
        obs_ts_policy =args.obs_ts_policy,
        modality_json_path=args.modality_json,
    )

    print(f"[evaluate] Running. UI must set Deploy=True to start policy loading.")
    print(f"[evaluate] mode={args.mode} action_method={args.action_method} "
          f"slow={args.slow_hz}Hz fast={args.fast_hz}Hz "
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
