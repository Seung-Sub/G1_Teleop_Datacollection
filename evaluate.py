"""evaluate.py -- run a trained external policy against the live SHM bus.

Runs in a *separate* conda environment from main.py (e.g. `gr00t`) where the
policy library (Isaac-GR00T) is pip-installed. main.py must already be
running with matching --hand/--camera/--vr-input options so that the
worker_g1_ctrl / worker_g1_ik / worker_hand_ctrl / camera workers own the
SHM segments.

Workflow
    Terminal 1:  conda activate teleop
                 python main.py --hand inspire --camera zed
    Terminal 2:  conda activate gr00t
                 python evaluate.py --mode gr00t_zed \\
                     --model-path /path/to/checkpoint-XXXXX \\
                     --data-config-key unitree_g1_inspire \\
                     --action-method tem

Once both are up, set the Language Instruction in the GUI (writes
GR00T_TASK_LAYOUT) and click Deploy (sets RECORD_MODE.deploy = True).
evaluate.py lazy-loads the policy on the first tick with deploy=True, then
publishes actions to ROBOT_ACTION (where worker_g1_ctrl / worker_hand_ctrl
pick them up).

This mirrors the historical deploy_gr00t.py pattern (commit bb58b6a) but
with hand_model / KISTAR / maniflow / ACT branches removed (inspire-only).
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import sys
import time

from workers.worker_deploy_policy import Gr00t_Inference

# SHM names must match main.py's SHM_CONFIG exactly (we *attach*, we do not
# create -- main.py is the owner).
SHM_NAMES = {
    "camera_shm":           "camera_shm",
    "robot_obs_shm":        "robot_obs_shm",
    "robot_action_shm":     "robot_action_shm",
    "gr00t_shm":            "gr00t_shm",
    "record_mode_shm":      "record_mode_shm",
    "workspace_mask_shm":   "workspace_mask_shm",
}


def _build_locks():
    """Local lock dict; main.py uses *its own* locks for cross-process
    synchronisation. evaluate.py only needs locks for its own internal
    SharedMemoryManager bookkeeping (the underlying SHM bytes are shared
    OS-wide so writes are atomic at numpy granularity per-field)."""
    return {
        "camera_lock":         mp.Lock(),
        "robot_obs_lock":      mp.Lock(),
        "robot_action_lock":   mp.Lock(),
        "gr00t_lock":          mp.Lock(),
        "record_lock":         mp.Lock(),
        "workspace_mask_lock": mp.Lock(),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="External GR00T policy evaluator (inspire-only)")
    p.add_argument("--mode", choices=["gr00t", "gr00t_zed"], default="gr00t_zed",
                   help="gr00t = RealSense single view; gr00t_zed = ZED stereo")
    p.add_argument("--model-path",    required=True,
                   help="Path to the trained GR00T checkpoint (the directory, "
                        "e.g. /path/to/checkpoint-100000)")
    p.add_argument("--data-config-key", default="unitree_g1_inspire",
                   help="Key into gr00t.experiment.data_config.DATA_CONFIG_MAP")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--action-method", choices=["base", "maf", "tem"], default="tem")
    p.add_argument("--decay",         type=float, default=0.3,
                   help="TEM decay coefficient (only used with --action-method tem)")
    p.add_argument("--window-size",   type=int,   default=5,
                   help="MAF/TEM window length")
    p.add_argument("--slow-hz",       type=float, default=20.0)
    p.add_argument("--fast-hz",       type=float, default=50.0)
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--binocular", dest="binocular", action="store_true", default=True)
    p.add_argument("--no-binocular", dest="binocular", action="store_false",
                   help="Disable right-view consumption even in --mode gr00t_zed")
    p.add_argument("--masking", action="store_true",
                   help="Apply workspace_mask_shm to input frames before inference")
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
    )

    print(f"[evaluate] Running. UI must set Deploy=True to start policy loading.")
    print(f"[evaluate] mode={args.mode} action_method={args.action_method} "
          f"slow={args.slow_hz}Hz fast={args.fast_hz}Hz")
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
