"""worker_deploy_policy: external policy (GR00T) inference adapter for inspire-only G1.

This worker reads observations from the same SHM segments that main.py
populates, calls an externally-trained policy at a slow rate (20Hz default),
upsamples the action chunk to a faster publish rate (50Hz default) with
cross-fade blending across chunk boundaries, applies one of {base, maf, tem}
post-processing methods, and writes back to ROBOT_ACTION. main.py's
worker_g1_ctrl / worker_hand_ctrl take it from there.

This is the inspire-only redesign of the historical worker_deploy_policy.py
that was removed in Phase 0. KISTAR / PCA / maniflow / ACT / reduced_v3 paths
are intentionally absent; if a non-inspire end-effector is added later it
should plug in here as a new mode.

The GR00T library import is *lazy* so this module is safely importable in the
teleop conda env (which does not have gr00t installed); the actual policy is
only loaded once UI/keyboard sets record_mode_shm.deploy = True, inside the
deploy conda env where evaluate.py runs.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import cv2

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    CAMERA, GR00T_TASK_LAYOUT, RECORD_MODE_LAYOUT,
    ROBOT_ACTION, ROBOT_OBS, WORKSPACE_MASK,
)

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


# =============================================================================
# Frame / observation helpers
# =============================================================================

def process_frame(frame: np.ndarray, side: str = "") -> Optional[np.ndarray]:
    """Validate + BGR -> RGB + uint8. Returns None if the frame is unusable.

    Same shape rules as utils.frame_utils.process_frames: HxWx3 only.
    """
    if not (isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3):
        logger_mp.warning(f"[Deploy] bad frame ({side}): shape={getattr(frame,'shape',None)}")
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def upsample_actions(seq: np.ndarray, slow_hz: float = 20.0, fast_hz: float = 50.0,
                     k: int = 5) -> np.ndarray:
    """Spline-upsample a (T, D) action chunk from slow_hz to fast_hz.

    Mirrors the historical behaviour: scipy quintic spline (k=5) with clamped
    edges. Returns shape (T_up, D) where T_up = ceil((T-1) * fast_hz/slow_hz)+1.
    """
    from scipy.interpolate import make_interp_spline
    T, D = seq.shape
    if T < 2:
        return seq
    t_in  = np.linspace(0.0, (T - 1) / slow_hz, T)
    n_out = int(round((T - 1) * (fast_hz / slow_hz))) + 1
    t_out = np.linspace(0.0, (T - 1) / slow_hz, n_out)
    out = np.empty((n_out, D), dtype=seq.dtype)
    order = min(k, T - 1)
    for j in range(D):
        spl = make_interp_spline(t_in, seq[:, j], k=order)
        out[:, j] = spl(t_out)
    return out


# =============================================================================
# Observation dict builders (GR00T DataConfig compatible)
# =============================================================================

def _split_qpos(qpos: np.ndarray):
    """qpos = concat(obs_waist[3], obs_head[2], obs_arm[14]) = 19-D
    Returns (waist, left_arm, right_arm) — head is *not* fed to inspire-only
    DataConfigs because they share the unitree_g1_inspire modality which does
    not include state.head.
    """
    waist     = qpos[:3].astype(np.float32)
    # qpos[3:5] is head; ignored on purpose.
    left_arm  = qpos[5:12].astype(np.float32)
    right_arm = qpos[12:19].astype(np.float32)
    return waist, left_arm, right_arm


def obs_dict_realsense(task_name: str, qpos: np.ndarray,
                       hand_qpos_12d: np.ndarray, rgb: np.ndarray) -> dict:
    """RealSense single-view inspire-only observation dict."""
    waist, left_arm, right_arm = _split_qpos(qpos)
    left_hand  = hand_qpos_12d[:6].astype(np.float32)
    right_hand = hand_qpos_12d[6:12].astype(np.float32)
    return {
        "video.rs_view":       rgb[None, None, ...],  # (1, 1, H, W, 3) uint8
        "state.waist":         waist[None, None, :],
        "state.left_arm":      left_arm[None, None, :],
        "state.right_arm":     right_arm[None, None, :],
        "state.left_hand":     left_hand[None, None, :],
        "state.right_hand":    right_hand[None, None, :],
        "annotation.human.action.task_description": np.array([task_name], dtype=object),
    }


def obs_dict_zed(task_name: str, qpos: np.ndarray, hand_qpos_12d: np.ndarray,
                 rgb_left: np.ndarray, rgb_right: Optional[np.ndarray] = None,
                 binocular: bool = True) -> dict:
    """ZED stereo (binocular=True) or single-view inspire-only observation dict."""
    waist, left_arm, right_arm = _split_qpos(qpos)
    left_hand  = hand_qpos_12d[:6].astype(np.float32)
    right_hand = hand_qpos_12d[6:12].astype(np.float32)
    d = {
        "video.ego_left_view":  rgb_left[None, None, ...],
        "state.waist":          waist[None, None, :],
        "state.left_arm":       left_arm[None, None, :],
        "state.right_arm":      right_arm[None, None, :],
        "state.left_hand":      left_hand[None, None, :],
        "state.right_hand":     right_hand[None, None, :],
        "annotation.human.action.task_description": np.array([task_name], dtype=object),
    }
    if binocular and rgb_right is not None:
        d["video.ego_right_view"] = rgb_right[None, None, ...]
    return d


def action_to_array(action: dict, i: int) -> tuple:
    """Pull one timestep out of a GR00T action chunk.

    GR00T action keys (inspire-only DataConfig):
        action.waist     (1, T, 3)
        action.left_arm  (1, T, 7)   # may be 6 in some checkpoints -- handled here
        action.right_arm (1, T, 7)
        action.left_hand (1, T, 6)
        action.right_hand(1, T, 6)
    Returns (action_np[19], hand_action_np[12]) for write_shm.
    """
    waist     = np.asarray(action["action.waist"][0, i],     dtype=np.float32)  # (3,)
    left_arm  = np.asarray(action["action.left_arm"][0, i],  dtype=np.float32)  # (6 or 7,)
    right_arm = np.asarray(action["action.right_arm"][0, i], dtype=np.float32)  # (7,)
    left_h    = np.asarray(action["action.left_hand"][0, i], dtype=np.float32)  # (6,)
    right_h   = np.asarray(action["action.right_hand"][0,i], dtype=np.float32)  # (6,)
    head      = np.zeros(2, dtype=np.float32)                                   # head is not commanded

    arm14_left  = left_arm  # may be (6,) -> caller will pad with current qpos[11]
    action_np   = np.concatenate([waist, head, arm14_left, right_arm], axis=0)  # 3+2+(6|7)+7
    hand_action = np.concatenate([left_h, right_h], axis=0)                     # 12
    return action_np, hand_action


# =============================================================================
# Lazy GR00T policy loader
# =============================================================================

def init_gr00t_policy(model_path: str, data_config_key: str,
                      embodiment_tag: str = "new_embodiment",
                      denoising_steps: int = 4, device: str = "cuda"):
    """Build a Gr00tPolicy. Import is local so this module loads in any env."""
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    data_config = DATA_CONFIG_MAP[data_config_key]
    return Gr00tPolicy(
        model_path=model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
        device=device,
    )


# =============================================================================
# Inference worker
# =============================================================================

class Gr00t_Inference:
    """Inspire-only GR00T inference worker.

    - slow loop (default 20Hz): read obs, call policy, upsample to fast_hz,
      blend cross-fade with the tail of the previous chunk, store as
      `self.primary_actions / primary_hand_actions`.
    - fast loop (default 50Hz): pop one timestep, apply --action_method,
      write to ROBOT_ACTION.
    - policy is loaded lazily on the first slow tick where
      record_mode_shm.deploy is True. Until then both loops idle.
    """

    def __init__(
        self,
        shm_name: dict,
        shared_lock: dict,
        shared_event: dict,
        mode: str = "gr00t_zed",                # {"gr00t", "gr00t_zed"}
        model_path: str = "",
        data_config_key: str = "unitree_g1_inspire",
        embodiment_tag: str = "new_embodiment",
        action_method: str = "tem",             # {"base", "maf", "tem"}
        decay: float = 0.3,
        window_size: int = 5,
        slow_hz: float = 20.0,
        fast_hz: float = 50.0,
        denoising_steps: int = 4,
        binocular: bool = True,
        masking: bool = False,
    ):
        assert mode in ("gr00t", "gr00t_zed"), f"unsupported mode: {mode}"
        assert action_method in ("base", "maf", "tem"), f"unknown method: {action_method}"

        self.shared_event = shared_event
        self.mode         = mode
        self.zed_mode     = (mode == "gr00t_zed")
        self.binocular    = bool(binocular and self.zed_mode)
        self.masking      = bool(masking)
        self.model_path   = model_path
        self.data_config_key = data_config_key
        self.embodiment_tag  = embodiment_tag
        self.denoising_steps = int(denoising_steps)
        self.action_method   = action_method
        self.decay           = float(decay)
        self.window_size     = int(window_size)
        self.slow_hz         = float(slow_hz)
        self.fast_hz         = float(fast_hz)

        # SHM handles
        self._init_shm(shm_name, shared_lock)

        # State shared between slow / fast loops.
        self._obs_lock     = threading.Lock()
        self._ctrl_lock    = threading.Lock()
        self.policy        = None
        self.deploy_mode   = False
        self.start_loop    = False
        self.qpos          = np.zeros(19, dtype=np.float32)
        self.hand_qpos     = np.zeros(12, dtype=np.float32)
        self.task_name     = ""
        self.primary_actions      = None   # (M, 19)
        self.primary_hand_actions = None   # (M, 12)
        self.primary_index        = 0
        self._prev_actions        = None   # for cross-fade
        self._prev_hand_actions   = None
        self._action_buf          = deque(maxlen=self.window_size)
        self._hand_buf            = deque(maxlen=self.window_size)

        # Lifecycle
        self._stop_event = threading.Event()
        self._slow_thread = threading.Thread(target=self._inference_loop, daemon=True, name="DEPLOY_SLOW")
        self._fast_thread = threading.Thread(target=self._ctrl_loop,      daemon=True, name="DEPLOY_FAST")
        self._slow_thread.start()
        self._fast_thread.start()
        logger_mp.info(f"[Deploy] Gr00t_Inference started "
                       f"(mode={mode}, action_method={action_method}, "
                       f"slow={slow_hz}Hz, fast={fast_hz}Hz, binocular={self.binocular})")

    # ----- init helpers ------------------------------------------------------
    def _init_shm(self, shm_name, shared_lock):
        self.camera_shm        = SharedMemoryManager(CAMERA,             shared_lock["camera_lock"],         shm_name["camera_shm"])
        self.robot_obs_shm     = SharedMemoryManager(ROBOT_OBS,          shared_lock["robot_obs_lock"],      shm_name["robot_obs_shm"])
        self.robot_action_shm  = SharedMemoryManager(ROBOT_ACTION,       shared_lock["robot_action_lock"],   shm_name["robot_action_shm"])
        self.gr00t_task_shm    = SharedMemoryManager(GR00T_TASK_LAYOUT,  shared_lock["gr00t_lock"],          shm_name["gr00t_shm"])
        self.record_mode_shm   = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"],         shm_name["record_mode_shm"])
        if self.masking:
            self.workspace_mask_shm = SharedMemoryManager(WORKSPACE_MASK, shared_lock["workspace_mask_lock"], shm_name["workspace_mask_shm"])
        else:
            self.workspace_mask_shm = None

    def _init_gr00t_policy(self):
        try:
            task_name = self.gr00t_task_shm.read_data()["task_name"].item().strip()
        except Exception:
            task_name = ""
        self.task_name = task_name
        self.policy = init_gr00t_policy(
            model_path=self.model_path, data_config_key=self.data_config_key,
            embodiment_tag=self.embodiment_tag, denoising_steps=self.denoising_steps,
        )
        logger_mp.info(f"[Deploy] Policy loaded. Language instruction: {task_name!r}")

    # ----- observation -------------------------------------------------------
    def get_real_obs(self) -> Optional[dict]:
        """Pull a single observation dict from SHM. Returns None when frames are unavailable."""
        cam = self.camera_shm.read_data()
        ro  = self.robot_obs_shm.read_data()

        obs_waist = ro["obs_waist"]; obs_head = ro["obs_head"]
        obs_arm   = ro["obs_arm"];   obs_hand = ro["obs_hand"]
        qpos      = np.concatenate([obs_waist, obs_head, obs_arm]).astype(np.float32)  # 19
        hand_qpos = np.asarray(obs_hand, dtype=np.float32)                              # 12

        try:
            self.task_name = self.gr00t_task_shm.read_data()["task_name"].item().strip()
        except Exception:
            pass

        if self.zed_mode:
            left  = process_frame(cam["camera_left"],  side="zed_left")
            right = process_frame(cam["camera_right"], side="zed_right") if self.binocular else None
            if left is None:
                return None
            if self.masking and self.workspace_mask_shm is not None:
                m = self.workspace_mask_shm.read_data()
                ml = m["mask_left_flat"].reshape(left.shape[:2])
                left = (left * ml[..., None].astype(np.uint8)).astype(np.uint8) if ml.dtype != np.uint8 else left
                if right is not None:
                    mr = m["mask_right_flat"].reshape(right.shape[:2])
                    right = (right * mr[..., None].astype(np.uint8)).astype(np.uint8) if mr.dtype != np.uint8 else right
            obs = obs_dict_zed(self.task_name, qpos, hand_qpos, left, right,
                               binocular=self.binocular)
        else:
            rs = process_frame(cam["realsense"], side="realsense")
            if rs is None:
                return None
            obs = obs_dict_realsense(self.task_name, qpos, hand_qpos, rs)

        with self._obs_lock:
            self.qpos      = qpos
            self.hand_qpos = hand_qpos
        return obs

    # ----- inference + chunk handling ---------------------------------------
    def gr00t_inference(self, obs: dict):
        """Run policy.get_action -> upsample -> cross-fade -> latch primary_*"""
        if self.policy is None:
            return
        action = self.policy.get_action(obs)

        # Build (T, D) action and (T, D_hand) hand sequences from the chunk.
        T = action["action.waist"].shape[1]
        full = []
        hand = []
        for i in range(T):
            a_np, h_np = action_to_array(action, i)
            full.append(a_np)
            hand.append(h_np)
        full = np.stack(full, axis=0)  # (T, 19 or 18 if left_arm=6)
        hand = np.stack(hand, axis=0)  # (T, 12)

        # Upsample 20Hz -> 50Hz with quintic spline
        full_up = upsample_actions(full, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=5)
        hand_up = upsample_actions(hand, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=5)

        # Cross-fade with the tail of the previous chunk if it is still in flight.
        if self._prev_actions is not None and self.primary_index < len(self._prev_actions):
            remain_prev = self._prev_actions[self.primary_index:]
            ovl = min(len(remain_prev), len(full_up))
            if ovl > 0:
                alpha = np.linspace(0.0, 1.0, ovl)[:, None]
                full_up[:ovl] = (1 - alpha) * remain_prev[:ovl] + alpha * full_up[:ovl]
                hand_up[:ovl] = (1 - alpha) * self._prev_hand_actions[self.primary_index:][:ovl] + alpha * hand_up[:ovl]

        with self._ctrl_lock:
            self._prev_actions      = self.primary_actions
            self._prev_hand_actions = self.primary_hand_actions
            self.primary_actions      = full_up
            self.primary_hand_actions = hand_up
            self.primary_index        = 0
            self._action_buf.clear()
            self._hand_buf.clear()
        self.start_loop = True

    # ----- action post-processing -------------------------------------------
    def get_action(self):
        """Return (action_np, hand_action_np) for one publish tick or (None, None) if not ready."""
        with self._ctrl_lock:
            if self.primary_actions is None or self.primary_index >= len(self.primary_actions):
                return None, None
            i = self.primary_index
            a = self.primary_actions[i]
            h = self.primary_hand_actions[i]

            if self.action_method == "base":
                out_a, out_h = a, h
                self.primary_index += 1
            elif self.action_method == "maf":
                self._action_buf.append(a); self._hand_buf.append(h)
                out_a = np.mean(np.stack(self._action_buf, axis=0), axis=0)
                out_h = np.mean(np.stack(self._hand_buf,   axis=0), axis=0)
                self.primary_index += 1
            elif self.action_method == "tem":
                remain = len(self.primary_actions) - i
                N = min(remain, self.window_size)
                w = np.exp(-self.decay * np.arange(N))
                w = w / w.sum()
                out_a = (w[:, None] * self.primary_actions[i:i+N]).sum(axis=0)
                out_h = (w[:, None] * self.primary_hand_actions[i:i+N]).sum(axis=0)
                self.primary_index += 1
            else:
                out_a, out_h = a, h
                self.primary_index += 1
        return out_a, out_h

    # ----- SHM write ---------------------------------------------------------
    def write_shm(self, action_np: Optional[np.ndarray], hand_action_np: Optional[np.ndarray]):
        if action_np is None or hand_action_np is None:
            return
        action_waist = action_np[:3]
        action_head  = action_np[3:5]
        arm_slice    = action_np[5:]

        # left_arm 6D-fallback: if model only outputs 6 left-arm joints,
        # keep current qpos[11] (last left-arm dof) and use 7D for right.
        if arm_slice.size == 13:
            left7 = self.qpos[5:12].copy().astype(np.float32)
            left7[:6] = arm_slice[:6]
            right7 = arm_slice[6:]
            action_arm = np.concatenate([left7, right7], axis=0)
        else:
            action_arm = arm_slice  # 14D

        self.robot_action_shm.write_data(
            action_waist     =action_waist.astype(np.float64),
            action_waist_tauff=np.zeros(3, dtype=np.float64),
            action_head      =action_head.astype(np.float64),
            action_arm       =action_arm.astype(np.float64),
            action_arm_tauff =np.zeros(14, dtype=np.float64),
            action_hand      =hand_action_np[:12].astype(np.float64),
        )

    # ----- UI trigger --------------------------------------------------------
    def get_ui_mode(self):
        try:
            self.deploy_mode = bool(self.record_mode_shm.read_data()["deploy"])
        except Exception:
            pass

    # ----- per-loop work units ----------------------------------------------
    def do_slow(self):
        obs = self.get_real_obs()
        if obs is None:
            return
        self.gr00t_inference(obs)

    def do_fast(self):
        a, h = self.get_action()
        self.write_shm(a, h)

    # ----- loops -------------------------------------------------------------
    def _inference_loop(self):
        period = 1.0 / self.slow_hz
        next_t = time.perf_counter()
        while not self._stop_event.is_set() and not self.shared_event["shutdown"].is_set():
            self.get_ui_mode()
            if self.deploy_mode:
                if self.policy is None:
                    try:
                        self._init_gr00t_policy()
                    except Exception as e:
                        logger_mp.exception(f"[Deploy] policy init failed: {e}")
                        time.sleep(1.0); continue
                try:
                    self.do_slow()
                except Exception as e:
                    logger_mp.exception(f"[Deploy] slow loop error: {e}")
            else:
                # deploy off -> idle, drop any pending chunk so we don't resume stale
                with self._ctrl_lock:
                    self.primary_actions = None
                    self.primary_hand_actions = None
                    self.primary_index = 0
                self.start_loop = False
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def _ctrl_loop(self):
        period = 1.0 / self.fast_hz
        next_t = time.perf_counter()
        while not self._stop_event.is_set() and not self.shared_event["shutdown"].is_set():
            if self.start_loop:
                try:
                    self.do_fast()
                except Exception as e:
                    logger_mp.exception(f"[Deploy] fast loop error: {e}")
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    # ----- shutdown ----------------------------------------------------------
    def stop(self):
        self._stop_event.set()
        self._slow_thread.join(timeout=2.0)
        self._fast_thread.join(timeout=2.0)
        self._cleanup()

    def _cleanup(self):
        for s in [self.camera_shm, self.robot_obs_shm, self.robot_action_shm,
                  self.gr00t_task_shm, self.record_mode_shm, self.workspace_mask_shm]:
            if s is not None:
                try: s.worker_close()
                except Exception: pass
        logger_mp.info("[Deploy] Gr00t_Inference stopped.")
