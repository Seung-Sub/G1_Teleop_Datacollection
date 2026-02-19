# 표준 라이브러리
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

import time  
import threading 

# 과학/데이터 라이브러리
import numpy as np  
import torch  
import cv2  
import pandas as pd 
from pathlib import Path  
from collections import deque  
from scipy.interpolate import make_interp_spline

import copy 

# 시각화 라이브러리
import matplotlib 
matplotlib.use('Agg')  
import matplotlib.pyplot as plt 

# 공유 메모리 관리 모듈
from sharedmemory.shmManager import SharedMemoryManager 
from sharedmemory.shm_schema import CAMERA, ROBOT_ACTION, ROBOT_OBS, GR00T_TASK_LAYOUT, RECORD_MODE_LAYOUT, KISTAR_HAND_RECEIVED, KISTAR_HAND_ACTION, WORKSPACE_MASK

# 터미널 UI(Rich)
from rich.live import Live  
from rich.table import Table 
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn  
from rich.console import Group  

# ----- (NEW 0211): PCA helpers -----
# PCA actiom basis 로드 + 디코더 함수 추가
PCA_BASIS_ACTION_PATH = "/home/ansur/G1_teleoperation/record/0209_apple_pickup_pcaK7/pca_basis_action_k7.npz"


_PCA_MU = None
_PCA_W = None
_PCA_SCALE = None

def load_pca_basis_once(npz_path=PCA_BASIS_ACTION_PATH):
    global _PCA_MU, _PCA_W, _PCA_SCALE
    if _PCA_MU is not None:
        return
    z = np.load(npz_path)
    _PCA_MU = z["mu"].astype(np.float32)       # (16,)
    _PCA_W  = z["W"].astype(np.float32)        # (16,7)
    _PCA_SCALE = z["scale"].astype(np.float32) # (7,)
    # 안전장치
    assert _PCA_W.shape[0] == 16, f"W rows should be 16, got {_PCA_W.shape}"
    assert _PCA_MU.shape[0] == 16, f"mu should be (16,), got {_PCA_MU.shape}"
    assert _PCA_SCALE.shape[0] == _PCA_W.shape[1], "scale dim should match K"

def expand_hand_action_pca(hand_action_7d: np.ndarray):
    """
    모델이 출력한 7차원(PCA synergy) 손 액션을 16차원으로 확장.
    - hand_action_7d가 7D가 아니면(이미 16D거나 다른 모드), 그대로 반환.
    """
    # basis 아직 안 읽었으면 로드
    if _PCA_MU is None:
        load_pca_basis_once()

    hand_action_7d = np.asarray(hand_action_7d, dtype=np.float32).reshape(-1)

    #  차원 안 맞으면 그대로 반환
    if hand_action_7d.shape[0] != _PCA_W.shape[1]:
        return hand_action_7d

    # 정규화 되돌리기: s_raw = s_norm * scale
    s_raw = hand_action_7d * _PCA_SCALE               # (7,)

    # 복원: rh16 = mu + W @ s_raw
    expanded_action = _PCA_MU + (_PCA_W @ s_raw)       # (16,)

    return expanded_action.astype(np.float32)

#PCA obs basis로드 + 인코더 함수 추가
PCA_BASIS_OBS_PATH = "/home/ansur/G1_teleoperation/record/0209_apple_pickup_pcaK7/pca_basis_obs_k7.npz"

_PCA_MU_O = None
_PCA_W_O = None
_PCA_SCALE_O = None

def load_pca_obs_basis_once(npz_path=PCA_BASIS_OBS_PATH):
    global _PCA_MU_O, _PCA_W_O, _PCA_SCALE_O
    if _PCA_MU_O is not None:
        return
    z = np.load(npz_path)
    _PCA_MU_O = z["mu"].astype(np.float32)        # (16,)
    _PCA_W_O  = z["W"].astype(np.float32)         # (16,7)
    _PCA_SCALE_O = z["scale"].astype(np.float32)  # (7,)
    assert _PCA_W_O.shape[0] == 16
    assert _PCA_SCALE_O.shape[0] == _PCA_W_O.shape[1]

def encode_hand_qpos_pca(hand_qpos_16d: np.ndarray, clip: float = 1.0):
    """
    오른손 qpos 16D -> PCA synergy 7D (정규화된 값)
    - 16D 아니면 그대로 반환
    """
    if _PCA_MU_O is None:
        load_pca_obs_basis_once()

    x = np.asarray(hand_qpos_16d, dtype=np.float32).reshape(-1)
    if x.shape[0] != 16:
        return x

    x0 = x - _PCA_MU_O            # (16,)
    s_raw = x0 @ _PCA_W_O         # (7,)
    s_norm = s_raw / _PCA_SCALE_O # (7,)

    if clip is not None:
        s_norm = np.clip(s_norm, -clip, clip)

    return s_norm.astype(np.float32)
# ----- (END 0211): PCA helpers -----


# 프로젝트(gr00t) 전용 모듈
# ----- (NEW 0210): lazy import helper -----
# Gr00t imports are heavy (they pull optional deps like pytorch3d).
# Make them LAZY so `--policy maniflow` can run without installing everything.
gr00t = None
DATA_CONFIG_MAP = None
Gr00tPolicy = None

def _lazy_import_gr00t():
    global gr00t, DATA_CONFIG_MAP, Gr00tPolicy

    if gr00t is None:
        import gr00t as _gr00t
        gr00t = _gr00t

    if DATA_CONFIG_MAP is None:
        from gr00t.experiment.data_config import DATA_CONFIG_MAP as _DCM
        DATA_CONFIG_MAP = _DCM

    if Gr00tPolicy is None:
        from gr00t.model.policy import Gr00tPolicy as _GP
        Gr00tPolicy = _GP

    return DATA_CONFIG_MAP, Gr00tPolicy
# ----- (END 0210): lazy import helper -----

# ACT 전용 모듈
# [변경] einops.rearrange 활성화: 이미지 텐서 재배열용 (원본은 주석처리되어 있었음)
from einops import rearrange
#from act.policy import ACTPolicy, CNNMLPPolicy
import yaml
import pickle

# ----- (NEW 0209): new imports for maniflow -----
import contextlib
import math
from datetime import datetime
try:
    import dill
    from omegaconf import OmegaConf
except Exception:
    dill = None
    OmegaConf = None
# ----- (END 0209): new imports for maniflow -----

# ----- (NEW 0209): helpers for maniflow policy adapter -----
def _strip_module_prefix(sd):
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}

def _register_omegaconf_resolvers():
    """Make saved Hydra/OmegaConf configs usable in a plain python script."""
    if OmegaConf is None:
        return

    import re

    def _safe_eval(expr: str):
        allowed = {"min": min, "max": max, "int": int, "float": float, "round": round, "math": math}
        return eval(expr, {"__builtins__": {}}, allowed)

    # ${eval:...}
    OmegaConf.register_new_resolver("eval", _safe_eval, replace=True)

    # ${now:...} (Hydra often uses this)
    def _now(fmt: str = "%Y-%m-%d_%H-%M-%S"):
        return datetime.now().strftime(fmt)

    OmegaConf.register_new_resolver("now", _now, replace=True)
# ----- (END 0209): helpers for maniflow policy adapter -----


# ----- (NEW 0209): maniflow policy adapter -----
class ManiFlowPolicyAdapter:
    """
    Drop-in replacement for Gr00tPolicy that satisfies:
      - update_obs(obs_gr00t): called at 20Hz
      - get_action(obs_gr00t): returns dict with keys action.waist/left_arm/right_arm/inspire_hand/kistar_hand

    Assumption (matches your checkpoint printouts):
      - obs keys expected by ManiFlow:
          head_cam_left:  (1, n_obs_steps, 3, 240, 320) float in [0,1]
          head_cam_right: (1, n_obs_steps, 3, 240, 320)
          agent_pos:      (1, n_obs_steps, 32)2
      - action predicted: (1, T, 32)
      - layout = [waist(3), left_arm(7), right_arm(7), inspire_hand(6), kistar_hand(9)] => 32
        which corresponds to hand_mode = reduced_v3 in this repo.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_ema: bool = True,
        num_inference_steps: int = 2,
        denoise_timesteps: int = 2,
        img_hw=(240, 320),  # (H,W)
        compile_model: bool = True,
        expected_layout: str = "inspire_kistar_reduced_v3",
    ):
        if dill is None or OmegaConf is None:
            raise ImportError(
                "ManiFlow adapter needs dill + omegaconf installed in this env. "
                "Try: pip install dill omegaconf hydra-core"
            )

        self.ckpt_path = ckpt_path
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.use_ema = use_ema
        self.num_inference_steps = num_inference_steps
        self.denoise_timesteps = denoise_timesteps
        self.H, self.W = img_hw
        self.expected_layout = expected_layout

        # Reduce CPU noise
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
            except Exception:
                pass

        _register_omegaconf_resolvers()

        # ---- load checkpoint payload (your cell 1/2 logic) ----
        payload = torch.load(self.ckpt_path, map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]

        # Lazy import hydra instantiate (keeps import errors clearer)
        from hydra.utils import instantiate

        student = instantiate(cfg.policy)

        # choose EMA weights by default (your preference)
        if use_ema and "ema_model" in payload.get("state_dicts", {}):
            sd = payload["state_dicts"]["ema_model"]
            self.policy = instantiate(cfg.policy)
            self.policy.load_state_dict(_strip_module_prefix(sd), strict=True)
            print("[ManiFlow] Loaded EMA weights: state_dicts['ema_model']")
        else:
            sd = payload["state_dicts"]["model"]
            self.policy = student
            self.policy.load_state_dict(_strip_module_prefix(sd), strict=True)
            print("[ManiFlow] Loaded Student weights: state_dicts['model']")

        self.policy.to(self.device)
        self.policy.eval()

        # set diffusion inference steps to your fast setting
        if hasattr(self.policy, "num_inference_steps"):
            self.policy.num_inference_steps = num_inference_steps
        if hasattr(self.policy, "denoise_timesteps"):
            self.policy.denoise_timesteps = denoise_timesteps

        # ----- (NEW 0210): make normalizer attachment "fail-fast" -----
        # ---- normalizer (critical!) ----
        # Your checkpoints contain the normalizer buffers inside the model state_dict
        # (e.g. keys like `normalizer.params_dict.*`), so load_state_dict() already
        # restores correct normalization. Do NOT try to instantiate the dataset here.
        norm_keys = [k for k in sd.keys() if "normalizer" in k.lower()]
        if not norm_keys:
            raise RuntimeError(
                "[DEPLOY][FATAL] Checkpoint state_dict has no normalizer.* keys. "
                "Refusing to run ManiFlow without correct normalization."
            )

        if (not hasattr(self.policy, "normalizer")) or (self.policy.normalizer is None):
            raise RuntimeError(
                "[DEPLOY][FATAL] ManiFlow policy has no `.normalizer` (or it is None) even though the checkpoint "
                "contains normalizer keys. Check ManiFlowTransformerImagePolicy defines `self.normalizer` and that "
                "state_dict names match."
            )

        print(f"[ManiFlow] Using normalizer from checkpoint (loaded via load_state_dict). normalizer_keys={len(norm_keys)}")
        # ----- (END 0210): make normalizer attachment "fail-fast" -----

        # n_obs_steps is needed for buffering
        self.n_obs_steps = int(getattr(self.policy, "n_obs_steps", 2))

        # --- rolling buffers updated at 20Hz ---
        self._imgL = deque(maxlen=self.n_obs_steps)   # each: (3,H,W) float16 in [0,1]
        self._imgR = deque(maxlen=self.n_obs_steps)
        self._state = deque(maxlen=self.n_obs_steps)  # each: (32,) float32

        # --- optional torch.compile (your cell 6.9/7 idea) ---
        if compile_model and hasattr(torch, "compile"):
            try:
                if hasattr(self.policy, "model"):
                    self.policy.model = torch.compile(self.policy.model, mode="reduce-overhead", fullgraph=False)
                    print("[ManiFlow] torch.compile(policy.model) ok")
                if hasattr(self.policy, "obs_encoder"):
                    self.policy.obs_encoder = torch.compile(self.policy.obs_encoder, mode="reduce-overhead", fullgraph=False)
                    print("[ManiFlow] torch.compile(policy.obs_encoder) ok")
            except Exception as e:
                print(f"[ManiFlow][WARN] torch.compile failed (ignored): {repr(e)}")

        # warmup (important for stable latency)
        self._warmup()

    def _warmup(self):
        # create dummy obs with fixed shapes so kernels/compile warm up
        B = 1
        T = self.n_obs_steps
        dummy = {
            "head_cam_left": torch.zeros((B, T, 3, self.H, self.W), device=self.device, dtype=torch.float16),
            "head_cam_right": torch.zeros((B, T, 3, self.H, self.W), device=self.device, dtype=torch.float16),
            "agent_pos": torch.zeros((B, T, 32), device=self.device, dtype=torch.float32),
        }
        for _ in range(10):
            _ = self._fast_predict(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[ManiFlow] Warmup done.")

    def _fast_predict(self, obs_torch: dict):
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            amp_ctx = torch.autocast("cuda", dtype=torch.float16)
        else:
            amp_ctx = contextlib.nullcontext()

        with torch.inference_mode():
            with amp_ctx:
                return self.policy.predict_action(obs_torch)

    @staticmethod
    def _extract_frame(obs: dict, key_candidates):
        for k in key_candidates:
            if k in obs:
                arr = obs[k]
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 5:
                        return arr[0, 0]  # (H,W,3)
                    if arr.ndim == 3:
                        return arr
        return None

    @staticmethod
    def _extract_vec(obs: dict, k: str):
        if k not in obs:
            return None
        arr = obs[k]
        if not isinstance(arr, np.ndarray):
            return None
        if arr.ndim == 3:
            return arr[0, 0].astype(np.float32)
        if arr.ndim == 1:
            return arr.astype(np.float32)
        return None

    def _preprocess_rgb(self, rgb_uint8: np.ndarray):
        # rgb_uint8: (H,W,3) uint8 RGB
        img = cv2.resize(rgb_uint8, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.transpose(2, 0, 1)  # (3,H,W)
        img = (img.astype(np.float16) / 255.0)
        return img

    def update_obs(self, obs_gr00t: dict):
        """
        Called at 20Hz: store last n_obs_steps frames + state.
        """
        if obs_gr00t is None:
            return

        # Images (ZED bino preferred)
        frameL = self._extract_frame(obs_gr00t, ["video.ego_left_view", "video.ego_view", "video.rs_view"])
        frameR = self._extract_frame(obs_gr00t, ["video.ego_right_view"])

        if frameL is None:
            return
        if frameR is None:
            frameR = frameL  # fallback: duplicate

        # State -> agent_pos(32)
        waist = self._extract_vec(obs_gr00t, "state.waist")
        la   = self._extract_vec(obs_gr00t, "state.left_arm")
        ra   = self._extract_vec(obs_gr00t, "state.right_arm")

        # left hand key differs by branch; handle both
        lh = self._extract_vec(obs_gr00t, "state.inspire_hand")
        if lh is None:
            lh = self._extract_vec(obs_gr00t, "state.left_hand")

        kh = self._extract_vec(obs_gr00t, "state.kistar_hand")

        if waist is None or la is None or ra is None or lh is None or kh is None:
            return

        # If kistar came in as 16D but we expect reduced_v3(9D), reduce it.
        if self.expected_layout == "inspire_kistar_reduced_v3":
            if kh.shape[0] == 16:
                kh = reduce_hand_qpos_v3(kh).astype(np.float32)
            if kh.shape[0] != 9:
                # last-resort: try reduced(6) and pad to 9
                if kh.shape[0] == 6:
                    kh = np.pad(kh, (0, 3), mode="constant")
        # build agent_pos in the assumed order
        agent_pos = np.concatenate([waist, la, ra, lh, kh], axis=0).astype(np.float32)

        # Safety check
        if agent_pos.shape[0] != 32:
            # Don’t crash the whole deploy loop; just skip update
            return

        self._imgL.append(self._preprocess_rgb(frameL))
        self._imgR.append(self._preprocess_rgb(frameR))
        self._state.append(agent_pos)

    def _pad_seq(self, dq: deque, n: int):
        if len(dq) == 0:
            return None
        if len(dq) >= n:
            return list(dq)[-n:]
        first = dq[0]
        return [first] * (n - len(dq)) + list(dq)

    def get_action(self, obs_gr00t: dict):
        """
        Called only when the deploy buffer needs refill.
        Returns a GR00T-style action dict with shape (1, T, dim) numpy arrays.
        """
        # If we never received updates (e.g., first call), try updating once
        if len(self._state) == 0:
            self.update_obs(obs_gr00t)

        imgsL = self._pad_seq(self._imgL, self.n_obs_steps)
        imgsR = self._pad_seq(self._imgR, self.n_obs_steps)
        states = self._pad_seq(self._state, self.n_obs_steps)

        if imgsL is None or imgsR is None or states is None:
            print(f"[DEBUG] imgsL: {imgsL is None}")
            print(f"[DEBUG] imgsR: {imgsR is None}")
            print(f"[DEBUG] states: {states is None}")
            raise RuntimeError("[ManiFlow] No observation history available yet.")

        # Stack -> (1, T, ...)
        head_cam_left = torch.from_numpy(np.stack(imgsL, axis=0)).unsqueeze(0).to(self.device, non_blocking=True)
        head_cam_right = torch.from_numpy(np.stack(imgsR, axis=0)).unsqueeze(0).to(self.device, non_blocking=True)
        agent_pos = torch.from_numpy(np.stack(states, axis=0)).unsqueeze(0).to(self.device, non_blocking=True)

        obs_mf = {
            "head_cam_left": head_cam_left,     # (1, T, 3, H, W)
            "head_cam_right": head_cam_right,   # (1, T, 3, H, W)
            "agent_pos": agent_pos,             # (1, T, 32)
        }

        out = self._fast_predict(obs_mf)
        a = out["action"] if isinstance(out, dict) and "action" in out else out
        a = a.detach().float().cpu().numpy()  # (1, Tpred, 32)

        # Split action layout
        waist = a[:, :, 0:3]
        left_arm = a[:, :, 3:10]
        right_arm = a[:, :, 10:17]
        inspire_hand = a[:, :, 17:23]
        kistar_hand = a[:, :, 23:32]  # 9D (reduced_v3)

        return {
            "action.waist": waist,
            "action.left_arm": left_arm,
            "action.right_arm": right_arm,
            "action.inspire_hand": inspire_hand,
            "action.kistar_hand": kistar_hand,
        }
# ----- (END 0209): maniflow policy adapter -----


def process_frame(frame, side=""):
    """하나의 프레임을 검증하고 전처리한 뒤 uint8 RGB 이미지로 반환합니다."""
    if not isinstance(frame, np.ndarray):
        print(f"[VIDEO] {side} 프레임 타입 오류: {type(frame)}")
        return None

    if frame.ndim == 2:
        img = np.stack((frame,) * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        img = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        img = frame[..., :3]
    elif frame.ndim == 3 and frame.shape[2] == 3:
        img = frame
    else:
        print(f"[VIDEO] {side} 지원하지 않는 프레임 shape: {frame.shape}")
        return None

    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print(f"[VIDEO] {side} cvtColor 실패: {e}")
        return None

    return img_rgb.astype(np.uint8)

# ----- (NEW 0210): new init groot using lazy import (since maniflow doesn't need gr00t) -----
def init_gr00t_policy(model_path, embodiment_tag, modality_config, modality_transform, denoising_steps, device):
    _, _Gr00tPolicy = _lazy_import_gr00t()

    policy = _Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
    )
    return policy
# ----- (END 0210): new init groot using lazy import (since maniflow doesn't need gr00t) -----

def reduce_hand_qpos(hand_qpos_16d):
    
    if hand_qpos_16d.shape[0] != 16:
        return hand_qpos_16d # 이미 축소되었거나 다른 차원이면 그대로 반환

    # 엄지 (2), 검지 (3), 중지 (3), 약지 (3), 새끼 (3), 나머지 (2)
    thumb_1 = hand_qpos_16d[0]
    thumb_2 = hand_qpos_16d[1]
    thumb_flexion = (hand_qpos_16d[2] + hand_qpos_16d[3]) / 2
    index_flexion = (3*hand_qpos_16d[5] + hand_qpos_16d[6] + hand_qpos_16d[7]) / 5
    middle_flexion = (3*hand_qpos_16d[9] + hand_qpos_16d[10] + hand_qpos_16d[11]) / 5
    ring_flexion = (3*hand_qpos_16d[13] + hand_qpos_16d[14] + hand_qpos_16d[15]) / 5
    
    return np.array([
        thumb_1, thumb_2, thumb_flexion, index_flexion, middle_flexion, ring_flexion
    ], dtype=np.float32)

def reduce_hand_qpos_v3(hand_qpos_16d):
    
    if hand_qpos_16d.shape[0] != 16:
        return hand_qpos_16d # 이미 축소되었거나 다른 차원이면 그대로 반환

    # 엄지 (2), 검지 (3), 중지 (3), 약지 (3), 새끼 (3), 나머지 (2)
    thumb_1 = hand_qpos_16d[0]
    thumb_2 = hand_qpos_16d[1]
    thumb_3 = hand_qpos_16d[2]
    thumb_4 = hand_qpos_16d[3]

    index_1 = hand_qpos_16d[5]
    index_flexion = (hand_qpos_16d[6]+hand_qpos_16d[7])/2

    middle_1 = hand_qpos_16d[9]
    middle_flexion = (hand_qpos_16d[10]+hand_qpos_16d[11])/2

    ring_flexion = (3*hand_qpos_16d[13] + hand_qpos_16d[14] + hand_qpos_16d[15]) / 5
    
    return np.array([
        thumb_1, thumb_2, thumb_3, thumb_4, index_1, index_flexion, middle_1, middle_flexion, ring_flexion
    ], dtype=np.float32)

def obs_dict_realsense(gr00t_task_name, qpos, hand_qpos, rgb, hand_deploy_mode = "inspire_kistar_full"):

    video_np = rgb[np.newaxis, np.newaxis, ...]            # (1, 1, H, W, 3)

    waist      = qpos[:3]
    left_arm   = qpos[5:12]
    right_arm  = qpos[12:19]

    waist_np      = waist[np.newaxis, np.newaxis, ...]     # (1, 1, 3)
    left_arm_np   = left_arm[np.newaxis, np.newaxis, ...]  # (1, 1, 7)
    right_arm_np  = right_arm[np.newaxis, np.newaxis, ...] # (1, 1, 7)

            # 3) Annotation(문자열)도 numpy array로 변환
            #    문자열은 object dtype 배열로 감싸서 (1,1) 형태로
    task_name     = gr00t_task_name                        # str
    annotation_np = np.array([[task_name]], dtype=object)  # (1, 1)

    if hand_deploy_mode == "inspire_kistar_full":
        right_hand = hand_qpos[12:]
        right_hand_np = right_hand[np.newaxis, np.newaxis, ...]# (1, 1, 16)
        
        left_hand = hand_qpos[0:6]
        left_hand_np = left_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
        
        obs = {
                "video.ego_view": video_np,
                "state.waist": waist_np,
                "state.left_arm": left_arm_np,
                "state.right_arm": right_arm_np,
                "state.inspire_hand": left_hand_np,
                "state.kistar_hand": right_hand_np,
                "annotation.human.action.task_description": annotation_np,
            }

    elif hand_deploy_mode == "inspire_kistar_reduced":

        right_hand = reduce_hand_qpos(hand_qpos[12:])
        right_hand_np = right_hand[np.newaxis, np.newaxis, ...] # (1, 1, 6)
        
        left_hand = hand_qpos[0:6]
        left_hand_np = left_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
        
        obs = {
                "video.ego_view": video_np,
                "state.waist": waist_np,
                "state.left_arm": left_arm_np,
                "state.right_arm": right_arm_np,
                "state.inspire_hand": left_hand_np,
                "state.kistar_hand": right_hand_np,
                "annotation.human.action.task_description": annotation_np,
            }
        
    elif hand_deploy_mode == "kistar_only_full":
        right_hand = hand_qpos[0:16]
        right_hand_np = right_hand[np.newaxis, np.newaxis, ...]
        
        obs = {
                "video.ego_view": video_np,
                "state.waist": waist_np,
                "state.left_arm": left_arm_np,
                "state.right_arm": right_arm_np,
                "state.kistar_hand": right_hand_np,
                "annotation.human.action.task_description": annotation_np,
            }
                
    elif hand_deploy_mode == "kistar_only_reduced":
        right_hand = reduce_hand_qpos(hand_qpos[:16])
        right_hand_np = right_hand[np.newaxis, np.newaxis, ...] 
        
        obs = {
                "video.ego_view": video_np,
                "state.waist": waist_np,
                "state.left_arm": left_arm_np,
                "state.right_arm": right_arm_np,
                "state.kistar_hand": right_hand_np,
                "annotation.human.action.task_description": annotation_np,
            }
        
    elif hand_deploy_mode == "inspire_only":
        left_hand  = hand_qpos[:6]
        right_hand = hand_qpos[6:]
        left_hand_np  = left_hand[np.newaxis, np.newaxis, ...] # (1, 1, 6)
        right_hand_np = right_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
        
        
        obs = {
                    "video.rs_view": video_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.left_hand": left_hand_np,
                    "state.right_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
        
    return obs

# 특징: zed_left, zed_right 비디오, head 상태, kistar_hand(16차원) 포함
def obs_dict_zed(gr00t_task_name, qpos, hand_qpos, rgb_left, rgb_right, hand_deploy_mode = "inspire_kistar_full", binocular=False):
    """ZED 양안 카메라를 위한 관측 딕셔너리 생성"""
    
    # ZED 양안 비디오 처리
    video_left_np = rgb_left[np.newaxis, np.newaxis, ...]   # (1, 1, H, W, 3)
    video_right_np = rgb_right[np.newaxis, np.newaxis, ...] # (1, 1, H, W, 3)

    waist      = qpos[:3]
    left_arm   = qpos[5:12]
    right_arm  = qpos[12:19]

    waist_np      = waist[np.newaxis, np.newaxis, ...]     # (1, 1, 3)
    left_arm_np   = left_arm[np.newaxis, np.newaxis, ...]  # (1, 1, 7)
    right_arm_np  = right_arm[np.newaxis, np.newaxis, ...] # (1, 1, 7)

    # ----- (NEW 0211): head obs for PCA -----
    head = np.zeros(2, dtype=np.float32)
    head_np = head[np.newaxis, np.newaxis, ...]
    # ----- (END 0211): head obs for PCA -----

    # Annotation(문자열)도 numpy array로 변환
    task_name     = gr00t_task_name                        # str
    annotation_np = np.array([[task_name]], dtype=object)  # (1, 1)

    if binocular:
        
        if hand_deploy_mode == "inspire_kistar_full":
            right_hand = hand_qpos[12:]
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...]# (1, 1, 16)
            
            left_hand = hand_qpos[0:6]
            left_hand_np = left_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "video.ego_right_view": video_right_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.inspire_hand": left_hand_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
            
        elif hand_deploy_mode == "inspire_kistar_reduced":

            right_hand = reduce_hand_qpos(hand_qpos[12:])
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...] # (1, 1, 6)
            
            left_hand = hand_qpos[0:6]
            left_hand_np = left_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "video.ego_right_view": video_right_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.inspire_hand": left_hand_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
            
        elif hand_deploy_mode == "inspire_kistar_reduced_v3":

            right_hand = reduce_hand_qpos_v3(hand_qpos[12:])
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...] # (1, 1, 6)
            
            left_hand = hand_qpos[0:6]
            left_hand_np = left_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "video.ego_right_view": video_right_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.inspire_hand": left_hand_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }            
            
        # ----- (NEW 0211): proprioceptive state for left/right hand PCA -----
        elif hand_deploy_mode == "inspire_kistar_pca":
            # 왼손(inspire) 6D 그대로
            left_hand = hand_qpos[0:6]
            left_hand_np = left_hand[np.newaxis, np.newaxis, ...]  # (1,1,6)

            # 오른손(kistar) 16D -> PCA 7D
            right_hand16 = hand_qpos[12:]                          # (16,)
            right_hand7 = encode_hand_qpos_pca(right_hand16, clip=1.0)  # (7,)
            right_hand_np = right_hand7[np.newaxis, np.newaxis, ...]     # (1,1,7)

            obs = {
                    "video.ego_left_view": video_left_np,
                    "video.ego_right_view": video_right_np,
                    "state.waist": waist_np,
                    "state.head" : head_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.left_hand": left_hand_np,
                    "state.right_hand_pca": right_hand_np,  # ✅ 7D
                    "annotation.human.action.task_description": annotation_np,
                }
            print(f"[DEBUG] obs: {obs}")
        # ----- (END 0211): proprioceptive state for left/right hand PCA -----
        
        
        elif hand_deploy_mode == "kistar_only":
            right_hand = hand_qpos[0:16]
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...]
            
            obs = {
                    "video.ego_view": video_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }



        elif hand_deploy_mode == "kistar_only_full":
            right_hand = hand_qpos[12:]
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...] 
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "video.ego_right_view": video_right_np,
                    "state.waist": waist_np,
                    "state.head" : head_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
                    
        elif hand_deploy_mode == "kistar_only_reduced":
            right_hand = reduce_hand_qpos(hand_qpos[12:])
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...] 
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "video.ego_right_view": video_right_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
            
        elif hand_deploy_mode == "inspire_only":
            left_hand  = hand_qpos[:6]
            right_hand = hand_qpos[6:]
            left_hand_np  = left_hand[np.newaxis, np.newaxis, ...] # (1, 1, 6)
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "video.ego_right_view": video_right_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.left_hand": left_hand_np,
                    "state.right_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
                
        return obs

    ## 단안 모드 
    else:
        if hand_deploy_mode == "inspire_kistar_full":
            right_hand = hand_qpos[12:]
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...]# (1, 1, 16)
            
            left_hand = hand_qpos[0:6]
            left_hand_np = left_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.left_hand": left_hand_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
            
        elif hand_deploy_mode == "inspire_kistar_reduced":

            right_hand = reduce_hand_qpos(hand_qpos[12:])
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...] # (1, 1, 6)
            
            left_hand = hand_qpos[0:6]
            left_hand_np = left_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.left_hand": left_hand_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
            
        elif hand_deploy_mode == "kistar_only_full":
            right_hand = hand_qpos[12:]
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...] 
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
                    
        elif hand_deploy_mode == "kistar_only_reduced":
            right_hand = reduce_hand_qpos(hand_qpos[12:])
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...] 
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.kistar_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
            
        elif hand_deploy_mode == "inspire_only":
            left_hand  = hand_qpos[:6]
            right_hand = hand_qpos[6:]
            left_hand_np  = left_hand[np.newaxis, np.newaxis, ...] # (1, 1, 6)
            right_hand_np = right_hand[np.newaxis, np.newaxis, ...]# (1, 1, 6)            
            
            obs = {
                    "video.ego_left_view": video_left_np,
                    "state.waist": waist_np,
                    "state.left_arm": left_arm_np,
                    "state.right_arm": right_arm_np,
                    "state.left_hand": left_hand_np,
                    "state.right_hand": right_hand_np,
                    "annotation.human.action.task_description": annotation_np,
                }
                
        return obs

def expand_hand_action(hand_action_6d):
    """모델이 출력한 6차원 손 액션을 16차원으로 확장합니다."""
    if hand_action_6d.shape[0] != 6:
        return hand_action_6d # 이미 확장되었거나 다른 차원이면 그대로 반환

    thumb_1, thumb_2, thumb_flexion, index_flexion, middle_flexion, ring_flexion = hand_action_6d
    
    # 확장 로직
    expanded_action = np.zeros(16, dtype=np.float32)
    expanded_action[0] = thumb_1
    expanded_action[1] = thumb_2
    expanded_action[2:4] = thumb_flexion
    expanded_action[4] = 0      
    expanded_action[5] = index_flexion * (15/11)     
    expanded_action[6:8] = middle_flexion * (5/11)
    expanded_action[8] = 0 
    expanded_action[9] = middle_flexion * (15/11)
    expanded_action[10:12] = middle_flexion * (5/11)
    expanded_action[12] = 0
    expanded_action[13] = ring_flexion * (15/11)
    expanded_action[14:16] = ring_flexion * (5/11)

    return expanded_action

def expand_hand_action_v3(hand_action_6d):
    """모델이 출력한 9차원 손 액션을 16차원으로 확장합니다."""
    if hand_action_6d.shape[0] != 9:
        return hand_action_6d # 이미 확장되었거나 다른 차원이면 그대로 반환

    thumb_1, thumb_2, thumb_3, thumb_4, index_1, index_flexion, middle_1, middle_flexion, ring_flexion = hand_action_6d
    
    # 확장 로직
    expanded_action = np.zeros(16, dtype=np.float32)
    expanded_action[0] = thumb_1
    expanded_action[1] = thumb_2
    expanded_action[2] = thumb_3
    expanded_action[3] = thumb_4
    expanded_action[4] = 0      
    expanded_action[5] = index_1 
    expanded_action[6:7] = index_flexion
    expanded_action[8] = 0 
    expanded_action[9] = middle_1
    expanded_action[10:12] = middle_flexion
    expanded_action[12] = 0
    expanded_action[13] = ring_flexion * (15/11)
    expanded_action[14:16] = ring_flexion * (5/11)

    return expanded_action
        
def action_to_array(action, i, hand_deploy_mode = "inspire_kistar_full"):
    waist      = action['action.waist'][0, i]      # shape (3,)
    left_arm   = action['action.left_arm'][0, i]   # shape (7,)
    right_arm  = action['action.right_arm'][0, i]  # shape (7,)
    
    head = np.zeros(2, dtype=np.float64) # head is fixed..

    action_np = np.concatenate([
            waist.astype(np.float64),
            head,
            left_arm.astype(np.float64),
            right_arm.astype(np.float64),
        ], axis=0)

    if hand_deploy_mode=="inspire_only":  

        left_hand = action['action.left_hand'][0, i] # shape (6,)
        right_hand = action['action.right_hand'][0, i] # shape (6,)

        hand_action_np = np.zeros(28, dtype=np.float64)
        hand_action_np[:6] = left_hand
        hand_action_np[6:12] = right_hand
    
    elif hand_deploy_mode=="kistar_only_full":  
        right_hand = action['action.kistar_hand'][0, i] # shape (16,)

        hand_action_np = np.zeros(28, dtype=np.float64)
        hand_action_np[12:] = right_hand
        
    elif hand_deploy_mode=="kistar_only_reduced":  
        right_hand = action['action.kistar_hand'][0, i] # shape (6,)

        right_hand_np = expand_hand_action(right_hand)

        left_hand_np = np.zeros(12, dtype=np.float64)
        
        hand_action_np=np.concatenate((left_hand_np,right_hand_np))

    elif hand_deploy_mode=="inspire_kistar_full":

        left_hand = action['action.inspire_hand'][0, i] # shape (6,)
        right_hand = action['action.kistar_hand'][0, i] # shape (16,)

        hand_action_np = np.zeros(28, dtype=np.float64)
        hand_action_np[:6] = left_hand
        hand_action_np[12:] = right_hand
        
    elif hand_deploy_mode=="inspire_kistar_reduced":

        left_hand = action['action.inspire_hand'][0, i] # shape (6,)
        right_hand = action['action.kistar_hand'][0, i] # shape (6,)

        right_hand_np = expand_hand_action(right_hand)
        
        left_hand_np = np.zeros(12, dtype=np.float64)
        left_hand_np[:6] = left_hand
        
        hand_action_np=np.concatenate((left_hand_np,right_hand_np))
        
    elif hand_deploy_mode=="inspire_kistar_reduced_v3":
        left_hand = action['action.inspire_hand'][0, i] # shape (6,)

        right_hand = action['action.kistar_hand'][0, i] # shape (9,)

        right_hand_np = expand_hand_action_v3(right_hand)
        
        left_hand_np = np.zeros(12, dtype=np.float64)
        left_hand_np[:6] = left_hand
        
        hand_action_np=np.concatenate((left_hand_np,right_hand_np))

    # ----- (NEW 0211): action mapping for PCA -----
    elif hand_deploy_mode=="inspire_kistar_pca":
        left_hand = action['action.left_hand'][0, i]   # (6,)
        s7 = action["action.right_hand_pca"][0, i]           # (7,)  synergy

        right_hand_np = expand_hand_action_pca(s7)         # (16,)

        hand_action_np = np.zeros(28, dtype=np.float64)
        hand_action_np[:6] = left_hand
        hand_action_np[12:] = right_hand_np
    # ----- (END 0211): action mapping for PCA -----

    elif hand_deploy_mode=="kistar_only":  
        right_hand = action['action.kistar_hand'][0, i] # shape (16,)

        hand_action_np = np.zeros(28, dtype=np.float64)
        hand_action_np[12:] = right_hand

    return action_np, hand_action_np

def temporal_ensemble_action(action_buffer, decay=0.3, window_size=5):
    """
    Temporal ensemble: 버퍼에 쌓인 미래 예측 액션들 중 최근 window_size개를 지수 감쇠 가중치로 앙상블하여 하나의 액션으로 합성하고,
    사용한 가장 오래된 항목을 버퍼에서 제거합니다.

    Args:
        action_buffer (list of tuple(np.ndarray, np.ndarray)): (action_np, hand_action_np) 쌍의 리스트
        decay (float): 지수 가중치 감쇠 계수
        window_size (int): 앙상블에 사용할 최대 버퍼 크기

    Returns:
        action_np (np.ndarray) : 앙상블된 로봇 액션 벡터
        hand_action_np (np.ndarray): 앙상블된 손 액션 벡터
    """
    if not action_buffer:
        return None, None

    # 앙상블에 사용할 실제 크기 계산
    N = min(len(action_buffer), window_size)

    # 버퍼의 앞쪽 N개 요소를 스택
    actions_stack = np.stack([action_buffer[i][0] for i in range(N)], axis=0)  # shape (N, D_action)
    hands_stack   = np.stack([action_buffer[i][1] for i in range(N)], axis=0)  # shape (N, D_hand)

    # 지수 감쇠 가중치 계산
    weights = np.exp(-decay * np.arange(N))
    weights = weights / weights.sum()

    # 가중합 연산
    action_np      = weights.dot(actions_stack)
    hand_action_np = weights.dot(hands_stack)

    # 사용한 가장 오래된 항목 제거
    action_buffer.pop(0)

    return action_np, hand_action_np

def moving_average_action(action_buffer, history):
    """
    이동 평균 필터: 버퍼에서 하나 꺼내어 history에 추가 후 평균을 계산
    """
    if not action_buffer:
        return None, None
    latest_a, latest_ha = action_buffer.pop(0)
    history.append((latest_a, latest_ha))
    a_stack = np.stack([a for a, _ in history], axis=0)
    ha_stack = np.stack([ha for _, ha in history], axis=0)
    return np.mean(a_stack, axis=0), np.mean(ha_stack, axis=0)

def upsample_actions(seq, slow_hz=20, fast_hz=50, k=5):
    """
    seq: [(D,), …] 형태의 액션(또는 손 액션) 리스트, 길이 N
    slow_hz: 원본 주파수 (20)
    fast_hz: 목표 주파수 (50)
    k: 스플라인 차수 (5)
    """
    arr = np.stack(seq, axis=0)       # (N, D)
    N, D = arr.shape
    t_slow = np.arange(N) / slow_hz   # 원본 타임스탬프
    duration = t_slow[-1] if N > 1 else 0
    M = int(duration * fast_hz) + 1
    t_fast = np.linspace(0, duration, M)
    spline = make_interp_spline(t_slow, arr, k=k, axis=0)
    return spline(t_fast)             # (M, D)

class Gr00t_Inference:
    """20 Hz와 50 Hz 루프를 각각 별도 스레드에서 돌리는 기본 구조."""

    # [변경] parent_dir 파라미터 추가: 디버그 이미지 및 플롯 저장 경로 지정용
    def __init__(self, shared_event, shm_name, shared_lock, policy, mode,
                action_method="base", hand_mode="full", decay=0.3, window_size=5, slow_hz: float = 20,
                fast_hz: float = 50, parent_dir=None):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.shared_event = shared_event
        # [변경] parent_dir 속성 추가: 디버그/플롯 파일 저장 경로 관리
        self.parent_dir = parent_dir
        
        self.policy_name = policy

        self.slow_hz = slow_hz
        self.fast_hz = fast_hz

        self.obs = None
        self.action = None

        self.action_np = np.zeros(19) # for body/ waist(3), head(2), left arm(7), right arm(7)
        self.qpos = np.zeros(19) # for body/ waist(3), head(2), left arm(7), right arm(7)
                
        # [추가] 플로팅을 위한 업샘플된 raw action 버퍼
        self.upsampled_raw_actions_for_plot = None

        self.shm_name = shm_name
        self.shared_lock = shared_lock
        
        self.hand_deploy_mode="inspire"
    
        self.hand_qpos = np.zeros(28) # for inspire hands (6x2) + kistar hand (16)
        self.hand_action_np = np.zeros(28) # for inspire hands (6x2) + kistar hand (16)
                                
        if mode in ["gr00t", "gr00t_zed"]:
            self.hand_deploy_mode="inspire_only"            
        
        elif mode=="gr00t_kistar":                        
            if hand_mode=="full":
                self.hand_deploy_mode="kistar_only_full"          
            elif hand_mode == "reduced":                
                self.hand_deploy_mode="kistar_only_reduced"       
            else:                
                self.hand_deploy_mode="kistar_only"     


        elif mode=="gr00t_kistar_inspire":            
            if hand_mode=="full":
                self.hand_deploy_mode="inspire_kistar_full"
            elif hand_mode=="reduced_v3":                
                self.hand_deploy_mode="inspire_kistar_reduced_v3"       
            # ----- (NEW 0211): add hand mode for PCA -----
            elif hand_mode=="reduced_pca":  
                self.hand_deploy_mode="inspire_kistar_pca"     
            # ----- (NEW 0211): add hand mode for PCA -----
            else:                
                self.hand_deploy_mode="inspire_kistar_reduced"            
        
        self.zed_mode = True
        self.binocular = True
        self.masking = False
        self.debug_plot = False 
        self.action_method = action_method
        self.decay = decay
        self.window_size = window_size

        self.liposolver = None

        self.infer_time_ms = None
        self.avg_infer_ms = None
        self.infer_times = deque(maxlen=10)
        self.avg_infer_ms = 0.0
        self.infer_time_count = 0

        self.infer_start_times = []  # 시작 시점 (sec)
        self.infer_end_times   = []  # 종료 시점 (sec)
        self._ref_time = None

        # 배열+인덱스 기반 버퍼
        self.primary_actions      = None   # np.ndarray shape (M, D_action)
        self.primary_hand_actions = None   # np.ndarray shape (M, D_hand)
        self.primary_index        = 8
        self.buffer_threshold     = 0
        # 이전 시퀀트 저장용
        self.prev_actions         = None
        self.prev_hand_actions    = None

        self.history = deque(maxlen=self.window_size)

        self.live = Live(refresh_per_second=self.slow_hz)
        
        # [추가] 디버그 이미지 저장 폴더 설정: OpenCV로 처리된 이미지를 파일로 저장
        if self.debug_plot:
            self.debug_dir = Path(parent_dir) / "debug_frames"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] Debug images will be saved to: {self.debug_dir}")

        # ----- (NEW 0210): wrap all all Gr00t-specific setup to only happen for Gr00t policy-----
        self.modality_config = None
        self.modality_transform = None
        self.data_config = None
        self.model_path = None
        self.embodiment_tag = None
        self.denoising_steps = None

        if self.policy_name in ("gr00t", "act"):
            DATA_CONFIG_MAP, _ = _lazy_import_gr00t()

            self.model_path = os.getenv("GR00T_MODEL_PATH", "/home/kist/.../vla_Jan05_v2")  # keep yours
            self.data_config = DATA_CONFIG_MAP["unitree_g1_inspire_kistar"]
            self.embodiment_tag = 'new_embodiment'

            self.modality_config = self.data_config.modality_config()
            self.modality_transform = self.data_config.transform()
            self.denoising_steps = 4

        elif self.policy_name == "maniflow":
            # Do nothing here — ManiFlowPolicyAdapter handles its own preproc.
            pass
        else:
            raise ValueError(f"Unknown policy: {self.policy_name}")

        # ----- (END 0210): wrap all all Gr00t-specific setup to only happen for Gr00t policy-----

        self.policy = None
        self.gr00t_task_name = ""

        self._init_shm()

        self._init_rich_live()


        self.start_loop = False

        #thread loop init

        self.slow_interval = int(1e9 / self.slow_hz)
        self.fast_interval = int(1e9 / self.fast_hz)

        self.obs_lock = threading.Lock()
        self.ctrl_lock = threading.Lock()
        self._stop_event = threading.Event()

        # 20Hz 추론 루프 
        self._slow_thread = threading.Thread(target=self._inference_loop, name=f"{self.slow_hz}HzLoop")
        self._slow_thread.daemon = True
        self._slow_thread.start()

        # 50Hz 제어 루프
        self._fast_thread = threading.Thread(target=self._ctrl_loop, name=f"{self.fast_hz}HzLoop")
        self._fast_thread.daemon = True
        self._fast_thread.start()

    # ----- (NEW 0210): compute replan threshold from measured inference latency -----
    def _compute_replan_threshold(self, buf_len: int) -> int:
        if buf_len <= 0:
            return 0

        # If we don't have a measurement yet, keep legacy behavior (half).
        if not self.infer_times:
            return max(0, int(buf_len // 2))

        infer_s = float(self.avg_infer_ms) / 1000.0
        slow_tick_s = 1.0 / float(self.slow_hz)

        margin_s = 0.03  # jitter margin
        need_s = infer_s + slow_tick_s + margin_s

        need_steps = int(math.ceil(need_s * self.fast_hz))
        min_steps = int(math.ceil(0.15 * self.fast_hz))  # keep at least 150ms

        remain_steps = max(need_steps, min_steps)
        thr = max(0, buf_len - remain_steps)
        return int(min(thr, buf_len - 1))
    # ----- (END 0210): compute replan threshold from measured inference latency -----

    # ----- (NEW 0209): make new initializer method for maniflow -----
    def _init_maniflow_policy(self):
        ckpt = os.environ.get("MANIFLOW_CKPT", "")
        if not ckpt:
            raise ValueError("Set MANIFLOW_CKPT=/path/to/latest.ckpt before running deploy_gr00t.py")

        use_ema = os.environ.get("MANIFLOW_USE_EMA", "1") == "1"
        compile_ok = os.environ.get("MANIFLOW_COMPILE", "1") == "1"
        steps = int(os.environ.get("MANIFLOW_STEPS", "2"))

        # This matches your assumption that this run is reduced_v3 => action dim 32
        self.policy = ManiFlowPolicyAdapter(
            ckpt_path=ckpt,
            device=self.device,
            use_ema=use_ema,
            num_inference_steps=steps,
            denoise_timesteps=steps,
            img_hw=(240, 320),
            compile_model=compile_ok,
            expected_layout="inspire_kistar_reduced_v3",
        )

        # Keep the buffering fields the same as GR00T
        self.primary_buffer = []
        self.secondary_buffer = []
        self.buffer_threshold = 8

        print("[DEPLOY] ManiFlow policy loaded successfully")
    # ----- (END 0209): make new initializer method for maniflow -----

    def _init_gr00t_policy(self):

        task_info = self.gr00t_task_shm.read_data()
        self.gr00t_task_name = task_info.get("task_name", "")
        self.policy = init_gr00t_policy(
            self.model_path, self.embodiment_tag, self.modality_config,
            self.modality_transform, self.denoising_steps, self.device
        )
        self.primary_buffer = []       # 이전 버퍼
        self.secondary_buffer = []     # 이후 버퍼
        self.buffer_threshold = 8      # 재생 전류 유지 임계치

        print("Policy initialized")

    def _init_shm(self):
        
        self.camera_shm = SharedMemoryManager(CAMERA, self.shared_lock["camera_lock"], self.shm_name["camera_shm"])
        self.robot_action_shm = SharedMemoryManager(ROBOT_ACTION, self.shared_lock["robot_action_lock"], self.shm_name["robot_action_shm"])
        self.robot_obs_shm = SharedMemoryManager(ROBOT_OBS, self.shared_lock["robot_obs_lock"], self.shm_name["robot_obs_shm"])

        self.hand_obs_shm =  SharedMemoryManager(KISTAR_HAND_RECEIVED, self.shared_lock["kistar_hand_received_lock"], self.shm_name["kistar_hand_received_shm"])
        self.kistar_hand_action_shm = SharedMemoryManager(KISTAR_HAND_ACTION, self.shared_lock["kistar_hand_action_lock"], self.shm_name["kistar_hand_action_shm"])

        self.gr00t_task_shm = SharedMemoryManager(GR00T_TASK_LAYOUT, self.shared_lock["gr00t_lock"],self.shm_name["gr00t_shm"])
        self.record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, self.shared_lock["record_lock"], self.shm_name["record_mode_shm"])
        self.workspace_mask_shm = SharedMemoryManager(WORKSPACE_MASK, self.shared_lock["workspace_mask_lock"], self.shm_name["workspace_mask_shm"])

    def _init_plot(self):
        self.groups = {"waist": slice(0,3),
                  "left_arm": slice(5,12),
                  "right_arm": slice(12,19)}
        fig, axes = plt.subplots(2, 3, figsize=(15,8), sharex=True)
        fig.suptitle("Test Mode: qpos & action_np Over Time", fontsize=16)
        lines_q, lines_a = {}, {}
        for ix, (name, sl) in enumerate(self.groups.items()):
            axq = axes[0,ix]; axq.set_title(f"qpos: {name}")
            lines_q[name] = [axq.plot([],[], label=f"dim{d}")[0] for d in range(sl.start, sl.stop)]
            axq.legend(fontsize="small"); axq.set_ylabel("qpos")
            axa = axes[1,ix]; axa.set_title(f"action: {name}")
            lines_a[name] = [axa.plot([],[], '--', label=f"dim{d}")[0] for d in range(sl.start, sl.stop)]
            axa.legend(fontsize="small"); axa.set_ylabel("action")
        for ax in axes[1]:
            ax.set_xlabel("Time step")
        self.qpos_buf, self.act_buf = [], []

        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 14

    def _init_rich_live(self):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=self.live.console,
        )

    def update_rich_live(self):

        with self.ctrl_lock:
            action_np = self.action_np
            hand_action_np = self.hand_action_np

        np.set_printoptions(precision=6, suppress=True, floatmode='fixed')

        table = Table(
            title=f"[bold white on blue] TASK: {self.gr00t_task_name} [/]", 
            title_justify="left",
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Metric", no_wrap=True)
        table.add_column("Value", overflow="fold")

        table.add_row("qpos", np.array2string(self.qpos, precision=3))
        table.add_row("hand_qpos", np.array2string(self.hand_qpos, precision=3))
        table.add_row("action", np.array2string(action_np, precision=3))
        table.add_row("hand_action", np.array2string(np.concatenate((hand_action_np[:12],np.degrees(hand_action_np[12:]))), precision=3))
        table.add_row("Inference Time (ms)",f"{self.infer_time_ms:.2f}ms  /  Avr {self.avg_infer_ms:.2f}ms")
        table.add_row("primary_index",f"{self.primary_index}")
        
        self.live.update(table)

    def _save_plots(self):
        """테스트 데이터 전체를 처리한 뒤 qpos·action 그래프를 파일로 저장한다."""
        if len(self.qpos_buf) == 0:      # 데이터가 없으면 스킵
            return
        
        arr_q = np.asarray(self.qpos_buf)     # (T, 19)
        arr_a = np.asarray(self.act_buf)      # (T, 19)
        times = np.arange(arr_q.shape[0]) / self.slow_hz  # 초 단위 시간 벡터

        prop_cycle = plt.rcParams['axes.prop_cycle']
        color_list = prop_cycle.by_key()['color']


        for name, sl in self.groups.items():
            fig, ax = plt.subplots(figsize=(24, 12), dpi=600)
            for d in range(sl.start, sl.stop):
                c = color_list[(d - sl.start) % len(color_list)]
                ax.plot(times, arr_q[:, d],           label=f"qpos {d}", color=c)
                ax.plot(times, arr_a[:, d], '--',     label=f"act {d}", color=c)

            for ts in self.infer_start_times:
                ax.axvline(x=ts, linestyle='--', linewidth=1, label='infer start')
            for ts in self.infer_end_times:
                ax.axvline(x=ts, linestyle=':',  linewidth=1, label='infer end')


            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize="small", ncol=2)


            ax.set_title(f"{name} (qpos & action)")
            ax.set_xlabel("Time step")
            fig.tight_layout()

            out_dir = Path(parent_dir) / "deploy_test_result"
            out_dir.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성

            out_path = out_dir / f"{self.policy_name}_{self.action_method}_qpos_action_{name}.png"

            fig.savefig(out_path)
            plt.close(fig)
            print(f"[Plot] Saved {name} plot → {out_path}")

        # 버퍼 초기화(재시작 대비)
        self.qpos_buf.clear()
        self.act_buf.clear()
        self.infer_start_times.clear()
        self.infer_end_times.clear()

    def write_shm(self):
        self.robot_action_shm.write_data(
            action_waist=self.action_np[:3],
            action_head = self.action_np[3:5],
            action_arm=self.action_np[5:19],      
            action_hand=self.hand_action_np[:12]             
        )

        self.kistar_hand_action_shm.write_data(
            hand_action = self.hand_action_np[12:]
        )

    
    self.robot_action_shm.write_data(
        action_waist=action_waist,
        action_head=action_head,
        action_arm=action_arm,
        action_hand=self.hand_action_np[:12],
    )

    self.kistar_hand_action_shm.write_data(
        hand_action=self.hand_action_np[12:]
    )


    def get_real_obs(self):
        img_dict   = self.camera_shm.read_data()
        robot_obs = self.robot_obs_shm.read_data()
        
        hand_obs = self.hand_obs_shm.read_data()

        obs_waist = robot_obs["obs_waist"]
        obs_head = robot_obs["obs_head"]
        obs_arm = robot_obs["obs_arm"]
        
        obs_hand_inspire = robot_obs["obs_hand"]
        obs_hand_kistar = hand_obs["hand_q_pos"]

        qpos = np.concatenate((obs_waist, obs_head, obs_arm))
        
        obs_hand = np.concatenate((obs_hand_inspire,obs_hand_kistar))

        self.qpos = qpos                                            # e.g. (N,)
        self.hand_qpos = obs_hand      

        task_info = self.gr00t_task_shm.read_data()
        self.gr00t_task_name = task_info.get("task_name", "")
        
        # [추가] ZED 카메라 모드 지원: RealSense 대신 ZED 카메라 사용 시 처리 분기
        if self.zed_mode:
                
            if self.binocular:
                # [추가] ZED 양안 카메라 모드: 왼쪽과 오른쪽 카메라 모두 사용
                frame_left = img_dict.get("camera_left")
                frame_right = img_dict.get("camera_right")

                # [추가] 마스킹 기능: 작업 공간 외 영역을 검은색으로 마스킹하여 제거
                if self.masking:
                    mask_data = self.workspace_mask_shm.read_data()
                    mask_left_flat = mask_data.get("mask_left_flat")
                    mask_right_flat = mask_data.get("mask_right_flat")

                    if frame_left is not None and mask_left_flat is not None and mask_left_flat.size > 0:
                        try:
                            h, w = frame_left.shape[:2]
                            mask_left = mask_left_flat.reshape(h, w).astype(np.uint8)
                            frame_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)
                        except Exception as e:
                            print(f"[MASKING] 왼쪽 마스크 적용 실패: {e}")

                    if frame_right is not None and mask_right_flat is not None and mask_right_flat.size > 0:
                        try:
                            h, w = frame_right.shape[:2]
                            mask_right = mask_right_flat.reshape(h, w).astype(np.uint8)
                            frame_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
                        except Exception as e:
                            print(f"[MASKING] 오른쪽 마스크 적용 실패: {e}")
                
                self.rgb_left = process_frame(frame_left, side="zed_left")
                self.rgb_right = process_frame(frame_right, side="zed_right")

                if self.rgb_left is None or self.rgb_right is None:
                    print(f"[OBS] ZED 양안 프레임 오류/없음. 추론 건너뜀.")
                    self.obs = None
                    return

                obs = obs_dict_zed(self.gr00t_task_name,
                                   self.qpos, self.hand_qpos,
                                   self.rgb_left, self.rgb_right,
                                   self.hand_deploy_mode,
                                   self.binocular
        
                                   )

            else:
                # [추가] ZED 단안 카메라 모드: 왼쪽 카메라만 사용
                frame_left = img_dict.get("camera_left")
                frame_right = img_dict.get("camera_right")

                # [추가] 마스킹 기능: 작업 공간 외 영역을 검은색으로 마스킹하여 제거
                if self.masking:
                    mask_data = self.workspace_mask_shm.read_data()
                    mask_left_flat = mask_data.get("mask_left_flat")
                    mask_right_flat = mask_data.get("mask_right_flat")

                    if frame_left is not None and mask_left_flat is not None and mask_left_flat.size > 0:
                        try:
                            h, w = frame_left.shape[:2]
                            mask_left = mask_left_flat.reshape(h, w).astype(np.uint8)
                            frame_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)
                        except Exception as e:
                            print(f"[MASKING] 왼쪽 마스크 적용 실패: {e}")

                    if frame_right is not None and mask_right_flat is not None and mask_right_flat.size > 0:
                        try:
                            h, w = frame_right.shape[:2]
                            mask_right = mask_right_flat.reshape(h, w).astype(np.uint8)
                            frame_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
                        except Exception as e:
                            print(f"[MASKING] 오른쪽 마스크 적용 실패: {e}")
                
                self.rgb_left = process_frame(frame_left, side="zed_left")
                self.rgb_right = process_frame(frame_right, side="zed_right")
  
                # [추가] 차원 축소 모드에 따른 분기: reduced_modeၼ 때 obs_dict_zed_mono_reduced 사용 (16차원 → 5차원)
                obs = obs_dict_zed(self.gr00t_task_name,
                                   self.qpos, self.hand_qpos,
                                   self.rgb_left,
                                   self.rgb_right,
                                   self.hand_deploy_mode,
                                   self.binocular
        
                                   )
        else:
            # 기존 RealSense 모드
            frame = img_dict.get("realsense")
            self.rgb = process_frame(frame, side="realsense")   # (H, W, 3) uint8
            obs = obs_dict_realsense(self.gr00t_task_name, self.qpos,
                                     self.hand_qpos, self.rgb,
                                     self.hand_deploy_mode)

        with self.obs_lock:
            self.obs = obs


    def base_action_buffer(self,new_action_chunk):
        """
        기본 방법: chunk 내 각 스텝을 배열로 변환해 순차적으로 버퍼에 추가
        """
        buffer = []

        for step_idx in range(self.chunk_size):

            a_np, ha_np = action_to_array(new_action_chunk, step_idx, self.hand_deploy_mode)

            buffer.append((a_np, ha_np))

        return buffer


    def gr00t_inference(self):

        now = time.perf_counter()
        if self._ref_time is None:
            self._ref_time = now

        with self.obs_lock:
            obs = self.obs  
        
        # ----- (NEW 0209): Call policy.update_obs(obs) every 20Hz -----
        if obs is None:
            return

        # NEW: keep ManiFlow obs history updated at 20Hz
        if hasattr(self.policy, "update_obs"):
            try:
                self.policy.update_obs(obs)
            except Exception as e:
                print(f"[DEPLOY][WARN] policy.update_obs failed: {e}")
        # ----- (END 0209): Call policy.update_obs(obs) every 20Hz -----
                                  # 버퍼의 절반 정도가 소진되면 미래 예측을 위해 새 추론 실행 
        if  self.primary_index >= self.buffer_threshold:
            # 인퍼런스 전 상태 저장
               # temporal ensemble 적용을 위해 미래 액션 시퀀스를 담고 있음 
            if self.primary_actions is not None:
                # 현재 버퍼에 남은 액션 개수 계산 
                remaining = len(self.primary_actions) - self.primary_index
                if remaining > 0:
                    self.prev_actions      = self.primary_actions[self.primary_index:]
                    self.prev_hand_actions = self.primary_hand_actions[self.primary_index:]
                else:
                    self.prev_actions = None

            # 2) 새로운 액션 시퀀스 생성 → 리스트 acts, hands
            self.start_infer = time.perf_counter()

            action_sequence = self.policy.get_action(obs)

            self.end_infer = time.perf_counter()
            self.infer_time_ms = (self.end_infer - self.start_infer) * 1000        
            
            self.infer_times.append(self.infer_time_ms)
            self.avg_infer_ms = sum(self.infer_times) / len(self.infer_times)
            self.chunk_size= action_sequence['action.waist'].shape[1]

            new_buffer = [(action_to_array(action_sequence, i, self.hand_deploy_mode)) for i in range(action_sequence['action.waist'].shape[1])]
            
            acts, hands = zip(*new_buffer)

            # 3) upsampling 20hz -> 50hz
            up_acts  = upsample_actions(acts,  slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=5) #38
            up_hands = upsample_actions(hands, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=5) #38

            # 4) 블렌딩: prev_*이 있으면 겹치는 부분만큼 크로스페이드
            if self.prev_actions is not None:
                               #18 스텝 
                ovl = min(len(self.prev_actions), len(up_acts))
                # 0→1 선형 알파
                alphas = np.linspace(0, 1, ovl)[:, None]
                
                blended_actions = self.prev_actions[:ovl] * (1 - alphas) + up_acts[:ovl] * alphas
                blended_hands   = self.prev_hand_actions[:ovl] * (1 - alphas) + up_hands[:ovl] * alphas
                # 합치기
                combined_actions      = np.concatenate([blended_actions, up_acts[ovl:]], axis=0)
                combined_hand_actions = np.concatenate([blended_hands, up_hands[ovl:]], axis=0)
            else:
                combined_actions      = up_acts
                combined_hand_actions = up_hands

            # 5) 새 primary 배열로 덮어쓰기 & 인덱스/threshold 초기화
            self.primary_actions      = combined_actions
            self.primary_hand_actions = combined_hand_actions
            self.primary_index        = 0
            self.buffer_threshold     = len(self.primary_actions) / 2

            # secondary 버퍼 초기화
            self.secondary_buffer.clear()


        # 타임스탬프 기록
        self.infer_start_times.append(self.start_infer - self._ref_time)
        self.infer_end_times.append(self.end_infer   - self._ref_time)

    def get_action(self):
        
            remain = len(self.primary_actions) - self.primary_index
            if remain <= 0:
                return  # 버퍼 비었으면 리턴
            
            # 버퍼에서 순차적으로 꺼내기
            if self.action_method == "base":
                action_np      = self.primary_actions[self.primary_index]
                hand_action_np = self.primary_hand_actions[self.primary_index]
                
                self.primary_index += 1

            # 이동 평균 필터링 
            elif self.action_method == "maf":
                # 이동 평균: 이번 인덱스 액션을 history에 추가 후 평균
                next_a  = self.primary_actions[self.primary_index]
                next_ha = self.primary_hand_actions[self.primary_index]
                self.primary_index += 1

                self.history.append((next_a, next_ha))
                # history 길이만큼 평균 계산
                a_stack  = np.stack([a for a, _  in self.history],  axis=0)
                ha_stack = np.stack([ha for _, ha in self.history],  axis=0)
                action_np      = a_stack.mean(axis=0)
                hand_action_np = ha_stack.mean(axis=0)

            # 지수 평균 앙상블 (가까운 미래에 더 높은 가중치 부여)
            elif self.action_method == "tem":
                # 지수 앙상블: 앞으로 window_size개 묶어서 가중합 후 하나씩 슬라이딩
                N = min(remain, self.window_size)
                window_a  = self.primary_actions[self.primary_index : self.primary_index + N]
                window_ha = self.primary_hand_actions[self.primary_index : self.primary_index + N]

                weights = np.exp(-self.decay * np.arange(N))
                weights = weights / weights.sum()

                action_np      = weights.dot(window_a)
                hand_action_np = weights.dot(window_ha)
                # 첫 요소를 소비하듯 인덱스만 증가
                self.primary_index += 1

            with self.ctrl_lock:
                self.action_np = action_np
                self.hand_action_np = hand_action_np


    def get_ui_mode(self):
        record_mode_data = self.record_mode_shm.read_data()
        self.deploy_mode = record_mode_data["deploy"]


    def _inference_loop(self) -> None:
        next_ts = time.perf_counter_ns()
        with Live(refresh_per_second=self.slow_hz) as self.live:

            while not (self._stop_event.is_set() or self.shared_event['shutdown'].is_set()):
                self.get_ui_mode()
                if self.deploy_mode:

                    if self.policy is None:
                        if self.policy_name == "gr00t":
                            self._init_gr00t_policy()
                        elif self.policy_name == "act":
                            self._init_act_policy()      
                        # ----- (NEW 0209): add maniflow initialization branch -----
                        elif self.policy_name == "maniflow":
                            self._init_maniflow_policy()
                        # ----- (END 0209): add maniflow initialization branch -----
                    else:
                        self.do_slow()  # 20 Hz 작업

                next_ts +=  self.slow_interval
                sleep_ns = next_ts - time.perf_counter_ns()
                if sleep_ns > 0:
                    if sleep_ns > 1_000_000:
                        time.sleep((sleep_ns - 500_000) / 1e9)
                    # 개선 코드 (busy-wait 최소화)
                    while True:
                        remaining = (next_ts - time.perf_counter_ns()) / 1e9
                        if remaining <= 0:
                            break
                        time.sleep(min(remaining, 0.001))  # 1ms 단위 sleep
                else:
                    next_ts = time.perf_counter_ns()

    def _ctrl_loop(self) -> None:
        next_ts = time.perf_counter_ns()
        while not (self._stop_event.is_set() or self.shared_event['shutdown'].is_set()):

            self.do_fast()  # 50 Hz 작업

            next_ts +=  self.fast_interval
            sleep_ns = next_ts - time.perf_counter_ns()
            if sleep_ns > 0:
                if sleep_ns > 1_000_000:
                    time.sleep((sleep_ns - 500_000) / 1e9)
                # 개선 코드 (busy-wait 최소화)
                while True:
                    remaining = (next_ts - time.perf_counter_ns()) / 1e9
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.001))  # 1ms 단위 sleep
            else:
                next_ts = time.perf_counter_ns()


    # ────────────── 실제 작업 (오버라이드/콜백용) ──────────────
    def do_slow(self) -> None:
        """20 Hz마다 실행되는 작업 예시."""
        # if self.policy_name == "gr00t":
        if self.policy_name in ["gr00t", "maniflow"]: # (NEW 0209): include maniflow in slow/fast loops
            self.get_real_obs()
            
            self.gr00t_inference()

            self.start_loop = True

            self.update_rich_live()
            
            self.write_shm()

    def do_fast(self) -> None:
        """50 Hz마다 실행되는 작업 예시."""
        # if self.policy_name == "gr00t":
        if self.policy_name in ["gr00t", "maniflow"]: # (NEW 0209): include maniflow in slow/fast loops
            if self.start_loop : 
                self.get_action()
                self.write_shm()


    def stop(self) -> None:
        """루프를 안전하게 종료하고 스레드를 join."""
        self._stop_event.set()
        self._slow_thread.join()
        self._fast_thread.join()


    def _cleanup(self) -> None:
        """자원 정리"""

        # [추가] 디버그 플롯 정리: 원본에는 없었던 처리
        if self.debug_plot:
            # cv2.destroyAllWindows() # 파일 저장 방식에서는 필요 없음
            pass

        self.camera_shm.worker_close()
        self.robot_obs_shm.worker_close()
        self.robot_action_shm.worker_close()
        self.gr00t_task_shm.worker_close()
        self.record_mode_shm.worker_close()
