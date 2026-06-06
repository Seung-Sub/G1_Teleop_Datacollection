"""worker_deploy_dp: external policy (Diffusion Policy) inference adapter for G1.

GR00T 용 worker_deploy_policy 의 검증된 골격(SHM 입출력, DualRate loop, upsample,
action_method, cross-fade, lag compensation, write_shm)을 재사용하되, 모델/obs
부분만 Diffusion Policy(real-stanford/diffusion_policy) 용으로 교체.

GR00T 대비 DP 차이 (실제 DP 코드 정독 확정):
  - 모델: torch.load(ckpt, dill) → workspace → ema_model (Gr00tPolicy 대신).
  - obs: 평탄 dict {camera_0/1/2: (B,To,C,H,W) float32 [0,1], state: (B,To,28) float32}.
    이미지 (H,W,C) uint8 → (C,H,W) float32 /255. (GR00T 는 (H,W,C) uint8 중첩 dict.)
  - n_obs_steps=2 → 과거 2 스텝 obs 누적 필요 (deque 히스토리). GR00T 는 To=1.
  - language 없음 (DP 단일 task). GR00T 는 language_key 필요.
  - 추론: policy.predict_action(obs_dict)["action"] → (1, n_action_steps, 28).
  - slow_hz=10 (DP 학습 60→10 다운샘플). GR00T 는 20. fast_hz=60 동일 (arm 제어).
  - action 28D = left_arm7+right_arm7+left_hand7+right_hand7 (waist/head 없음).

카메라 role / state 레이아웃 / robot_action_shm 출력은 GR00T 와 동일 (하드웨어 동일).
main.py teleop env 에서 torch 없어도 import 안전 (lazy import). 실 추론은 별도
conda env (umi/dp) 의 evaluate_dp.py 가 SHM 에 attach 해 실행.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional, Dict, List

import numpy as np
import cv2

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    CAMERA, CAMERA_VIEW, GR00T_TASK_LAYOUT, RECORD_MODE_LAYOUT,
    ROBOT_ACTION, ROBOT_OBS, WORKSPACE_MASK, TELEOP_CONFIG, HAND_MAPPING_INV,
    WAIST_MAPPING_INV, HEAD_MAPPING_INV,
)
from utils.modality_layout import build_state_layout, layout_from_modality_json

import logging_mp
# logging_mp 호환 shim: 일부 PyPI 빌드(0.2.1)는 snake_case 별칭 get_logger/basic_config
# 가 없고 getLogger/basicConfig 만 제공한다. G1 코드는 get_logger 를 쓰므로, 설치된
# 빌드에 별칭이 없으면 여기서 보강해 배포 import 가 깨지지 않게 한다.
if not hasattr(logging_mp, "get_logger") and hasattr(logging_mp, "getLogger"):
    logging_mp.get_logger = logging_mp.getLogger
if not hasattr(logging_mp, "basic_config") and hasattr(logging_mp, "basicConfig"):
    logging_mp.basic_config = logging_mp.basicConfig
logger_mp = logging_mp.get_logger(__name__)


# =============================================================================
# Frame / observation helpers
# =============================================================================

def process_frame(frame: np.ndarray, side: str = "") -> Optional[np.ndarray]:
    """Validate + BGR → RGB + uint8. Returns None if the frame is unusable.

    Channel-order 정합 (Part 2 P2-8): 수집 측 video_sink 는 RealSense BGR8 / ZED
    BGRA→BGR 그대로 mp4 저장 (imageio.FFMPEG 가 mp4 saving 시 자동 RGB 변환은
    아니지만, process_frames helper 가 BGR2RGB 적용). 즉 mp4 = RGB. 본 deploy 측
    process_frame 도 SHM(BGR) → RGB. 결과적으로 학습 입력과 deploy 입력의 채널
    순서가 일치한다.
    """
    if not (isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3):
        logger_mp.warning(f"[Deploy] bad frame ({side}): shape={getattr(frame,'shape',None)}")
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def upsample_actions(seq: np.ndarray, slow_hz: float = 10.0, fast_hz: float = 60.0,
                     k: int = 5) -> np.ndarray:
    """Spline-upsample a (T, D) action chunk from slow_hz to fast_hz.

    DP: slow_hz=10 (학습 데이터 60→10 다운샘플 타임스텝), fast_hz=60 (arm 제어 루프).
        실제로는 인스턴스의 self.slow_hz/self.fast_hz 사용 — 기본값은 참고용.
    hand 처럼 이산/포화 신호는 k=1 (linear), arm 연속 신호는 k=5 (quintic).
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

def _parts_from_obs(qpos: np.ndarray, hand_qpos: np.ndarray, hand_type: str) -> dict:
    """qpos (19D = waist3+head2+arm14) + hand_qpos (14D) 를 layout 이름별 dict 로.

    Phase M (PART3 §2.5): build_state_layout 이 요구하는 name → array 매핑을 만들고,
    layout 의 토글 (waist_on/head_on) 에 따라 필요한 키만 obs_dict 에 포함된다.
    """
    hd_side = 7 if hand_type == 'dex3' else 6
    return {
        'waist':      qpos[:3].astype(np.float32),
        'head':       qpos[3:5].astype(np.float32),
        'left_arm':   qpos[5:12].astype(np.float32),
        'right_arm':  qpos[12:19].astype(np.float32),
        'left_hand':  hand_qpos[:hd_side].astype(np.float32),
        'right_hand': hand_qpos[hd_side:2 * hd_side].astype(np.float32),
    }


def build_obs_frame(qpos: np.ndarray, hand_qpos: np.ndarray,
                    frames: Dict[str, np.ndarray], hand_type: str,
                    camera_key_map: Dict[str, str]) -> dict:
    """DP 용 *단일 프레임* obs 추출 (히스토리 누적은 호출측에서).

    DP policy.predict_action 입력 (base_image_policy.py): {key: (B,To,*)}.
    여기서는 1 프레임만 만들고, get_real_obs 가 deque 로 To 개 쌓아 (1,To,*) 구성.

    반환 (단일 프레임, batch/time 차원 없음):
        {
          "state":    (28,) float32,                 # left_arm7+right_arm7+left_hand7+right_hand7
          "camera_0": (C,H,W) float32 [0,1],          # ego
          "camera_1": (C,H,W) float32 [0,1],          # wrist_l
          "camera_2": (C,H,W) float32 [0,1],          # wrist_r
        }
    DP 학습 dataset(__getitem__)과 동일: 이미지 (H,W,C)uint8 → (C,H,W)float32 /255.
    state 28D 순서 = convert_to_dp 의 state 레이아웃과 정합.

    camera_key_map: {role: "camera_N"} (예 {"ego":"camera_0","wrist_l":"camera_1",...}).
                    shape_meta(g1_dex3_image.yaml)의 obs 키와 일치해야 함.
    """
    parts = _parts_from_obs(qpos, hand_qpos, hand_type)
    # DP state 28D = arm+hand (waist/head 제외 — DP 학습 데이터 28D 구성).
    state = np.concatenate([
        parts["left_arm"], parts["right_arm"],
        parts["left_hand"], parts["right_hand"],
    ], axis=0).astype(np.float32)   # (28,)

    out: dict = {"state": state}
    for role, rgb in frames.items():
        cam_key = camera_key_map.get(role)
        if cam_key is None:
            continue
        # (H,W,C) uint8 → (C,H,W) float32 [0,1] (학습 dataset 과 동일)
        chw = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0
        out[cam_key] = chw
    return out


def action_to_array(action: dict, i: int, hand_type: str, layout) -> tuple:
    """[GR00T 잔재 — DP 에서는 미사용] N1.7 GR00T action chunk 분해용.
    DP 는 action_chunk_to_array (아래) 를 사용. import 호환 위해 정의만 유지.
    """
    layout_names = {n for n, _ in layout}

    def _opt(key: str, dim: int) -> np.ndarray:
        if key in action:
            return np.asarray(action[key][0, i], dtype=np.float32)
        return np.zeros(dim, dtype=np.float32)

    waist = _opt("waist", 3) if 'waist' in layout_names else np.zeros(3, dtype=np.float32)
    head  = _opt("head",  2) if 'head'  in layout_names else np.zeros(2, dtype=np.float32)
    left_arm  = _opt("left_arm",  7)
    right_arm = _opt("right_arm", 7)
    hd_side = 7 if hand_type == 'dex3' else 6
    left_h  = _opt("left_hand",  hd_side)
    right_h = _opt("right_hand", hd_side)

    action_np   = np.concatenate([waist, head, left_arm, right_arm], axis=0)    # 19 (3+2+7+7)
    hand_action = np.concatenate([left_h, right_h], axis=0)                     # 12 or 14
    return action_np, hand_action


def dp_action_chunk_to_arrays(action_chunk: np.ndarray, hand_type: str) -> tuple:
    """DP predict_action 출력 (n_action_steps, 14+2*hd) → (full[T,19], hand[T,14|12]).

    DP action 레이아웃 (convert_to_dp = parquet action 컬럼 그대로 = record layout):
        [0:7]=left_arm, [7:14]=right_arm, [14:14+hd]=left_hand, [14+hd:14+2*hd]=right_hand.
        => DEX3(hd=7): 28D (좌손 14:21, 우손 21:28). Inspire(hd=6): 26D (좌손 14:20, 우손 20:26).
    write_shm 의 fixed-shape (waist3+head2+arm14, hand14) 에 맞춰:
        full[T,19] = [waist3=0, head2=0, left_arm7, right_arm7]
        hand[T,14|12] = [left_hand, right_hand]  (DEX3=14, inspire=12)
    waist/head 는 DP 가 제어 안 함 (학습 = arm+hand) → zero.
    """
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim == 3:          # (B,T,D) → (T,D)
        chunk = chunk[0]
    T = chunk.shape[0]
    hd = 7 if hand_type == 'dex3' else 6
    left_arm   = chunk[:, 0:7]
    right_arm  = chunk[:, 7:14]
    left_hand  = chunk[:, 14:14 + hd]
    right_hand = chunk[:, 14 + hd:14 + 2 * hd]   # 손은 6+6/7+7 연속 (record layout 그대로).
    waist = np.zeros((T, 3), dtype=np.float32)
    head  = np.zeros((T, 2), dtype=np.float32)
    full = np.concatenate([waist, head, left_arm, right_arm], axis=1)   # (T,19)
    hand = np.concatenate([left_hand, right_hand], axis=1)              # (T, 2*hd)
    return full, hand


# =============================================================================
# Lazy Diffusion Policy loader
# =============================================================================

def init_dp_policy(ckpt_path: str, device: str = "cuda"):
    """Load a Diffusion Policy from a .ckpt. Import is local so module loads in any env.

    체크포인트 구조 (base_workspace.py save_checkpoint): payload = {cfg, state_dicts, pickles}.
    로딩 (create_from_checkpoint / load_payload 패턴):
      1. torch.load(ckpt, pickle_module=dill) → payload
      2. payload['cfg'] 로 workspace 클래스 인스턴스화 (hydra.utils.get_class)
      3. workspace.load_payload(payload) → state_dicts 로딩
      4. ema_model(use_ema) 또는 model 이 추론 policy
      5. policy.eval().to(device), reset()
    반환: (policy, cfg) — cfg 에서 n_obs_steps/n_action_steps 등 추론 파라미터 추출.
    """
    import dill
    import torch
    import hydra

    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)        # workspace 클래스
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # EMA 가중치 우선 (학습 시 use_ema=True 면 ema_model 이 보통 더 좋음).
    policy = workspace.model
    use_ema = bool(getattr(cfg.training, "use_ema", False))
    if use_ema and hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        policy = workspace.ema_model

    device_t = torch.device(device)
    policy.eval().to(device_t)
    if hasattr(policy, "reset"):
        policy.reset()
    logger_mp.info(f"[DeployDP] policy loaded. use_ema={use_ema} "
                   f"n_obs_steps={cfg.n_obs_steps} n_action_steps={cfg.n_action_steps} "
                   f"horizon={cfg.horizon} device={device}")
    return policy, cfg


# =============================================================================
# Inference worker
# =============================================================================

class DP_Inference:
    """Diffusion Policy inference worker. GR00T worker 골격 재사용, 모델/obs 만 DP 로.

    - slow loop (default 10Hz, DP 학습 다운샘플): obs 히스토리(deque, To=n_obs_steps) →
      policy.predict_action → (n_action_steps,28) → upsample(10→fast) → cross-fade →
      primary_actions 저장.
    - fast loop (default 60Hz, arm 제어): pop 1 step, action_method, write ROBOT_ACTION.
    - policy 는 record_mode_shm.deploy=True 인 첫 slow tick 에 lazy 로딩.

    - slow loop (default 20Hz): read obs, call policy, upsample to fast_hz,
      blend cross-fade with the tail of the previous chunk, store as
      `self.primary_actions / primary_hand_actions`.
    - fast loop (default 50Hz): pop one timestep, apply --action_method,
      write to ROBOT_ACTION.
    - policy is loaded lazily on the first slow tick where
      record_mode_shm.deploy is True.
    """

    def __init__(
        self,
        shm_name: dict,
        shared_lock: dict,
        shared_event: dict,
        mode: str = "gr00t_rs_multi",           # {"gr00t_rs_multi", "gr00t_zed"}
        model_path: str = "",                   # DP .ckpt 경로
        action_method: str = "tem",             # {"base", "maf", "tem"}
        decay: float = 0.3,
        window_size: int = 5,
        slow_hz: float = 10.0,                  # DP 학습 60→10 다운샘플 타임스텝
        fast_hz: float = 60.0,                  # arm 제어 루프(worker_g1_ctrl ACT_HZ=60)와 일치
        device: str = "cuda",
        binocular: bool = True,
        masking: bool = False,
        lag_compensate: bool = True,
        lag_log_every: int = 50,
        obs_ts_policy: str = 'min',
        modality_json_path: Optional[str] = None,
        camera_key_map: Optional[Dict[str, str]] = None,  # {role: camera_N} (shape_meta 키와 일치)
    ):
        assert mode in ("gr00t_rs_multi", "gr00t_zed", "gr00t"), f"unsupported mode: {mode}"
        if mode == "gr00t":
            mode = "gr00t_rs_multi"
        assert action_method in ("base", "maf", "tem"), f"unknown method: {action_method}"
        assert obs_ts_policy in ("min", "max"), f"unknown obs_ts_policy: {obs_ts_policy}"

        self.shared_event = shared_event
        self.mode         = mode
        self.zed_mode     = (mode == "gr00t_zed")
        self.binocular    = bool(binocular and self.zed_mode)
        self.masking      = bool(masking)
        self.model_path   = model_path          # DP .ckpt
        self.device          = device
        self.action_method   = action_method
        self.decay           = float(decay)
        self.window_size     = int(window_size)
        self.slow_hz         = float(slow_hz)
        self.fast_hz         = float(fast_hz)
        self.lag_compensate  = bool(lag_compensate)
        self.lag_log_every   = max(1, int(lag_log_every))
        self.obs_ts_policy   = obs_ts_policy
        self.modality_json_path = modality_json_path
        # role → camera_N (shape_meta g1_dex3_image.yaml 의 obs 키와 일치).
        self.camera_key_map = camera_key_map or {
            "ego": "camera_0", "wrist_l": "camera_1", "wrist_r": "camera_2",
            # ZED fallback
            "ego_left": "camera_0", "ego_right": "camera_1",
        }
        self._lag_chunk_count = 0
        self._lag_ns_acc      = 0
        self._lag_ns_max      = 0
        # DP 추론 파라미터 (정책 로딩 시 cfg 에서 갱신).
        self.n_obs_steps     = 2
        self.n_action_steps  = 8
        self.cfg             = None
        # obs 히스토리 (To 프레임 누적). 키별 deque.
        self._obs_hist       = None   # 로딩 후 deque(maxlen=n_obs_steps) 초기화

        # SHM handles + hand_type / active camera roles 결정 + modality layout 로드
        self._init_shm(shm_name, shared_lock)
        self._resolve_runtime_config()

        # State shared between slow / fast loops.
        self._obs_lock     = threading.Lock()
        self._ctrl_lock    = threading.Lock()
        self.policy        = None
        self.deploy_mode   = False
        self.start_loop    = False
        self.qpos          = np.zeros(19, dtype=np.float32)
        self.hand_qpos     = np.zeros(14, dtype=np.float32)
        self.task_name     = ""
        self.primary_actions      = None
        self.primary_hand_actions = None
        self.primary_index        = 0
        self._prev_actions        = None
        self._prev_hand_actions   = None
        self._action_buf          = deque(maxlen=self.window_size)
        self._hand_buf            = deque(maxlen=self.window_size)

        # Lifecycle
        self._stop_event = threading.Event()
        self._slow_thread = threading.Thread(target=self._inference_loop, daemon=True, name="DEPLOY_SLOW")
        self._fast_thread = threading.Thread(target=self._ctrl_loop,      daemon=True, name="DEPLOY_FAST")
        self._slow_thread.start()
        self._fast_thread.start()
        logger_mp.info(
            f"[Deploy] DP_Inference started (mode={mode}, hand={self.hand_type}, "
            f"camera_roles={self.camera_roles}, action_method={action_method}, "
            f"slow={slow_hz}Hz, fast={fast_hz}Hz, obs_ts_policy={obs_ts_policy})"
        )

    # ----- init helpers ------------------------------------------------------
    def _init_shm(self, shm_name, shared_lock):
        self.robot_obs_shm     = SharedMemoryManager(ROBOT_OBS,          shared_lock["robot_obs_lock"],      shm_name["robot_obs_shm"])
        self.robot_action_shm  = SharedMemoryManager(ROBOT_ACTION,       shared_lock["robot_action_lock"],   shm_name["robot_action_shm"])
        self.gr00t_task_shm    = SharedMemoryManager(GR00T_TASK_LAYOUT,  shared_lock["gr00t_lock"],          shm_name["gr00t_shm"])
        self.record_mode_shm   = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"],         shm_name["record_mode_shm"])
        self.teleop_config_shm = SharedMemoryManager(TELEOP_CONFIG,      shared_lock["record_lock"],         shm_name["teleop_config_shm"])
        if self.masking:
            self.workspace_mask_shm = SharedMemoryManager(WORKSPACE_MASK, shared_lock["workspace_mask_lock"], shm_name["workspace_mask_shm"])
        else:
            self.workspace_mask_shm = None

        # Phase L1: 카메라 SHM 들 attach (ZED 경로 / RealSense 멀티뷰 경로 분기).
        # 운영 옵션상 ZED 와 RealSense 가 동시 활성되는 케이스는 없음 (main.py 가 한쪽만 spawn).
        self.camera_shms: Dict[str, SharedMemoryManager] = {}
        self.legacy_camera_shm: Optional[SharedMemoryManager] = None
        if self.zed_mode:
            # ZED stereo 는 legacy CAMERA SHM 사용 (현재 main.py 의 worker_zed 는 CAMERA_VIEW 로
            # 마이그레이션됐지만, 이름은 그대로 'camera_shm' 사용 — backward-compat).
            if 'camera_shm' in shm_name:
                self.legacy_camera_shm = SharedMemoryManager(
                    CAMERA_VIEW, shared_lock["camera_lock"], shm_name["camera_shm"]
                )
        else:
            for role, shm_key in [
                ('ego',     'rs_ego_shm'),
                ('wrist_l', 'rs_wrist_l_shm'),
                ('wrist_r', 'rs_wrist_r_shm'),
            ]:
                if shm_key in shm_name:
                    lock_key = {'ego': 'rs_ego_lock',
                                'wrist_l': 'rs_wrist_l_lock',
                                'wrist_r': 'rs_wrist_r_lock'}[role]
                    self.camera_shms[role] = SharedMemoryManager(
                        CAMERA_VIEW, shared_lock[lock_key], shm_name[shm_key]
                    )

    def _resolve_runtime_config(self):
        """teleop_config_shm 의 hand_type + 토글 모드 → modality layout 결정.

        Phase M4 (PART3 §2.5): modality.json 단일 진실 출처. 우선순위:
          1. self.modality_json_path 명시 → 그 파일 layout 사용 (학습 데이터셋 메타)
          2. teleop_config_shm 의 토글 (waist_mode/head_mode/hand_type) 로 build_state_layout
        둘 다 안 되면 default (inspire / waist=on / head=on).
        """
        self.hand_type = 'inspire'
        waist_on = True
        head_on  = True
        try:
            cfg = self.teleop_config_shm.read_data()
            self.hand_type = HAND_MAPPING_INV.get(int(cfg["hand_type"]), 'inspire')
            waist_mode = WAIST_MAPPING_INV.get(int(cfg["waist_mode"]), 'hmd')
            head_mode  = HEAD_MAPPING_INV.get(int(cfg["head_mode"]),  'dxl')
            waist_on = (waist_mode == 'hmd')
            head_on  = (head_mode  == 'dxl')
        except Exception as e:
            logger_mp.warning(f"[Deploy] teleop_config read 실패: {e} — defaults 사용")
        self.hand_dim = 14 if self.hand_type == 'dex3' else 12

        # modality.json 로드 (학습 데이터셋과 동일 layout 사용 위함).
        self.layout = None
        if self.modality_json_path:
            try:
                import json as _json
                with open(self.modality_json_path) as f:
                    m = _json.load(f)
                self.layout = layout_from_modality_json(m)
                logger_mp.info(
                    f"[Deploy] modality.json layout loaded from {self.modality_json_path}: "
                    f"{[(n, d) for n, d in self.layout]}"
                )
            except Exception as e:
                logger_mp.warning(f"[Deploy] modality.json read 실패: {e} — fallback to TELEOP_CONFIG")
        if self.layout is None:
            self.layout = build_state_layout(self.hand_type, waist_on=waist_on, head_on=head_on)
            logger_mp.info(
                f"[Deploy] layout from TELEOP_CONFIG: hand={self.hand_type} "
                f"waist_on={waist_on} head_on={head_on} → "
                f"{[(n, d) for n, d in self.layout]}"
            )

        if self.zed_mode:
            self.camera_roles: List[str] = ['ego_left'] + (['ego_right'] if self.binocular else [])
        else:
            self.camera_roles = list(self.camera_shms.keys())

    def _init_policy(self):
        self.policy, self.cfg = init_dp_policy(self.model_path, device=self.device)
        # cfg 에서 추론 파라미터 추출.
        self.n_obs_steps    = int(self.cfg.n_obs_steps)
        self.n_action_steps = int(self.cfg.n_action_steps)
        # obs 히스토리 deque 초기화 (키별 maxlen=n_obs_steps).
        self._obs_hist = deque(maxlen=self.n_obs_steps)
        logger_mp.info(f"[DeployDP] ready. n_obs_steps={self.n_obs_steps} "
                       f"n_action_steps={self.n_action_steps} slow={self.slow_hz}Hz "
                       f"fast={self.fast_hz}Hz")

    # ----- observation -------------------------------------------------------
    def get_real_obs(self):
        """Pull obs dict + sample-time ts from SHM. Returns (obs, obs_ts_ns) or None.

        Phase L2: obs_ts_ns 는 모든 활성 모달리티 (body / hand / 카메라) 의 ts 들
        중 obs_ts_policy 에 따라 min (가장 stale) 또는 max (가장 최근) 선택. min
        이 의미상 더 정확 (가장 오래된 모달리티가 obs 묶음의 유효시각을 결정).
        """
        ro = self.robot_obs_shm.read_data()
        ts_body = int(ro["obs_body_ts"])
        ts_hand = int(ro["obs_hand_ts"])
        if ts_body <= 0 and ts_hand <= 0:
            return None

        obs_waist = ro["obs_waist"]; obs_head = ro["obs_head"]
        obs_arm   = ro["obs_arm"];   obs_hand_arr = ro["obs_hand"]
        qpos      = np.concatenate([obs_waist, obs_head, obs_arm]).astype(np.float32)
        hand_qpos = np.asarray(obs_hand_arr, dtype=np.float32)

        # DP 는 language 미사용 — task_name 읽기 불필요 (단일 task).

        # 카메라 frame + ts read. Phase L2: ts 후보에 카메라 ts 도 포함.
        ts_candidates = [t for t in (ts_body, ts_hand) if t > 0]

        frames: Dict[str, np.ndarray] = {}
        if self.zed_mode:
            if self.legacy_camera_shm is None:
                return None
            d = self.legacy_camera_shm.read_data()
            cam_ts = int(d.get('frame_ts', 0))
            if cam_ts > 0:
                ts_candidates.append(cam_ts)
            left = process_frame(d['frame_left'], side='zed_left')
            right = process_frame(d['frame_right'], side='zed_right') if self.binocular else None
            if left is None:
                return None
            if self.masking and self.workspace_mask_shm is not None:
                m = self.workspace_mask_shm.read_data()
                ml = m["mask_left_flat"].reshape(left.shape[:2])
                if ml.dtype != np.uint8:
                    left = (left * ml[..., None].astype(np.uint8)).astype(np.uint8)
                if right is not None:
                    mr = m["mask_right_flat"].reshape(right.shape[:2])
                    if mr.dtype != np.uint8:
                        right = (right * mr[..., None].astype(np.uint8)).astype(np.uint8)
            frames['ego_left'] = left
            if right is not None:
                frames['ego_right'] = right
        else:
            # RealSense 멀티뷰 (Phase L1).
            if not self.camera_shms:
                return None
            for role, shm in self.camera_shms.items():
                d = shm.read_data()
                cam_ts = int(d.get('frame_ts', 0))
                if cam_ts > 0:
                    ts_candidates.append(cam_ts)
                rgb = process_frame(d['frame_left'], side=f'rs_{role}')
                if rgb is None:
                    continue
                frames[role] = rgb
            if not frames:
                return None
        # DP: 단일 프레임 obs 추출 (state 28D + camera_N CHW float32 [0,1]).
        frame_obs = build_obs_frame(qpos, hand_qpos, frames,
                                    self.hand_type, self.camera_key_map)

        if not ts_candidates:
            return None
        if self.obs_ts_policy == 'min':
            obs_ts_ns = min(ts_candidates)
        else:
            obs_ts_ns = max(ts_candidates)

        # obs 히스토리 누적 (To=n_obs_steps). deque 가 maxlen 으로 오래된 것 자동 제거.
        if self._obs_hist is None:
            self._obs_hist = deque(maxlen=self.n_obs_steps)
        self._obs_hist.append(frame_obs)
        # To 개 미만이면 첫 프레임 복제로 패딩 (시작 시).
        hist = list(self._obs_hist)
        while len(hist) < self.n_obs_steps:
            hist.insert(0, hist[0])

        # (1, To, *) 텐서 dict 구성 — predict_action 입력 형식.
        import torch
        obs: Dict[str, "torch.Tensor"] = {}
        keys = hist[-1].keys()
        for k in keys:
            stacked = np.stack([h[k] for h in hist], axis=0)   # (To, *)
            obs[k] = torch.from_numpy(stacked[None, ...]).float()  # (1, To, *)

        with self._obs_lock:
            self.qpos      = qpos
            self.hand_qpos = hand_qpos
        return (obs, obs_ts_ns)

    # ----- inference + chunk handling ---------------------------------------
    def dp_inference(self, obs: dict, obs_ts_ns: int):
        """policy.predict_action → upsample → lag trim → cross-fade → latch primary_*"""
        if self.policy is None:
            return
        import torch
        # DP predict_action 입력: {key: (1,To,*)}, 출력: {action:(1,Ta,28), action_pred:...}.
        dev = next(self.policy.parameters()).device
        obs_dev = {k: v.to(dev) for k, v in obs.items()}
        with torch.no_grad():
            result = self.policy.predict_action(obs_dev)
        action_chunk = result["action"].detach().cpu().numpy()   # (1, n_action_steps, 28)
        t_after_ns = time.perf_counter_ns()

        # DP action 28D = arm14+hand14 분해 → (T,19 full), (T,14|12 hand).
        full, hand = dp_action_chunk_to_arrays(action_chunk, self.hand_type)

        # arm = spline(k=5), hand = linear(k=1). 10→fast(60) 업샘플.
        full_up = upsample_actions(full, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=5)
        hand_up = upsample_actions(hand, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=1)

        # ---- lag compensation (stale 기준 obs_ts 사용) ----
        lag_ns       = max(0, t_after_ns - obs_ts_ns)
        trim_samples = int(round(lag_ns * self.fast_hz / 1e9)) if self.lag_compensate else 0
        if trim_samples > 0 and trim_samples < len(full_up) - 1:
            full_up = full_up[trim_samples:]
            hand_up = hand_up[trim_samples:]
        elif trim_samples >= len(full_up) - 1:
            full_up = full_up[-1:]
            hand_up = hand_up[-1:]

        self._lag_chunk_count += 1
        self._lag_ns_acc     += lag_ns
        if lag_ns > self._lag_ns_max:
            self._lag_ns_max  = lag_ns
        if self._lag_chunk_count % self.lag_log_every == 0:
            avg_ms = (self._lag_ns_acc / self._lag_chunk_count) / 1e6
            max_ms = self._lag_ns_max / 1e6
            logger_mp.info(
                f"[Deploy] lag stats: avg={avg_ms:.1f}ms max={max_ms:.1f}ms "
                f"trim_last={trim_samples} chunk_remain={len(full_up)} compensate={self.lag_compensate}"
            )

        # Cross-fade with tail of previous chunk.
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

        # left_arm 6D-fallback: 모델 left_arm 이 6D 면 현재 qpos[11] 유지 + right 7D.
        if arm_slice.size == 13:
            left7 = self.qpos[5:12].copy().astype(np.float32)
            left7[:6] = arm_slice[:6]
            right7 = arm_slice[6:]
            action_arm = np.concatenate([left7, right7], axis=0)
        else:
            action_arm = arm_slice  # 14D

        # hand action SHM 은 항상 14D. Inspire=12D 면 뒤 2개 0 pad.
        hand14 = np.zeros(14, dtype=np.float64)
        n = min(len(hand_action_np), 14)
        hand14[:n] = hand_action_np[:n].astype(np.float64)

        # Phase L2 (P0-2): body/hand publish ts 분리. 같은 perf_counter 한 번 호출에서
        # 두 값을 따로 캡처해 의미적 분리 + 향후 publish 경로 분리 시 호환.
        ts_body = np.int64(time.perf_counter_ns())
        ts_hand = np.int64(time.perf_counter_ns())
        self.robot_action_shm.write_data(
            action_body_ts   =ts_body,
            action_hand_ts   =ts_hand,
            action_waist     =action_waist.astype(np.float64),
            action_waist_tauff=np.zeros(3, dtype=np.float64),
            action_head      =action_head.astype(np.float64),
            action_arm       =action_arm.astype(np.float64),
            action_arm_tauff =np.zeros(14, dtype=np.float64),
            action_hand      =hand14,
        )

    # ----- UI trigger --------------------------------------------------------
    def get_ui_mode(self):
        try:
            self.deploy_mode = bool(self.record_mode_shm.read_data()["deploy"])
        except Exception:
            pass

    # ----- per-loop work units ----------------------------------------------
    def do_slow(self):
        res = self.get_real_obs()
        if res is None:
            return
        obs, obs_ts_ns = res
        self.dp_inference(obs, obs_ts_ns)

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
                        self._init_policy()
                    except Exception as e:
                        logger_mp.exception(f"[DeployDP] policy init failed: {e}")
                        time.sleep(1.0); continue
                try:
                    self.do_slow()
                except Exception as e:
                    logger_mp.exception(f"[DeployDP] slow loop error: {e}")
            else:
                with self._ctrl_lock:
                    self.primary_actions = None
                    self.primary_hand_actions = None
                    self.primary_index = 0
                # deploy 종료 시 obs 히스토리 초기화 (다음 시작 시 fresh).
                if self._obs_hist is not None:
                    self._obs_hist.clear()
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
        handles = [self.robot_obs_shm, self.robot_action_shm,
                   self.gr00t_task_shm, self.record_mode_shm,
                   self.teleop_config_shm, self.workspace_mask_shm,
                   self.legacy_camera_shm]
        handles.extend(self.camera_shms.values())
        for s in handles:
            if s is not None:
                try: s.worker_close()
                except Exception: pass
        logger_mp.info("[Deploy] DP_Inference stopped.")
