"""worker_deploy_policy: external policy (GR00T) inference adapter for G1.

Phase L (Part 2 P0-1 + P0-2 + P1-3) 재작성:
  - hand 종류 (Inspire 6+6 / DEX3 7+7) 를 teleop_config_shm 의 hand_type 으로 자동
    분기. obs_dict / action_to_array 가 hand DOF 에 맞게 동작.
  - 카메라 입력 = role 별 CAMERA_VIEW SHM (rs_ego_shm / rs_wrist_l_shm /
    rs_wrist_r_shm) attach. modality.json 의 video.{ego,wrist_l,wrist_r}.original_key
    = "observation.images.<role>" 와 정합되도록 obs dict 의 video 키도
    "video.ego", "video.wrist_l", "video.wrist_r" 사용. ZED 는 mode='gr00t_zed'
    유지 (single CAMERA SHM, legacy).
  - obs_ts_ns 는 모든 활성 모달리티 (body / hand / 카메라들) 중 가장 stale (min)
    또는 가장 최근 (max) 을 선택 — --obs-ts-policy 로 조정. default 는 min
    (가장 stale 한 모달리티가 obs 유효시각 결정 — Part2 §2.1 권고).
  - action_body_ts / action_hand_ts 분리. publish 시각만 다른 게 아니라 향후
    body/hand publish 경로가 분리될 때 영향 없도록.
  - hand upsample 은 spline → linear (k=1). DEX3 grasp 같은 이산/포화 신호의
    ringing 방지. arm/waist 는 spline 유지.

main.py 의 teleop env 에서는 gr00t 가 없어도 import 안전 (lazy import) — main.py
가 spawn 하지 않는 worker 라서 worker file 자체 import 만 OK 면 됨. 실 추론은
별도 conda env (gr00t) 의 evaluate.py 가 SHM 에 attach 해 실행.
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
logger_mp = logging_mp.get_logger(__name__)


# =============================================================================
# Camera frame ring buffer (Phase B — video temporal history)
# =============================================================================

class _CameraFrameRing:
    """Per-camera ring buffer holding (frame_ts_ns, frame_rgb_uint8) entries.

    학습 데이터는 video.delta_indices=[-20, 0] 으로, 20fps 다운샘플 후 -20 = 정확히
    1.0 초 전 프레임을 의미한다. 카메라가 60fps 로 SHM 갱신되므로 (worker_camera.py
    L67 enable_stream @60fps), 추론(20Hz slow tick) 시점 마다 ts 기반으로 1초 전
    프레임을 정확히 pick 하기 위한 ring buffer.

    동시성:
      - 폴링 thread (60Hz) 가 append (단일 producer)
      - inference thread (20Hz) 가 pick (단일 consumer)
      - lock 으로 snapshot 보호 (Python deque append 자체는 thread-safe 하나 ts 검색은
        스냅샷 일관성 필요).

    Sizing:
      - 60fps × 1.2s = 72 슬롯. 여유 위해 capacity=120.
      - 프레임당 360×640×3 = 691 KB. 카메라 3대 × 120 = ~248 MB. 허용 범위.

    Warmup:
      - 첫 1초간 ring 에 1초 전 ts 의 entry 가 없으면 None 반환 → caller 가 현재
        프레임 복제로 처리 (학습 step_index<20 시 allow_padding clamp 거동과 일치).
    """

    def __init__(self, capacity: int = 120, role: str = ""):
        self._buf: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.role = role
        self._last_ts: int = 0

    def push(self, ts_ns: int, frame_rgb: np.ndarray) -> bool:
        """Append (ts, frame) entry. Dedup on identical ts (camera 가 같은 frame 을
        반복 publish 한 경우). 반환 True = 새 frame 으로 추가, False = dedup."""
        if ts_ns <= self._last_ts:
            return False
        with self._lock:
            self._buf.append((int(ts_ns), frame_rgb))
            self._last_ts = int(ts_ns)
        return True

    def latest(self) -> Optional[tuple]:
        """가장 최근 (ts, frame) 반환. ring 이 비었으면 None."""
        with self._lock:
            if not self._buf:
                return None
            return self._buf[-1]

    def pick_at(self, target_ts_ns: int, tol_ns: int = 80_000_000) -> Optional[tuple]:
        """target_ts 에 가장 가까운 (ts, frame) 반환. tol 안에 들어와야 valid (기본
        80ms = 카메라 1.5 frame @60fps 여유). ring 에 target_ts 보다 이전 entry 가
        없거나 가까운 게 tol 밖이면 None.

        구현: target_ts 보다 작거나 같은 entry 중 가장 큰 ts (즉 target 직전의
        가장 최근 frame — ZOH 의미). 학습 시 다운샘플도 [step-20] = 그 시점의
        프레임이므로 동일 의미.
        """
        with self._lock:
            if not self._buf:
                return None
            # 선형 스캔 (deque, len<=120 라 비용 무시 수준).
            best = None
            for ts, fr in self._buf:
                if ts <= target_ts_ns and (best is None or ts > best[0]):
                    best = (ts, fr)
            if best is None:
                return None
            if abs(best[0] - target_ts_ns) > tol_ns:
                return None
            return best

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


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


def upsample_actions(seq: np.ndarray, slow_hz: float = 20.0, fast_hz: float = 60.0,
                     k: int = 5) -> np.ndarray:
    """Spline-upsample a (T, D) action chunk from slow_hz to fast_hz.

    slow_hz=20: GR00T action chunk 의 각 step 은 학습 데이터(60→20 다운샘플) 타임스텝.
    fast_hz=60: 실제 arm 제어 루프 주파수 (worker_g1_ctrl ACT_HZ=60, 수집 체인과 일치).
                deploy fast loop 가 60Hz 로 robot_action_shm 에 write → g1_ctrl 이 60Hz read.

    Phase L3 (P1-3): hand 처럼 이산/포화 신호는 k=1 (linear) 로 호출해 ringing 방지.
    arm/waist 같은 연속 신호는 k=5 (quintic) 유지.
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


def build_obs_dict(task_name: str, qpos: np.ndarray, hand_qpos: np.ndarray,
                   frames: Dict[str, List[np.ndarray]], hand_type: str,
                   layout, language_key: str) -> dict:
    """N1.7 Gr00tPolicy 용 *중첩* obs dict (공식 권고 — custom robot 은 Gr00tPolicy 직접 사용).

    N1.7 Gr00tPolicy.get_action 입력 형식 (gr00t_policy.py L246-342):
        obs = {
          "video":    { role: (B,T,H,W,C) uint8 },     # T == video delta_indices 길이
          "state":    { name: (B,T,D) float32 },        # T == state delta_indices 길이(=1)
          "language": { language_key: list[list[str]] } # (B, 1)
        }
    여기서 B=1. video T 는 학습 시 video.delta_indices 길이 (g1_dex3_config 는 2).
    video uint8 / state float32 보장.

    layout: build_state_layout(...) 또는 layout_from_modality_json(...) 결과.
            modality.json 의 state 키만 포함 — 학습 데이터셋 layout 과 정합 (차원 자동).
    language_key: policy.language_key (체크포인트서 자동 = 'annotation.human.task_description').
                  하드코딩하지 않고 policy 가 알려주는 키를 써서 학습-추론 정합 보장.
    frames: {role: list[rgb_np uint8]} — 각 카메라 별 T 개 frame (학습 delta_indices 순서).
            예) video.delta_indices=[-20, 0] 이면 [과거(1초 전), 현재] 순서.
            extract_step_data L41 이 동일 순서 ([step+(-20), step+0]) 로 video 를
            로드하므로 deploy 도 같은 순서 stack 필요. ★ 순서 중요.
    """
    parts = _parts_from_obs(qpos, hand_qpos, hand_type)
    obs: dict = {"video": {}, "state": {}, "language": {}}
    for name, _dim in layout:
        # (B=1, T=1, D) float32
        obs["state"][name] = parts[name][None, None, :].astype(np.float32)
    for role, frame_list in frames.items():
        # frame_list: list of (H, W, C) uint8 frames in delta_indices order.
        # → np.stack(axis=0) = (T, H, W, C) → [None] = (B=1, T, H, W, C) uint8.
        stacked = np.stack(frame_list, axis=0).astype(np.uint8)
        obs["video"][role] = stacked[None, ...]
    # language: (B=1, T=1) = list[list[str]]
    obs["language"][language_key] = [[task_name]]
    return obs


def action_to_array(action: dict, i: int, hand_type: str, layout) -> tuple:
    """Pull one timestep out of a N1.7 GR00T action chunk.

    N1.7 Gr00tPolicy.get_action 출력 (gr00t_policy.py L444, check_action):
        action[<modality_key>] = np.ndarray (B, T, D) float32
        키는 modality_keys 그대로 — 접두사 없음 (left_arm, right_arm, left_hand, right_hand).
        (PolicyWrapper 만 'action.' 접두사를 붙임. 여기선 Gr00tPolicy 직접 사용.)

    layout 에 waist/head 가 있으면 그 키도 시도하나, g1_dex3_config 의 action modality_keys
    는 left_arm/right_arm/left_hand/right_hand 4개뿐 → waist/head 는 자동 zero.

    Returns (action_np[19] full body, hand_action_np[12 or 14]) for write_shm.
    write_shm 은 fixed-shape (waist[3],head[2],arm[14],hand[14]) 이므로 없는 모달리티는 zero.
    """
    layout_names = {n for n, _ in layout}

    def _opt(key: str, dim: int) -> np.ndarray:
        # N1.7: 접두사 없는 키. action[key] shape (B=1, T, D) → [0, i] = (D,)
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


# =============================================================================
# Lazy GR00T policy loader
# =============================================================================

def init_gr00t_policy(model_path: str,
                      embodiment_tag: str = "new_embodiment",
                      device: str = "cuda"):
    """Build a N1.7 Gr00tPolicy. Import is local so this module loads in any env.

    N1.7 (gr00t/policy/gr00t_policy.py L74-81) 시그니처:
        Gr00tPolicy(embodiment_tag, model_path, *, device, strict=True)
    - modality_config/modality_transform/denoising_steps 인자 없음.
      finetune 시 --modality-config-path 로 준 config 가 체크포인트(processor)에 저장되어
      inference 시 자동 로딩됨 (공식 policy.md). 즉 deploy 는 embodiment_tag + model_path
      + device 만 필요. DATA_CONFIG_MAP 불필요 (N1.7 은 gr00t/configs/data 로 이동, deploy 무관).
    - language_key 도 체크포인트에서 자동 (policy.language_key = modality_keys[0]).
    """
    from gr00t.policy import Gr00tPolicy
    return Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device=device,
    )


# =============================================================================
# Inference worker
# =============================================================================

class Gr00t_Inference:
    """GR00T inference worker. hand_type + camera roles 자동 분기 (Phase L1).

    - slow loop (default 20Hz): read obs, call policy, upsample to fast_hz,
      blend cross-fade with the tail of the previous chunk, store as
      `self.primary_actions / primary_hand_actions`.
    - fast loop (default 60Hz): pop one timestep, apply --action_method,
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
        model_path: str = "",
        data_config_key: str = "unitree_g1",
        embodiment_tag: str = "new_embodiment",
        action_method: str = "tem",             # {"base", "maf", "tem"}
        decay: float = 0.3,
        window_size: int = 5,
        slow_hz: float = 20.0,
        fast_hz: float = 60.0,                  # arm 제어 루프(worker_g1_ctrl ACT_HZ=60)와 일치
        denoising_steps: int = 4,
        device: str = "cuda",                   # N1.7 Gr00tPolicy device
        binocular: bool = True,
        masking: bool = False,
        lag_compensate: bool = True,
        lag_log_every: int = 50,
        obs_ts_policy: str = 'min',             # Phase L2: 'min' (stale 기준) | 'max'
        modality_json_path: Optional[str] = None,  # Phase M4 (PART3 §2.5): 학습 데이터셋의 modality.json
        chunking_mode: str = "legacy",          # {"legacy", "soft_rtc"} — soft_rtc = receding horizon + tail-continuation (backward-jump 방지)
        exec_frac: float = 0.5,                 # soft_rtc: 버퍼의 이 비율만큼 소비 후 replan (receding horizon). ref=0.5
        blend_curve: str = "exp",               # soft_rtc: tail↔새청크 blend 가중 곡선 {"exp","linear"} (LeRobot RTC prefix_attention_schedule 차용)
        inference_mode: str = "pytorch",        # {"pytorch", "trt_full_pipeline"} — trt = Gr00tPolicy in-place 패치(setup_tensorrt_engines)
        trt_engine_path: str = "",              # trt_full_pipeline 시 엔진 디렉토리 (vit_bf16.engine 등). 예: ~/Isaac-GR00T/gr00t_trt_deploy_20k/engines
        trt_scripts_path: str = "/home/kist/Isaac-GR00T/scripts/deployment",  # trt_model_forward.py 위치 (sys.path 추가용)
    ):
        assert mode in ("gr00t_rs_multi", "gr00t_zed", "gr00t"), f"unsupported mode: {mode}"
        # 'gr00t' (legacy single RealSense) 는 alias for gr00t_rs_multi (ego only).
        if mode == "gr00t":
            mode = "gr00t_rs_multi"
        assert action_method in ("base", "maf", "tem"), f"unknown method: {action_method}"
        assert obs_ts_policy in ("min", "max"), f"unknown obs_ts_policy: {obs_ts_policy}"
        assert chunking_mode in ("legacy", "soft_rtc"), f"unknown chunking_mode: {chunking_mode}"
        assert blend_curve in ("exp", "linear"), f"unknown blend_curve: {blend_curve}"
        assert inference_mode in ("pytorch", "trt_full_pipeline"), f"unknown inference_mode: {inference_mode}"

        self.shared_event = shared_event
        self.mode         = mode
        self.zed_mode     = (mode == "gr00t_zed")
        self.binocular    = bool(binocular and self.zed_mode)
        self.masking      = bool(masking)
        self.model_path   = model_path
        self.data_config_key = data_config_key   # N1.7 미사용 (체크포인트 자동 로딩). 하위호환용.
        self.embodiment_tag  = embodiment_tag
        self.denoising_steps = int(denoising_steps)  # N1.7 Gr00tPolicy 생성 인자 아님. 미사용.
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
        self.chunking_mode   = chunking_mode
        self.exec_frac       = float(exec_frac)
        self.blend_curve     = blend_curve
        self.inference_mode  = inference_mode
        self.trt_engine_path = trt_engine_path
        self.trt_scripts_path = trt_scripts_path
        self._rtc_threshold  = 0      # soft_rtc: primary_index 가 이 값 이상이면 replan (첫 tick=0 이라 즉시 추론)
        self._rtc_count      = 0
        self._lag_chunk_count = 0
        self._lag_ns_acc      = 0
        self._lag_ns_max      = 0

        # SHM handles + hand_type / active camera roles 결정 + modality layout 로드
        self._init_shm(shm_name, shared_lock)
        self._resolve_runtime_config()

        # State shared between slow / fast loops.
        self._obs_lock     = threading.Lock()
        self._ctrl_lock    = threading.Lock()
        self.policy        = None
        # N1.7 language_key 기본값 (정책 로딩 시 policy.language_key 로 갱신).
        # build_obs_dict 가 정책 로딩 전 호출돼도 AttributeError 방지.
        self.language_key  = "annotation.human.task_description"
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

        # Phase B: 카메라 frame ring buffer (video.delta_indices=[-20, 0] 지원).
        # 학습 data: 20fps × -20 = 1.0 초 전. deploy 도 1.0 초 전 frame 을 모델에 전달
        # 해야 학습-배포 정합. 카메라 60fps SHM 갱신 (worker_camera.py) 을 polling
        # thread 가 모두 캡처해 ring 에 저장 → slow tick (20Hz) 시점에 ts 기반 pick.
        # 카메라 role 별 독립 ring (ego/wrist_l/wrist_r 또는 zed 모드의 ego_left/_right).
        self._video_history_offset_ns: int = 1_000_000_000   # 1.0초 = -20 @ 20fps (legacy 상수)
        self._video_history_tol_ns: int = 80_000_000          # 80ms pick tolerance
        # video delta_indices: 정책 로딩 시 모델 config 로 갱신 → frame 구성 자동 적응
        # (2-frame [-20,0] / 1-frame [0] 둘 다 한 코드로 호환). 기본값 = 2-frame.
        self._video_deltas: List[int] = [-20, 0]
        self._video_fps: float = 20.0      # 학습 데이터 다운샘플 fps (delta_indices 단위)
        self._cam_rings: Dict[str, _CameraFrameRing] = {
            role: _CameraFrameRing(capacity=120, role=role) for role in self.camera_roles
        }

        # Lifecycle
        self._stop_event = threading.Event()
        self._slow_thread = threading.Thread(target=self._inference_loop, daemon=True, name="DEPLOY_SLOW")
        self._fast_thread = threading.Thread(target=self._ctrl_loop,      daemon=True, name="DEPLOY_FAST")
        # Phase B: 60Hz 카메라 frame ring 채우는 별도 polling thread. slow tick
        # 추론 지연과 무관하게 ring 이 늘 채워져 있어야 1초 전 frame 을 정확히 줄 수 있음.
        self._cam_poll_thread = threading.Thread(target=self._camera_poll_loop, daemon=True, name="DEPLOY_CAMPOLL")
        self._slow_thread.start()
        self._fast_thread.start()
        self._cam_poll_thread.start()
        logger_mp.info(
            f"[Deploy] Gr00t_Inference started (mode={mode}, hand={self.hand_type}, "
            f"camera_roles={self.camera_roles}, action_method={action_method}, "
            f"chunking_mode={self.chunking_mode}, exec_frac={self.exec_frac}, blend={self.blend_curve}, "
            f"slow={slow_hz}Hz, fast={fast_hz}Hz, obs_ts_policy={obs_ts_policy}, "
            f"video_history_offset={self._video_history_offset_ns/1e9:.2f}s, "
            f"ring_capacity=120 per camera)"
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

    def _init_gr00t_policy(self):
        try:
            task_name = self.gr00t_task_shm.read_data()["task_name"].item().strip()
        except Exception:
            task_name = ""
        self.task_name = task_name
        self.policy = init_gr00t_policy(
            model_path=self.model_path,
            embodiment_tag=self.embodiment_tag,
            device=self.device,
        )
        # ★ TRT full pipeline: Gr00tPolicy 를 in-place 패치 (내부 모델→TRT 엔진). get_action 인터페이스 동일.
        if self.inference_mode == "trt_full_pipeline":
            try:
                import sys as _sys
                if self.trt_scripts_path not in _sys.path:
                    _sys.path.insert(0, self.trt_scripts_path)
                from trt_model_forward import setup_tensorrt_engines
                setup_tensorrt_engines(self.policy, self.trt_engine_path, mode="n17_full_pipeline")
                print(f"[Deploy] TRT engines 로드 완료: {self.trt_engine_path} (n17_full_pipeline)", flush=True)
            except Exception as e:
                logger_mp.exception(f"[Deploy] TRT setup 실패 → PyTorch fallback: {e}")
                print(f"[Deploy] TRT setup 실패 → PyTorch 로 진행: {e}", flush=True)
                self.inference_mode = "pytorch"
        # N1.7: language_key 는 체크포인트에서 자동 결정됨 (policy.language_key).
        # build_obs_dict 에 넘겨 학습-추론 language 키 정합 보장.
        self.language_key = getattr(self.policy, "language_key",
                                    "annotation.human.task_description")
        # ★ 모델의 video delta_indices 읽어 deploy frame 구성 자동 적응 (1-frame/2-frame 호환)
        try:
            self._video_deltas = list(self.policy.modality_configs["video"].delta_indices)
        except Exception as e:
            logger_mp.warning(f"[Deploy] video delta_indices 읽기 실패: {e} — 기본 {self._video_deltas} 사용")
        logger_mp.info(f"[Deploy] Policy loaded (N1.7). embodiment={self.embodiment_tag} "
                       f"language_key={self.language_key!r} task={task_name!r} "
                       f"video_deltas={self._video_deltas}({len(self._video_deltas)}frame)")
        print(f"[Deploy] video_deltas={self._video_deltas} ({len(self._video_deltas)} frame)", flush=True)
        # ★ warmup-on-load: 첫 추론 cold-start(CUDA 커널 컴파일) 제거 → #1 STALL위험 해소
        try:
            self._warmup_policy()
        except Exception as e:
            logger_mp.warning(f"[Deploy] warmup 실패(무시): {e}")

    def _warmup_policy(self):
        """정책 로딩 직후 dummy obs 로 get_action 1회 — CUDA 커널 warmup (첫 실추론 cold-start 제거)."""
        H, W = 360, 640
        T = max(1, len(self._video_deltas))
        vk = list(self.policy.modality_configs["video"].modality_keys)
        obs = {"video": {}, "state": {}, "language": {}}
        for k in vk:
            obs["video"][k] = np.zeros((1, T, H, W, 3), dtype=np.uint8)
        for name, dim in self.layout:
            obs["state"][name] = np.zeros((1, 1, dim), dtype=np.float32)
        obs["language"][self.language_key] = [[self.task_name or "warmup"]]
        t0 = time.perf_counter()
        self.policy.get_action(obs)
        print(f"[Deploy] policy warmup done ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)

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

        try:
            self.task_name = self.gr00t_task_shm.read_data()["task_name"].item().strip()
        except Exception:
            pass

        # 카메라 frame 을 ring 에서 가져와 모델의 video delta_indices 순서대로 stack.
        # ★ self._video_deltas 기반 자동 적응: [-20,0]=2frame[과거,현재] / [0]=1frame[현재].
        #   각 delta d 에 대해 target = now + d*(1/video_fps) 시점 frame pick (d=0 은 현재).
        #   학습 extract_step_data 가 delta 순서대로 frame 로드하므로 동일 순서 보장.
        # ts 후보 (obs_ts_policy 용): body/hand + 각 카메라의 현재 frame ts.
        ts_candidates = [t for t in (ts_body, ts_hand) if t > 0]
        now_ns = time.perf_counter_ns()

        frames: Dict[str, List[np.ndarray]] = {}
        # masking workspace mask (zed_mode 만 사용). 학습 시엔 mask 적용 X 가 일반적
        # 이지만 기존 동작 보존 위해 zed_mode 의 양 frame 에 동일 mask 적용.
        ws_mask = None
        if self.zed_mode and self.masking and self.workspace_mask_shm is not None:
            try:
                ws_mask = self.workspace_mask_shm.read_data()
            except Exception:
                ws_mask = None

        def _apply_ws_mask(role: str, frame: np.ndarray) -> np.ndarray:
            """zed_mode masking: left/right 별 mask 곱셈. 다른 모드는 frame 그대로."""
            if ws_mask is None or not self.zed_mode:
                return frame
            key = "mask_left_flat" if role == 'ego_left' else "mask_right_flat"
            if key not in ws_mask:
                return frame
            m = ws_mask[key].reshape(frame.shape[:2])
            if m.dtype != np.uint8:
                return (frame * m[..., None].astype(np.uint8)).astype(np.uint8)
            return frame

        for role in self.camera_roles:
            ring = self._cam_rings.get(role)
            if ring is None or len(ring) == 0:
                # ring 비어 있음 (시작 직후 첫 polling 전). slow tick skip.
                return None
            latest = ring.latest()
            if latest is None:
                return None
            cur_ts, cur_frame = latest
            ts_candidates.append(cur_ts)

            # video delta_indices 순서대로 frame 구성 (auto-adapt).
            # d=0 → 현재 frame. d<0 → |d|/video_fps 초 전 frame (ring ts 기반 pick).
            # ring 에 해당 과거 frame 없으면(warmup) 현재 복제 — 학습 allow_padding clamp 와 정합.
            seq: List[np.ndarray] = []
            for d in self._video_deltas:
                if d == 0:
                    fr = cur_frame
                else:
                    target_ns = now_ns + int(d * 1e9 / self._video_fps)   # d<0 → 과거
                    picked = ring.pick_at(target_ns, tol_ns=self._video_history_tol_ns)
                    fr = picked[1] if picked is not None else cur_frame   # warmup: 현재 복제
                seq.append(_apply_ws_mask(role, fr))
            frames[role] = seq   # delta_indices 순서 (1-frame=[현재], 2-frame=[과거,현재])

        if not frames:
            return None
        # Phase M4 (PART3 §2.5): 동적 obs dict (layout 기반). N1.7: 중첩 dict + language_key.
        obs = build_obs_dict(self.task_name, qpos, hand_qpos, frames,
                             self.hand_type, self.layout, self.language_key)

        if not ts_candidates:
            return None
        if self.obs_ts_policy == 'min':
            obs_ts_ns = min(ts_candidates)
        else:
            obs_ts_ns = max(ts_candidates)

        with self._obs_lock:
            self.qpos      = qpos
            self.hand_qpos = hand_qpos
        return (obs, obs_ts_ns)

    # ----- inference + chunk handling ---------------------------------------
    def gr00t_inference(self, obs: dict, obs_ts_ns: int):
        """policy.get_action → upsample → lag trim → cross-fade → latch primary_*"""
        if self.policy is None:
            return
        # N1.7 get_action 은 (action, info) 튜플 반환 (policy.py L80-105 return action, info).
        action, _info = self.policy.get_action(obs)
        t_after_ns = time.perf_counter_ns()

        # chunk 길이 T: action key 중 첫 활성 key 의 shape 에서 추출.
        # N1.7 Gr00tPolicy 직접 사용 → 키 접두사 없음 (left_arm 등). g1_dex3 action
        # modality_keys 에 waist/head 없으므로 left_arm 사용 (항상 존재).
        first_key = "waist" if "waist" in action else "left_arm"
        T = action[first_key].shape[1]
        full = []
        hand = []
        for i in range(T):
            a_np, h_np = action_to_array(action, i, self.hand_type, self.layout)
            full.append(a_np)
            hand.append(h_np)
        full = np.stack(full, axis=0)
        hand = np.stack(hand, axis=0)

        # Phase L3 (P1-3): arm/waist = spline(k=5), hand = linear(k=1).
        # hand 는 grasp toggle 같은 이산/포화 신호라 quintic spline 시 overshoot.
        full_up = upsample_actions(full, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=5)
        hand_up = upsample_actions(hand, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=1)

        # ---- Phase E: lag compensation (Part 2 §2: stale 기준 obs_ts 사용) ----
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
        # primary_index / _prev_* 는 fast loop(60Hz)가 동시에 갱신하므로, race 로
        # arm·hand 에 서로 다른 idx 가 적용돼 길이가 어긋나는 것(예: (91,1)vs(90,14)
        # broadcast 실패)을 막기 위해 lock 안에서 일관된 스냅샷을 한 번에 읽는다.
        with self._ctrl_lock:
            idx       = self.primary_index
            prev_arm  = self._prev_actions
            prev_hand = self._prev_hand_actions
        if prev_arm is not None and idx < len(prev_arm):
            remain_prev = prev_arm[idx:]
            ovl = min(len(remain_prev), len(full_up))
            if ovl > 0:
                alpha = np.linspace(0.0, 1.0, ovl)[:, None]
                full_up[:ovl] = (1 - alpha) * remain_prev[:ovl] + alpha * full_up[:ovl]
                # hand 는 자신의 길이로 ovl 재클램프 (arm 과 어긋나도 안전).
                if prev_hand is not None:
                    ovl_h = min(ovl, len(prev_hand) - idx, len(hand_up))
                    if ovl_h > 0:
                        a_h = alpha[:ovl_h]
                        hand_up[:ovl_h] = (1 - a_h) * prev_hand[idx:idx + ovl_h] + a_h * hand_up[:ovl_h]

        with self._ctrl_lock:
            self._prev_actions      = self.primary_actions
            self._prev_hand_actions = self.primary_hand_actions
            self.primary_actions      = full_up
            self.primary_hand_actions = hand_up
            self.primary_index        = 0
            self._action_buf.clear()
            self._hand_buf.clear()
        self.start_loop = True

    # ----- soft RTC (receding horizon + tail-continuation) ------------------
    def _blend_alpha(self, n: int) -> np.ndarray:
        """overlap 구간 blend 가중 0→1. exp=초반 tail 강하게 유지(프리즈)후 후반 새청크로.
        (LeRobot RTC prefix_attention_schedule 의 아이디어를 attention 아닌 blend 가중으로 차용)."""
        t = np.linspace(0.0, 1.0, n)
        if self.blend_curve == "exp":
            g = 3.0
            a = (np.exp(g * t) - 1.0) / (np.exp(g) - 1.0)   # convex: 느린 시작(tail 유지) → 빠른 끝
        else:
            a = t
        return a[:, None]

    def do_slow_rtc(self):
        """soft RTC: 버퍼를 exec_frac 만큼 소비한 뒤에만 replan (receding horizon).
        그동안 fast loop 가 현재 청크의 *전진하는* 미실행 tail 을 계속 소비 →
        index 0 리셋/매tick replan 이 만들던 stale-obs backward jump 를 구조적으로 제거."""
        with self._ctrl_lock:
            have = self.primary_actions is not None
            idx  = self.primary_index
            thr  = self._rtc_threshold
        if have and idx < thr:
            return  # 아직 replan 시점 아님 — 현재 청크 계속 소비
        res = self.get_real_obs()
        if res is None:
            return
        obs, obs_ts_ns = res
        self._replan_rtc(obs, obs_ts_ns)

    def _replan_rtc(self, obs: dict, obs_ts_ns: int):
        """get_action → 측정 lag 로 native 청크 정렬(trim) → upsample → 현재 미실행 tail 과 blend → latch."""
        if self.policy is None:
            return
        action, _info = self.policy.get_action(obs)
        t_after_ns = time.perf_counter_ns()
        first_key = "waist" if "waist" in action else "left_arm"
        T = action[first_key].shape[1]
        full = []; hand = []
        for i in range(T):
            a_np, h_np = action_to_array(action, i, self.hand_type, self.layout)
            full.append(a_np); hand.append(h_np)
        full = np.stack(full, axis=0); hand = np.stack(hand, axis=0)

        # ── 입력↔출력 추적 (원인 좁히기): 처음 6회 + 매 lag_log_every
        if self._rtc_count < 6 or (self._rtc_count % self.lag_log_every == 0):
            try:
                st = obs.get("state", {})
                o_la = np.asarray(st["left_arm"])[0, 0]  if "left_arm"  in st else None
                o_ra = np.asarray(st["right_arm"])[0, 0] if "right_arm" in st else None
                msg = (f"[RTC-trace #{self._rtc_count}] T={T} "
                       f"obs.L={np.round(o_la,3) if o_la is not None else None} "
                       f"obs.R={np.round(o_ra,3) if o_ra is not None else None} | "
                       f"act.L[0]={np.round(full[0,5:12],3)} act.L[-1]={np.round(full[-1,5:12],3)} | "
                       f"act.R[0]={np.round(full[0,12:19],3)} act.R[-1]={np.round(full[-1,12:19],3)}")
                if o_la is not None:
                    msg += f" | act.L[0]-obs.L={np.round(full[0,5:12]-o_la,3)}(rel복원OK면~0)"
                print(msg, flush=True)
            except Exception as _e:
                print(f"[RTC-trace] err: {_e}", flush=True)

        # 보강1: 측정 lag(t_after-obs_ts)→native(20Hz) step 환산 후 그만큼 앞을 trim
        #   (이미 경과한 prefix 제거 = 새 청크를 "지금"에 정렬). 업샘플 전 native 에서 적용(보강 timebase).
        lag_ns = max(0, t_after_ns - obs_ts_ns)
        lag_native = int(round(lag_ns * self.slow_hz / 1e9)) if self.lag_compensate else 0
        lag_native = max(0, min(lag_native, T - 2))
        full_n = full[lag_native:]; hand_n = hand[lag_native:]

        full_up = upsample_actions(full_n, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=5)
        hand_up = upsample_actions(hand_n, slow_hz=self.slow_hz, fast_hz=self.fast_hz, k=1)

        # 현재 청크의 미실행 tail 과 blend (구조적 연속화). alpha=0 에서 tail → 명령 점프 0.
        with self._ctrl_lock:
            idx = self.primary_index
            tail   = self.primary_actions[idx:].copy()      if self.primary_actions is not None else None
            tail_h = self.primary_hand_actions[idx:].copy() if self.primary_hand_actions is not None else None
        splice_l2 = 0.0
        if tail is not None and len(tail) > 0 and len(full_up) > 0:
            ovl = int(min(len(tail), len(full_up), len(tail_h), len(hand_up)))
            if ovl > 0:
                # 보강(splice 정렬): tail[0](현재 팔 명령) vs 새 청크 정렬 시작값. 0 에 가까워야 정렬 정확.
                splice_l2 = float(np.linalg.norm(np.asarray(tail[0])[5:19] - np.asarray(full_up[0])[5:19]))
                a = self._blend_alpha(ovl)
                full_up[:ovl] = tail[:ovl]   * (1 - a) + full_up[:ovl] * a
                hand_up[:ovl] = tail_h[:ovl] * (1 - a) + hand_up[:ovl] * a

        with self._ctrl_lock:
            self.primary_actions      = full_up
            self.primary_hand_actions = hand_up
            self.primary_index        = 0
            self._rtc_threshold       = max(2, int(len(full_up) * self.exec_frac))
            self._action_buf.clear(); self._hand_buf.clear()
        self.start_loop = True

        # 로깅 (보강: splice 정렬 + lag + stall margin). 처음 3회 + lag_log_every 마다.
        self._rtc_count += 1
        if self._rtc_count <= 3 or self._rtc_count % self.lag_log_every == 0:
            lag_up = int(lag_native * self.fast_hz / self.slow_hz)
            margin = len(full_up) - self._rtc_threshold
            print(
                f"[Deploy/RTC] #{self._rtc_count} lag={lag_ns/1e6:.0f}ms(native {lag_native}) "
                f"buf={len(full_up)} replan_at={self._rtc_threshold} margin={margin} lag_up={lag_up} "
                f"splice_L2={splice_l2:.4f}(0 에 가까울수록 정렬 정확)"
                + ("  ⚠️STALL위험" if margin <= lag_up else ""), flush=True
            )

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
        self.gr00t_inference(obs, obs_ts_ns)

    def do_fast(self):
        a, h = self.get_action()
        self.write_shm(a, h)

    # ----- loops -------------------------------------------------------------
    def _camera_poll_loop(self):
        """Phase B: 60Hz 로 모든 카메라 SHM 을 polling 해 frame ring buffer 를 채운다.

        polling 주기 = 카메라 publish 주기와 일치 (1/60s). dedup 은 ring 의
        push() 가 last_ts 기반으로 처리하므로 동일 frame 반복 read 해도 ring 에
        중복 안 들어감. inference thread 가 ring 의 1초 전 frame 을 ts 기반 pick.

        zed_mode: legacy_camera_shm 에서 stereo left/right 별도로 ring 에 push
                  (camera_roles = ['ego_left', 'ego_right']).
        멀티뷰:    self.camera_shms 의 각 SHM 에서 frame_left 만 push
                  (camera_roles = ['ego', 'wrist_l', 'wrist_r']).
        """
        period = 1.0 / 60.0
        next_t = time.perf_counter()
        while not self._stop_event.is_set() and not self.shared_event["shutdown"].is_set():
            try:
                if self.zed_mode:
                    if self.legacy_camera_shm is not None:
                        d = self.legacy_camera_shm.read_data()
                        ts = int(d.get('frame_ts', 0))
                        if ts > 0:
                            left = process_frame(d['frame_left'], side='zed_left')
                            if left is not None and 'ego_left' in self._cam_rings:
                                self._cam_rings['ego_left'].push(ts, left)
                            if self.binocular:
                                right = process_frame(d['frame_right'], side='zed_right')
                                if right is not None and 'ego_right' in self._cam_rings:
                                    self._cam_rings['ego_right'].push(ts, right)
                else:
                    for role, shm in self.camera_shms.items():
                        d = shm.read_data()
                        ts = int(d.get('frame_ts', 0))
                        if ts <= 0:
                            continue
                        rgb = process_frame(d['frame_left'], side=f'rs_{role}')
                        if rgb is None:
                            continue
                        if role in self._cam_rings:
                            self._cam_rings[role].push(ts, rgb)
            except Exception as e:
                logger_mp.warning(f"[Deploy] camera poll loop error: {e}")
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

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
                    if self.chunking_mode == "soft_rtc":
                        self.do_slow_rtc()
                    else:
                        self.do_slow()
                except Exception as e:
                    logger_mp.exception(f"[Deploy] slow loop error: {e}")
            else:
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
        self._cam_poll_thread.join(timeout=2.0)
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
        logger_mp.info("[Deploy] Gr00t_Inference stopped.")
