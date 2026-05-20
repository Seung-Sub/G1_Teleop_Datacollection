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
)

import logging_mp
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


def upsample_actions(seq: np.ndarray, slow_hz: float = 20.0, fast_hz: float = 50.0,
                     k: int = 5) -> np.ndarray:
    """Spline-upsample a (T, D) action chunk from slow_hz to fast_hz.

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

def _split_qpos(qpos: np.ndarray):
    """qpos = concat(obs_waist[3], obs_head[2], obs_arm[14]) = 19-D.

    Phase L1 (Part 2 §11 head 일관성 검증): 현재 modality_{inspire,dex3}.json 에
    state.head 가 **없다** (state 는 waist + left_arm + right_arm + left/right_hand
    만). 수집 측은 align_and_save_episode 가 obs_head 를 state_vec 에 포함하므로
    실제로는 31D/33D 가 아니라 33D/35D 가 저장된다 (3+2+14+12 or 14).
    → 후속 작업 (REMAINING_VERIFICATION_TASKS.md L6-A 참고): modality.json 에
       state.head 추가 + 학습/배포에서 head 사용 여부 통일 필요.
    현재는 deploy 측이 head 를 버려 modality.json 과 정합되도록 _split_qpos 유지.
    """
    waist     = qpos[:3].astype(np.float32)
    # qpos[3:5] is head; modality.json 에 state.head 없어 의도적으로 버림.
    left_arm  = qpos[5:12].astype(np.float32)
    right_arm = qpos[12:19].astype(np.float32)
    return waist, left_arm, right_arm


def _split_hand(hand_qpos: np.ndarray, hand_type: str):
    """obs_hand (14D, Inspire 는 뒤 2개 0) → left_hand / right_hand 분리.

    hand_type='inspire': 6+6=12D (앞 12개 사용)
    hand_type='dex3'   : 7+7=14D (전체 14개 사용)
    """
    if hand_type == 'dex3':
        return hand_qpos[:7].astype(np.float32), hand_qpos[7:14].astype(np.float32)
    # inspire
    return hand_qpos[:6].astype(np.float32), hand_qpos[6:12].astype(np.float32)


def obs_dict_multi_rs(task_name: str, qpos: np.ndarray, hand_qpos: np.ndarray,
                      frames: Dict[str, np.ndarray], hand_type: str) -> dict:
    """RealSense 멀티뷰 observation dict.

    frames: {'ego': rgb_np, 'wrist_l': ..., 'wrist_r': ...}  — 활성 role 만 key 로.
    GR00T DataConfig 의 video_keys 가 'video.ego', 'video.wrist_l', 'video.wrist_r'
    (modality_{inspire,dex3}.json 의 video.<role> 와 동명) 에 맞춰진다고 가정.
    DataConfig 측에서 새 키를 등록해야 함 (REMAINING_VERIFICATION_TASKS.md L1-B).
    """
    waist, left_arm, right_arm = _split_qpos(qpos)
    left_hand, right_hand = _split_hand(hand_qpos, hand_type)
    d = {
        "state.waist":      waist[None, None, :],
        "state.left_arm":   left_arm[None, None, :],
        "state.right_arm":  right_arm[None, None, :],
        "state.left_hand":  left_hand[None, None, :],
        "state.right_hand": right_hand[None, None, :],
        "annotation.human.action.task_description": np.array([task_name], dtype=object),
    }
    for role, rgb in frames.items():
        d[f"video.{role}"] = rgb[None, None, ...]
    return d


def obs_dict_zed(task_name: str, qpos: np.ndarray, hand_qpos: np.ndarray,
                 rgb_left: np.ndarray, rgb_right: Optional[np.ndarray],
                 binocular: bool, hand_type: str) -> dict:
    """ZED stereo (binocular) 또는 single-view obs dict. Legacy 호환."""
    waist, left_arm, right_arm = _split_qpos(qpos)
    left_hand, right_hand = _split_hand(hand_qpos, hand_type)
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


def action_to_array(action: dict, i: int, hand_type: str) -> tuple:
    """Pull one timestep out of a GR00T action chunk.

    Phase L1: hand_type 으로 hand DOF 분기 (6+6 or 7+7).
    GR00T action keys:
        action.waist     (1, T, 3)
        action.left_arm  (1, T, 7)   # 6 일 수도 — write_shm 의 fallback
        action.right_arm (1, T, 7)
        action.left_hand (1, T, 6 or 7)
        action.right_hand(1, T, 6 or 7)
    Returns (action_np[19], hand_action_np[12 or 14]) for write_shm.
    """
    waist     = np.asarray(action["action.waist"][0, i],     dtype=np.float32)  # (3,)
    left_arm  = np.asarray(action["action.left_arm"][0, i],  dtype=np.float32)  # (6 or 7,)
    right_arm = np.asarray(action["action.right_arm"][0, i], dtype=np.float32)  # (7,)
    left_h    = np.asarray(action["action.left_hand"][0, i], dtype=np.float32)  # (6 or 7)
    right_h   = np.asarray(action["action.right_hand"][0,i], dtype=np.float32)  # (6 or 7)
    head      = np.zeros(2, dtype=np.float32)                                   # head not commanded

    action_np   = np.concatenate([waist, head, left_arm, right_arm], axis=0)    # 3+2+(6|7)+7
    hand_action = np.concatenate([left_h, right_h], axis=0)                     # 12 or 14
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

    if data_config_key not in DATA_CONFIG_MAP:
        keys = list(DATA_CONFIG_MAP.keys())
        raise KeyError(
            f"data_config_key={data_config_key!r} 가 DATA_CONFIG_MAP 에 없음. "
            f"사용 가능 키: {keys}. DEX3 + RS3뷰 운용 시 DataConfig 신규 등록 필요 "
            f"(REMAINING_VERIFICATION_TASKS.md L1-B 참고)."
        )
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
    """GR00T inference worker. hand_type + camera roles 자동 분기 (Phase L1).

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
        model_path: str = "",
        data_config_key: str = "unitree_g1",
        embodiment_tag: str = "new_embodiment",
        action_method: str = "tem",             # {"base", "maf", "tem"}
        decay: float = 0.3,
        window_size: int = 5,
        slow_hz: float = 20.0,
        fast_hz: float = 50.0,
        denoising_steps: int = 4,
        binocular: bool = True,
        masking: bool = False,
        lag_compensate: bool = True,
        lag_log_every: int = 50,
        obs_ts_policy: str = 'min',             # Phase L2: 'min' (stale 기준) | 'max'
    ):
        assert mode in ("gr00t_rs_multi", "gr00t_zed", "gr00t"), f"unsupported mode: {mode}"
        # 'gr00t' (legacy single RealSense) 는 alias for gr00t_rs_multi (ego only).
        if mode == "gr00t":
            mode = "gr00t_rs_multi"
        assert action_method in ("base", "maf", "tem"), f"unknown method: {action_method}"
        assert obs_ts_policy in ("min", "max"), f"unknown obs_ts_policy: {obs_ts_policy}"

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
        self.lag_compensate  = bool(lag_compensate)
        self.lag_log_every   = max(1, int(lag_log_every))
        self.obs_ts_policy   = obs_ts_policy
        self._lag_chunk_count = 0
        self._lag_ns_acc      = 0
        self._lag_ns_max      = 0

        # SHM handles + hand_type / active camera roles 결정
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
            f"[Deploy] Gr00t_Inference started (mode={mode}, hand={self.hand_type}, "
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
        """teleop_config_shm 의 hand_type 을 읽어 자동 분기. main.py 가 owner-create
        한 SHM 이므로 즉시 valid."""
        self.hand_type = 'inspire'
        try:
            cfg = self.teleop_config_shm.read_data()
            self.hand_type = HAND_MAPPING_INV.get(int(cfg["hand_type"]), 'inspire')
        except Exception as e:
            logger_mp.warning(f"[Deploy] teleop_config read 실패: {e} — hand_type=inspire fallback")
        self.hand_dim = 14 if self.hand_type == 'dex3' else 12

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
            model_path=self.model_path, data_config_key=self.data_config_key,
            embodiment_tag=self.embodiment_tag, denoising_steps=self.denoising_steps,
        )
        logger_mp.info(f"[Deploy] Policy loaded. data_config_key={self.data_config_key} task={task_name!r}")

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

        # 카메라 frame + ts read. Phase L2: ts 후보에 카메라 ts 도 포함.
        ts_candidates = [t for t in (ts_body, ts_hand) if t > 0]

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
            obs = obs_dict_zed(self.task_name, qpos, hand_qpos, left, right,
                               binocular=self.binocular, hand_type=self.hand_type)
        else:
            # RealSense 멀티뷰 (Phase L1).
            if not self.camera_shms:
                return None
            frames: Dict[str, np.ndarray] = {}
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
            obs = obs_dict_multi_rs(self.task_name, qpos, hand_qpos, frames, self.hand_type)

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
        action = self.policy.get_action(obs)
        t_after_ns = time.perf_counter_ns()

        T = action["action.waist"].shape[1]
        full = []
        hand = []
        for i in range(T):
            a_np, h_np = action_to_array(action, i, self.hand_type)
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
        self.gr00t_inference(obs, obs_ts_ns)

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
