"""Record-side collectors + alignment + save helpers (Phase D / K6 확장).

`RecordCollectors` spawns one background thread per SHM and pushes
(ts, payload) tuples into per-substream RawStreamBuffer. On
`stop_and_dump()` returns the raw streams. `save_aligned_episode` then
builds a uniform-rate common axis, interpolates each stream onto it
(linear for continuous signals, ZOH for image frames), and writes the
result through ParquetSink + VideoSink — preserving the LeRobot v2.1
file layout.

Why this exists: G1 obs writes at ~300 Hz, hand at ~50 Hz, camera at
~30 Hz, VR/controller at ~50 Hz, all in independent worker processes.
Per-cycle snapshot recording (the old worker_record path) silently
admitted up to one full period of phase offset between streams in
every parquet row. With the collectors and post-align, every parquet
row corresponds to a single time-axis sample t_k that is the cosine /
ZOH interpolation of the *actual* source samples bracketing t_k. The
raw source timestamps for each stream are also written to parquet as
metadata columns for downstream phase analysis.
"""
from __future__ import annotations
import threading
import time
from typing import Dict, Optional, Tuple, List

import numpy as np

from utils.raw_stream import RawStreamBuffer
from utils.align import interp_to_axis, common_time_axis
from utils.record_config import BASE_FOLDER

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


# 공통 정렬 출력 주파수. 학습 측 modality 가 기대하는 control rate 와 일치시킴
# (G1 ctrl 50Hz 와 동일). 필요 시 worker_record 가 override 가능.
DEFAULT_OUTPUT_HZ = 50.0


# ============================================================================
# Phase K6 (P1-8): modality.json hand 종류별 자동 배치
# ============================================================================

import os as _os
import shutil as _shutil

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_MODALITY_SRC = {
    'inspire': _os.path.join(_REPO_ROOT, 'utils', 'parquet', 'modality_inspire.json'),
    'dex3':    _os.path.join(_REPO_ROOT, 'utils', 'parquet', 'modality_dex3.json'),
}

def _ensure_meta_modality(task_name: str, hand_type: str) -> None:
    """record/<task>/meta/modality.json 에 hand 종류별 modality 파일 복사.
    이미 존재하면 hand_type 변경 시에만 갱신 (사용자 수동 편집 보존)."""
    src = _MODALITY_SRC.get(hand_type)
    if src is None or not _os.path.exists(src):
        logger_mp.warning(f"[Record] modality src not found for hand_type={hand_type}: {src}")
        return
    dst_dir = _os.path.join(BASE_FOLDER, task_name, 'meta')
    _os.makedirs(dst_dir, exist_ok=True)
    dst = _os.path.join(dst_dir, 'modality.json')
    if _os.path.exists(dst):
        # 같은 hand_type 이면 skip. 다르면 백업 후 갱신.
        try:
            with open(dst) as f:
                import json as _json
                cur = _json.load(f)
            cur_dim = int(cur.get('state', {}).get('right_hand', {}).get('end', 0))
            new_dim = 33 if hand_type == 'dex3' else 31
            if cur_dim == new_dim:
                return  # 이미 정확
        except Exception:
            pass
        _shutil.copyfile(dst, dst + '.bak')
        logger_mp.info(f"[Record] modality.json 갱신 (이전 → .bak): hand_type={hand_type}")
    _shutil.copyfile(src, dst)
    logger_mp.info(f"[Record] modality.json 배치: {dst} (hand_type={hand_type})")


class RecordCollectors:
    """Per-episode background pollers for streaming SHMs.

    Args:
        shm: dict with keys 'robot_obs', 'robot_action', 'television', 'controller'
            (필수) — pre-attached SharedMemoryManager handles owned by worker_record.
        camera_shms: dict {role: SharedMemoryManager} — role 별 카메라 SHM 핸들
            (Phase K7-A). 예: {'ego': shm, 'wrist_l': shm, 'wrist_r': shm}.
            빈 dict 면 카메라 없음 (--camera none).
    """

    def __init__(self, shm: Dict[str, object], camera_shms: Optional[Dict[str, object]] = None):
        self.shm_obs   = shm['robot_obs']
        self.shm_act   = shm['robot_action']
        self.shm_tv    = shm['television']
        self.shm_ctrl  = shm['controller']
        self.camera_shms: Dict[str, object] = dict(camera_shms or {})

        self.bufs: Dict[str, RawStreamBuffer] = {
            'obs_body':    RawStreamBuffer('obs_body',    maxlen=300_000),
            'obs_hand':    RawStreamBuffer('obs_hand',    maxlen=100_000),
            'action_body': RawStreamBuffer('action_body', maxlen=100_000),
            'action_hand': RawStreamBuffer('action_hand', maxlen=100_000),
            'television':  RawStreamBuffer('television',  maxlen=50_000),
            'controller':  RawStreamBuffer('controller',  maxlen=50_000),
        }
        # role 별 camera buffer 동적 생성 (Phase K7-A)
        for role in self.camera_shms.keys():
            self.bufs[f'camera_{role}'] = RawStreamBuffer(f'camera_{role}', maxlen=10_000)

        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._threads:
            raise RuntimeError("RecordCollectors already started")
        self._stop.clear()
        for b in self.bufs.values():
            b.clear()
        self._threads = [
            threading.Thread(target=self._poll_robot,      daemon=True, name='REC_ROBOT'),
            threading.Thread(target=self._poll_television, daemon=True, name='REC_TV'),
            threading.Thread(target=self._poll_camera,     daemon=True, name='REC_CAM'),
        ]
        for t in self._threads:
            t.start()
        logger_mp.info("[RecordCollectors] started (robot/tv/camera pollers)")

    def stop_and_dump(self) -> Dict[str, Tuple[np.ndarray, list]]:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        dumped = {name: buf.dump() for name, buf in self.bufs.items()}
        for name, (ts, payloads) in dumped.items():
            logger_mp.info(f"[RecordCollectors] {name:14s} samples={len(payloads)}")
        return dumped

    # --- pollers -----------------------------------------------------------
    def _poll_robot(self) -> None:
        # ROBOT_OBS (body 300Hz, hand 50Hz) + ROBOT_ACTION (50Hz). 1kHz poll.
        period = 1.0 / 1000.0
        while not self._stop.is_set():
            try:
                obs = self.shm_obs.read_data()
                act = self.shm_act.read_data()
            except Exception:
                time.sleep(period); continue
            ts_ob = int(obs['obs_body_ts'])
            self.bufs['obs_body'].append(ts_ob, {
                'waist': obs['obs_waist'].copy(),
                'head':  obs['obs_head'].copy(),
                'arm':   obs['obs_arm'].copy(),
            })
            ts_oh = int(obs['obs_hand_ts'])
            self.bufs['obs_hand'].append(ts_oh, {'hand': obs['obs_hand'].copy()})
            ts_ab = int(act['action_body_ts'])
            self.bufs['action_body'].append(ts_ab, {
                'waist': act['action_waist'].copy(),
                'head':  act['action_head'].copy(),
                'arm':   act['action_arm'].copy(),
            })
            ts_ah = int(act['action_hand_ts'])
            self.bufs['action_hand'].append(ts_ah, {'hand': act['action_hand'].copy()})
            time.sleep(period)

    def _poll_television(self) -> None:
        # TELEVISION + QUEST_CONTROLLER ~50Hz. 200Hz poll.
        period = 1.0 / 200.0
        while not self._stop.is_set():
            try:
                tv = self.shm_tv.read_data()
                ct = self.shm_ctrl.read_data()
            except Exception:
                time.sleep(period); continue
            self.bufs['television'].append(int(tv['television_ts']), {
                'head_rmat':       tv['head_rmat'].copy(),
                'left_wrist_mat':  tv['left_wrist_mat'].copy(),
                'right_wrist_mat': tv['right_wrist_mat'].copy(),
            })
            self.bufs['controller'].append(int(ct['controller_ts']), {
                'left_trigger':  float(ct['left_trigger']),
                'left_squeeze':  float(ct['left_squeeze']),
                'left_buttons':  ct['left_buttons'].copy(),
                'right_trigger': float(ct['right_trigger']),
                'right_squeeze': float(ct['right_squeeze']),
                'right_buttons': ct['right_buttons'].copy(),
            })
            time.sleep(period)

    def _poll_camera(self) -> None:
        # Phase K7-A: 카메라별 독립 SHM (CAMERA_VIEW schema). role 별로 폴링.
        # SHM 분리로 read_data() 가 한 카메라 분량만 copy → 멀티 카메라에서도 cost 낮음.
        # RawStreamBuffer.append 가 ts<=last_ts dedup 처리.
        period = 1.0 / 100.0
        while not self._stop.is_set():
            for role, shm in self.camera_shms.items():
                try:
                    d = shm.read_data()
                except Exception:
                    continue
                ts = int(d['frame_ts'])
                is_stereo = bool(int(d.get('is_stereo', 0)))
                payload = {'left': d['frame_left'].copy()}
                if is_stereo:
                    payload['right'] = d['frame_right'].copy()
                self.bufs[f'camera_{role}'].append(ts, payload)
            time.sleep(period)


# ============================================================================
# Alignment + save helpers
# ============================================================================

def _stack(payloads: list, field: str, default_shape: tuple, dtype=np.float64) -> np.ndarray:
    """Stack a single field across payloads → (N, *shape). Empty → (0, *shape)."""
    if not payloads:
        return np.zeros((0,) + default_shape, dtype=dtype)
    return np.stack([np.asarray(p[field], dtype=dtype) for p in payloads], axis=0)


def _zoh_pick(ts_src: np.ndarray, payloads: list, field: str, ts_dst: np.ndarray):
    """ZOH-pick payloads[i][field] for each ts_dst, given ts_src. Returns list."""
    if ts_src.size == 0:
        return [None] * ts_dst.size
    idx = np.searchsorted(ts_src, ts_dst, side='right') - 1
    idx = np.clip(idx, 0, ts_src.size - 1)
    return [payloads[i][field] for i in idx]


def align_and_save_episode(
    dumped: Dict[str, Tuple[np.ndarray, list]],
    parquet_sink,
    video_sink,
    task_name: str,
    ep_idx: int,
    output_hz: float = DEFAULT_OUTPUT_HZ,
    hand_type: str = 'inspire',
    camera_roles: Optional[list] = None,
) -> bool:
    """Build common axis, interpolate every stream onto it, save parquet+mp4.

    Returns True if save succeeded (>=2 axis samples), False if no usable data.
    """
    # 공통 시간축 — body obs / hand obs / camera 가 모두 존재할 때 그들 intersection.
    # 이중 하나라도 없으면 그 stream 의 ts 는 제외하고 나머지로 축 구성.
    camera_roles = list(camera_roles or [])
    streams_ts = [
        dumped['obs_body'][0],
        dumped['obs_hand'][0],
        dumped['action_body'][0],
        dumped['action_hand'][0],
    ]
    for role in camera_roles:
        key = f'camera_{role}'
        if key in dumped and dumped[key][0].size > 0:
            streams_ts.append(dumped[key][0])
    ts_axis = common_time_axis(streams_ts, rate_hz=output_hz)
    if ts_axis.size < 2:
        logger_mp.warning(f"[Record] common axis too small ({ts_axis.size}); skip save.")
        return False
    n = ts_axis.size
    t_sec_axis = (ts_axis - ts_axis[0]).astype(np.float64) * 1e-9

    # ---- linear-interp continuous streams --------------------------------
    ts_ob, p_ob = dumped['obs_body']
    obs_waist = interp_to_axis(ts_ob, _stack(p_ob, 'waist', (3,)),  ts_axis, 'linear') if ts_ob.size else np.zeros((n, 3))
    obs_head  = interp_to_axis(ts_ob, _stack(p_ob, 'head',  (2,)),  ts_axis, 'linear') if ts_ob.size else np.zeros((n, 2))
    obs_arm   = interp_to_axis(ts_ob, _stack(p_ob, 'arm',   (14,)), ts_axis, 'linear') if ts_ob.size else np.zeros((n, 14))

    ts_oh, p_oh = dumped['obs_hand']
    obs_hand = interp_to_axis(ts_oh, _stack(p_oh, 'hand', (14,)), ts_axis, 'linear') if ts_oh.size else np.zeros((n, 14))

    # Phase K5 (P0-2): action 계열은 piecewise-constant (publish 시점에 고정된 명령) →
    # ZOH 보간이 의미론적으로 맞다. linear 로 보간하면 실제 보낸 적 없는 중간값이
    # 생성된다 (특히 DEX3 trigger toggle 같은 이산적 명령).
    ts_ab, p_ab = dumped['action_body']
    act_waist = interp_to_axis(ts_ab, _stack(p_ab, 'waist', (3,)),  ts_axis, 'zoh') if ts_ab.size else np.zeros((n, 3))
    act_head  = interp_to_axis(ts_ab, _stack(p_ab, 'head',  (2,)),  ts_axis, 'zoh') if ts_ab.size else np.zeros((n, 2))
    act_arm   = interp_to_axis(ts_ab, _stack(p_ab, 'arm',   (14,)), ts_axis, 'zoh') if ts_ab.size else np.zeros((n, 14))

    ts_ah, p_ah = dumped['action_hand']
    act_hand = interp_to_axis(ts_ah, _stack(p_ah, 'hand', (14,)), ts_axis, 'zoh') if ts_ah.size else np.zeros((n, 14))

    # ---- ZOH-pick image streams (Phase K7-A: role 기반 dict) ------------
    # role 별 frame 시퀀스 (stereo 카메라는 'right' 도 함께). VideoSink 가 키별 view 로 저장.
    camera_frames: Dict[str, list] = {}    # role → list of left frames (len=n)
    camera_right_frames: Dict[str, list] = {}  # stereo 일 때만
    for role in camera_roles:
        key = f'camera_{role}'
        if key not in dumped or dumped[key][0].size == 0:
            camera_frames[role] = [None] * n
            continue
        ts_c, p_c = dumped[key]
        camera_frames[role] = _zoh_pick(ts_c, p_c, 'left', ts_axis)
        if p_c and 'right' in p_c[0]:
            camera_right_frames[role] = _zoh_pick(ts_c, p_c, 'right', ts_axis)

    # ---- Save parquet (LeRobot v2.1 호환 구조 유지) ----------------------
    # Phase K6 (P1-8): hand 종류에 따라 hand DOF 를 12 (Inspire) 또는 14 (DEX3) 로
    # truncate. modality.json 의 left_hand/right_hand 범위 (Inspire=6+6 / DEX3=7+7)
    # 와 정확히 일치하도록 저장. SHM 의 obs_hand/action_hand 는 항상 14D 지만,
    # Inspire 의 경우 [:12] 만 의미가 있고 나머지 [12:14] 는 0 (worker_hand_ctrl 의
    # zero-pad). 저장 시 hand_type 따라 분기.
    if hand_type == 'dex3':
        hand_dim = 14
    else:  # inspire (default + backward-compat)
        hand_dim = 12
    parquet_sink.start_episode(task_name, ep_idx)
    sensor_zero = np.zeros(12, dtype=np.float32)   # 자리표 (촉각 toggle 시 확장)
    for k in range(n):
        state_vec  = np.concatenate([obs_waist[k], obs_head[k], obs_arm[k],
                                     obs_hand[k, :hand_dim]])
        action_vec = np.concatenate([act_waist[k], act_head[k], act_arm[k],
                                     act_hand[k, :hand_dim]])
        parquet_sink.append(state_vec, sensor_zero, action_vec, t_sec=float(t_sec_axis[k]))

    # raw source ts (보간 시 사용된 직전 sample 의 ts)
    def _src_ts(ts_src):
        if ts_src.size == 0:
            return np.zeros(n, dtype=np.int64)
        idx = np.searchsorted(ts_src, ts_axis, side='right') - 1
        idx = np.clip(idx, 0, ts_src.size - 1)
        return ts_src[idx]

    parquet_sink.add_extra_column('axis_ts_ns',          ts_axis)
    parquet_sink.add_extra_column('raw_ts_obs_body',     _src_ts(ts_ob))
    parquet_sink.add_extra_column('raw_ts_obs_hand',     _src_ts(ts_oh))
    parquet_sink.add_extra_column('raw_ts_action_body',  _src_ts(ts_ab))
    parquet_sink.add_extra_column('raw_ts_action_hand',  _src_ts(ts_ah))
    for role in camera_roles:
        key = f'camera_{role}'
        if key in dumped:
            parquet_sink.add_extra_column(f'raw_ts_camera_{role}', _src_ts(dumped[key][0]))
    parquet_sink.close_episode()

    # Phase K6 (P1-8): record/<task>/meta/modality.json 자동 배치 — hand 종류별 분기.
    # 학습 측이 task-level 메타로 읽도록.
    try:
        _ensure_meta_modality(task_name, hand_type)
    except Exception as e:
        logger_mp.warning(f"[Record] modality.json 배치 실패: {e}")

    # ---- Save mp4 (Phase K7-A: role 별 view 동적 저장) ------------------
    # VideoSink.start_episode 가 active view 명을 받아 동적으로 mp4 파일 생성.
    # 단일 카메라 = ego 1 view 만 활성, 멀티 카메라 = ego + wrist_l + wrist_r.
    active_views = []  # ['observation.images.ego', ...]
    view_frames: Dict[str, list] = {}
    for role in camera_roles:
        # LeRobot 호환 키: observation.images.<role>. modality.json 의 video.<role>.original_key 와 일치.
        active_views.append(f'observation.images.{role}')
        view_frames[f'observation.images.{role}'] = camera_frames[role]
        if role in camera_right_frames:
            # stereo 의 right 도 별도 view 로 저장 (예: ego.right).
            active_views.append(f'observation.images.{role}_right')
            view_frames[f'observation.images.{role}_right'] = camera_right_frames[role]

    video_sink.start_episode(task_name, ep_idx, views=active_views)
    for k in range(n):
        view_payload = {v: view_frames[v][k] for v in active_views}
        video_sink.append_views(view_payload)
    video_sink.close_episode()

    return True
