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
from utils.modality_layout import (
    build_state_layout, build_modality_json, concat_state_parts, layout_max_end,
)

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


# 공통 정렬 출력 주파수. 카메라 60fps + hand 100Hz + robot obs 300Hz 구성에서,
# 가장 낮은 연속 모달리티가 카메라(60Hz)가 되도록 hand 를 100Hz 로 올렸다(이전 50Hz).
# → 60Hz 저장 시 모든 모달리티가 업샘플 없이 자연스러운 다운샘플/ZOH 정렬된다.
# (hand 100Hz→60 다운, camera 60→60 그대로, robot 300→60 다운.) 최종 학습용
# 다운샘플(DP 10Hz, GR00T 15~30Hz)은 60 의 정수 약수(10/15/30/60)라 깔끔.
# 필요 시 worker_record 가 override 가능.
DEFAULT_OUTPUT_HZ = 60.0


# ============================================================================
# Phase M (PART3 §2): modality.json 동적 빌드 (정적 파일 복사 → 토글 기반 생성)
# ============================================================================

import os as _os
import json as _json
import shutil as _shutil


def _ensure_meta_modality(
    task_name: str,
    hand_type: str = 'inspire',
    waist_on:  bool = True,
    head_on:   bool = True,
    tactile_on: bool = False,
    tactile_dim: int = 12,
    camera_roles: Optional[List[str]] = None,
) -> None:
    """record/<task>/meta/modality.json 을 build_modality_json() 결과로 생성/갱신.

    이전 (Phase K6) 의 정적 파일 복사 (modality_{inspire,dex3}.json) 는 폐기. 토글
    조합에 따라 매번 동적으로 빌드. 이미 존재하면 layout 변경 시에만 갱신 +
    이전 파일 .bak 백업.
    """
    new_m = build_modality_json(
        hand_type=hand_type,
        waist_on=waist_on,
        head_on=head_on,
        tactile_on=tactile_on,
        tactile_dim=tactile_dim,
        camera_roles=camera_roles,
    )
    dst_dir = _os.path.join(BASE_FOLDER, task_name, 'meta')
    _os.makedirs(dst_dir, exist_ok=True)
    dst = _os.path.join(dst_dir, 'modality.json')
    if _os.path.exists(dst):
        try:
            with open(dst) as f:
                cur = _json.load(f)
            # state layout 동일하면 skip (사용자 수동 편집 보존).
            if cur.get('state') == new_m['state'] and cur.get('video') == new_m['video']:
                return
        except Exception:
            pass
        _shutil.copyfile(dst, dst + '.bak')
        logger_mp.info(f"[Record] modality.json 갱신 (이전 → .bak): {dst}")
    with open(dst, 'w') as f:
        _json.dump(new_m, f, indent=2)
    logger_mp.info(
        f"[Record] modality.json 빌드: {dst} "
        f"(hand={hand_type}, waist_on={waist_on}, head_on={head_on}, "
        f"tactile_on={tactile_on}, cameras={camera_roles})"
    )


class RecordCollectors:
    """Per-episode background pollers for streaming SHMs.

    Args:
        shm: dict with keys 'robot_obs', 'robot_action', 'television', 'controller'
            (필수) — pre-attached SharedMemoryManager handles owned by worker_record.
        camera_shms: dict {role: SharedMemoryManager} — role 별 카메라 SHM 핸들
            (Phase K7-A). 예: {'ego': shm, 'wrist_l': shm, 'wrist_r': shm}.
            빈 dict 면 카메라 없음 (--camera none).
    """

    def __init__(self, shm: Dict[str, object], camera_shms: Optional[Dict[str, object]] = None,
                 touch_shms: Optional[Dict[str, object]] = None):
        """Args:
            shm:         {'robot_obs','robot_action','television','controller'}
            camera_shms: {role: SharedMemoryManager} — Phase K7-A
            touch_shms:  {'left': SharedMemoryManager, 'right': SharedMemoryManager} or None.
                         Phase M6 — Inspire LEFT/RIGHT_TOUCH_SENSOR SHM (tactile_on 일 때만).
        """
        self.shm_obs   = shm['robot_obs']
        self.shm_act   = shm['robot_action']
        self.shm_tv    = shm['television']
        self.shm_ctrl  = shm['controller']
        self.camera_shms: Dict[str, object] = dict(camera_shms or {})
        self.touch_shms:  Dict[str, object] = dict(touch_shms or {})

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
        # tactile buffer (Phase M6, Inspire 먼저 — LEFT/RIGHT touch sensors).
        # DEX3 는 sequence length N 확정 후 같은 메커니즘에 차원만 채움.
        for side in self.touch_shms.keys():
            self.bufs[f'touch_{side}'] = RawStreamBuffer(f'touch_{side}', maxlen=200_000)

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
        if self.touch_shms:
            self._threads.append(threading.Thread(target=self._poll_touch, daemon=True, name='REC_TOUCH'))
        for t in self._threads:
            t.start()
        logger_mp.info(f"[RecordCollectors] started (robot/tv/camera{'/touch' if self.touch_shms else ''})")

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

    def _poll_touch(self) -> None:
        """Inspire LEFT/RIGHT_TOUCH_SENSOR SHM polling (Phase M6 / PART3 §3).

        worker_hand_dds 가 ~1kHz 로 write. 100Hz polling 으로 충분 (수집 정렬 축 50Hz).
        SHM read_data 가 모든 field copy 라 비용 큼 — ts 만 먼저 보는 최적화는 후속.
        """
        period = 1.0 / 100.0
        while not self._stop.is_set():
            for side, shm in self.touch_shms.items():
                try:
                    d = shm.read_data()
                except Exception:
                    continue
                ts_field = f'{side[0]}_touch_ts'  # 'l_touch_ts' or 'r_touch_ts'
                ts = int(d.get(ts_field, 0))
                # palm_touch 만 우선 저장 (~8x14=112 ints). 다른 finger field 도 필요 시
                # 추가. 학습 측에서 modality.json 의 sensor.tactile 차원에 맞춰 사용.
                key = f'{side[0]}_palm_touch'
                payload = {'palm': d[key].copy() if key in d else np.zeros((8, 14), dtype=np.int16)}
                self.bufs[f'touch_{side}'].append(ts, payload)
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
    waist_on:  bool = True,
    head_on:   bool = True,
    tactile_on: bool = False,
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

    # ---- Save parquet (Phase M / PART3 §2.3 동적 state_vec) --------------
    # 단일 진실 출처 = modality_layout.build_state_layout. modality.json 빌드 시
    # 사용하는 layout 과 정확히 동일한 함수를 여기서도 호출 → 차원 정합 자동.
    # hand_dim per side: dex3=7, inspire=6. SHM obs_hand 는 항상 14D 라 Inspire 의
    # 경우 [:6] / [6:12] 만 의미가 있고 [12:14] 는 zero pad.
    layout = build_state_layout(hand_type, waist_on=waist_on, head_on=head_on)
    hand_dim_side = 7 if hand_type == 'dex3' else 6

    # Phase M6 (PART3 §3): tactile_on 시 left/right palm touch 를 ZOH-pick 후 axis 에
    # 정렬. Inspire 의 LEFT/RIGHT_TOUCH_SENSOR.palm_touch (8,14) 만 우선 저장 (다른
    # finger field 는 후속). flatten 하여 observation.sensor 1-D 컬럼에 저장.
    # tactile_dim_per_side = 8 * 14 = 112. 양손 합 224.
    tactile_axis: Optional[list] = None  # k 별 (224,) np.float32 또는 None
    if tactile_on:
        tactile_axis = []
        ts_tl, p_tl = dumped.get('touch_left',  (np.empty(0, dtype=np.int64), []))
        ts_tr, p_tr = dumped.get('touch_right', (np.empty(0, dtype=np.int64), []))
        palm_l_picks = _zoh_pick(ts_tl, p_tl, 'palm', ts_axis) if ts_tl.size else [None] * n
        palm_r_picks = _zoh_pick(ts_tr, p_tr, 'palm', ts_axis) if ts_tr.size else [None] * n
        zero112 = np.zeros(112, dtype=np.float32)
        for k in range(n):
            pl = palm_l_picks[k]
            pr = palm_r_picks[k]
            l = np.asarray(pl, dtype=np.float32).reshape(-1) if pl is not None else zero112
            r = np.asarray(pr, dtype=np.float32).reshape(-1) if pr is not None else zero112
            tactile_axis.append(np.concatenate([l, r]))   # (224,)

    parquet_sink.start_episode(task_name, ep_idx)
    # observation.sensor 자리표 — tactile_on 일 때만 실 데이터. off 면 zero (legacy 동일).
    sensor_zero = np.zeros(12, dtype=np.float32)
    for k in range(n):
        parts_obs = {
            'left_arm':  obs_arm[k, :7],
            'right_arm': obs_arm[k, 7:14],
            'left_hand':  obs_hand[k, :hand_dim_side],
            'right_hand': obs_hand[k, hand_dim_side:2 * hand_dim_side],
        }
        parts_act = {
            'left_arm':  act_arm[k, :7],
            'right_arm': act_arm[k, 7:14],
            'left_hand':  act_hand[k, :hand_dim_side],
            'right_hand': act_hand[k, hand_dim_side:2 * hand_dim_side],
        }
        if waist_on:
            parts_obs['waist'] = obs_waist[k]
            parts_act['waist'] = act_waist[k]
        if head_on:
            parts_obs['head'] = obs_head[k]
            parts_act['head'] = act_head[k]
        state_vec  = concat_state_parts(parts_obs, layout)
        action_vec = concat_state_parts(parts_act, layout)
        sensor_k = tactile_axis[k] if tactile_axis is not None else sensor_zero
        parquet_sink.append(state_vec, sensor_k, action_vec, t_sec=float(t_sec_axis[k]))

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

    # Phase M (PART3 §2.2): modality.json 동적 빌드 — 토글 조합 반영.
    # tactile_dim 은 Inspire palm 양손 합 224 (8x14x2). DEX3 는 sequence length N
    # 확정 후 12*N 로 갱신 (REMAINING §L8).
    tactile_dim = 224 if (tactile_on and hand_type == 'inspire') else 12
    try:
        _ensure_meta_modality(
            task_name=task_name,
            hand_type=hand_type,
            waist_on=waist_on,
            head_on=head_on,
            tactile_on=tactile_on,
            tactile_dim=tactile_dim,
            camera_roles=camera_roles,
        )
    except Exception as e:
        logger_mp.warning(f"[Record] modality.json 빌드 실패: {e}")

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
