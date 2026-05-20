"""Record-side collectors + alignment + save helpers (Phase D).

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

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


# 공통 정렬 출력 주파수. 학습 측 modality 가 기대하는 control rate 와 일치시킴
# (G1 ctrl 50Hz 와 동일). 필요 시 worker_record 가 override 가능.
DEFAULT_OUTPUT_HZ = 50.0


class RecordCollectors:
    """Per-episode background pollers for streaming SHMs.

    Args:
        shm: dict with keys 'robot_obs', 'robot_action', 'television',
            'controller', 'camera' — pre-attached SharedMemoryManager
            handles owned by worker_record.
        use_zed / use_realsense: which camera substreams to collect.
            (worker_record already knows from teleop_config_shm.)
    """

    def __init__(self, shm: Dict[str, object], use_zed: bool, use_realsense: bool):
        self.shm_obs   = shm['robot_obs']
        self.shm_act   = shm['robot_action']
        self.shm_tv    = shm['television']
        self.shm_ctrl  = shm['controller']
        self.shm_cam   = shm['camera']
        self.use_zed       = bool(use_zed)
        self.use_realsense = bool(use_realsense)

        self.bufs: Dict[str, RawStreamBuffer] = {
            'obs_body':    RawStreamBuffer('obs_body',    maxlen=300_000),
            'obs_hand':    RawStreamBuffer('obs_hand',    maxlen=100_000),
            'action_body': RawStreamBuffer('action_body', maxlen=100_000),
            'action_hand': RawStreamBuffer('action_hand', maxlen=100_000),
            'television':  RawStreamBuffer('television',  maxlen=50_000),
            'controller':  RawStreamBuffer('controller',  maxlen=50_000),
            'camera_zed':  RawStreamBuffer('camera_zed',  maxlen=10_000),
            'camera_rs':   RawStreamBuffer('camera_rs',   maxlen=10_000),
        }
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
        # ZED ~30Hz, RealSense ~30Hz. 100Hz poll, copy big frames only on ts change.
        period = 1.0 / 100.0
        while not self._stop.is_set():
            try:
                cam = self.shm_cam.read_data()
            except Exception:
                time.sleep(period); continue
            if self.use_zed:
                ts_z = int(cam['camera_zed_ts'])
                if ts_z > 0 and (self.bufs['camera_zed']._last_ts is None
                                 or ts_z > self.bufs['camera_zed']._last_ts):
                    self.bufs['camera_zed'].append(ts_z, {
                        'left':  cam['camera_left'].copy(),
                        'right': cam['camera_right'].copy(),
                    })
            if self.use_realsense:
                ts_r = int(cam['camera_realsense_ts'])
                if ts_r > 0 and (self.bufs['camera_rs']._last_ts is None
                                 or ts_r > self.bufs['camera_rs']._last_ts):
                    self.bufs['camera_rs'].append(ts_r, {
                        'frame': cam['realsense'].copy(),
                    })
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
    use_zed: bool,
    use_realsense: bool,
    output_hz: float = DEFAULT_OUTPUT_HZ,
) -> bool:
    """Build common axis, interpolate every stream onto it, save parquet+mp4.

    Returns True if save succeeded (>=2 axis samples), False if no usable data.
    """
    # 공통 시간축 — body obs / hand obs / camera 가 모두 존재할 때 그들 intersection.
    # 이중 하나라도 없으면 그 stream 의 ts 는 제외하고 나머지로 축 구성.
    streams_ts = [
        dumped['obs_body'][0],
        dumped['obs_hand'][0],
        dumped['action_body'][0],
        dumped['action_hand'][0],
    ]
    if use_zed and dumped['camera_zed'][0].size > 0:
        streams_ts.append(dumped['camera_zed'][0])
    if use_realsense and dumped['camera_rs'][0].size > 0:
        streams_ts.append(dumped['camera_rs'][0])
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

    # ---- ZOH-pick image streams ------------------------------------------
    ts_cz, p_cz = dumped['camera_zed']
    ts_cr, p_cr = dumped['camera_rs']
    zed_left_frames  = _zoh_pick(ts_cz, p_cz, 'left',  ts_axis) if use_zed       else [None]*n
    zed_right_frames = _zoh_pick(ts_cz, p_cz, 'right', ts_axis) if use_zed       else [None]*n
    rs_frames        = _zoh_pick(ts_cr, p_cr, 'frame', ts_axis) if use_realsense else [None]*n

    # ---- Save parquet (LeRobot v2.1 호환 구조 유지) ----------------------
    parquet_sink.start_episode(task_name, ep_idx)
    sensor_zero = np.zeros(12, dtype=np.float32)   # 자리표 (touch 도입 시 확장)
    for k in range(n):
        state_vec  = np.concatenate([obs_waist[k], obs_head[k], obs_arm[k], obs_hand[k]])
        action_vec = np.concatenate([act_waist[k], act_head[k], act_arm[k], act_hand[k]])
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
    if use_zed:       parquet_sink.add_extra_column('raw_ts_camera_zed', _src_ts(ts_cz))
    if use_realsense: parquet_sink.add_extra_column('raw_ts_camera_rs',  _src_ts(ts_cr))
    parquet_sink.close_episode()

    # ---- Save mp4 --------------------------------------------------------
    video_sink.start_episode(task_name, ep_idx)
    for k in range(n):
        video_sink.append(zed_left_frames[k], zed_right_frames[k], rs_frames[k])
    video_sink.close_episode()

    return True
