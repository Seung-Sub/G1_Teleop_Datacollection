"""Time alignment helper for multi-stream recording.

interp_to_axis(ts_src, val_src, ts_dst, mode) — episode close 시 worker_record
가 각 stream 을 공통 시간축으로 정렬할 때 사용.

mode='linear': numpy.interp 기반, per-column 선형 보간. 연속 신호 (qpos, ee
pose, controller analog 값) 에 사용.
mode='zoh' (zero-order hold): np.searchsorted 로 직전 sample 인덱스 선택.
discrete event (button bool, mode flag, image frame) 에 사용.
"""
from __future__ import annotations
from typing import Literal

import numpy as np


def interp_to_axis(
    ts_src: np.ndarray,
    val_src: np.ndarray,
    ts_dst: np.ndarray,
    mode: Literal['linear', 'zoh'] = 'linear',
) -> np.ndarray:
    """Interpolate val_src (defined at ts_src) onto ts_dst.

    Args:
        ts_src: (N,) int64 monotonic ns timestamps. Must be sorted ascending.
        val_src: (N,) or (N, D...) sample values. dtype preserved.
        ts_dst: (M,) int64 target timestamps. Need not be sorted (sortidx
            preserved on output).
        mode: 'linear' or 'zoh'.

    Returns:
        (M,) or (M, D...) array of interpolated values.
    """
    n = ts_src.size
    if n == 0:
        return np.zeros((ts_dst.size,) + val_src.shape[1:], dtype=val_src.dtype)
    if n == 1 or mode == 'zoh':
        # ZOH: idx = max i s.t. ts_src[i] <= ts_dst
        # searchsorted with side='right' gives upper bound; -1 → floor
        idx = np.searchsorted(ts_src, ts_dst, side='right') - 1
        idx = np.clip(idx, 0, n - 1)
        return val_src[idx]
    if mode == 'linear':
        # Convert ns → seconds float64 for numerical stability of np.interp
        x_src = ts_src.astype(np.float64) * 1e-9
        x_dst = ts_dst.astype(np.float64) * 1e-9
        if val_src.ndim == 1:
            out = np.interp(x_dst, x_src, val_src.astype(np.float64))
            return out.astype(val_src.dtype, copy=False)
        # multi-dim: flatten trailing dims then interp per-column
        flat = val_src.reshape(n, -1).astype(np.float64)
        D    = flat.shape[1]
        out_flat = np.empty((ts_dst.size, D), dtype=np.float64)
        for d in range(D):
            out_flat[:, d] = np.interp(x_dst, x_src, flat[:, d])
        out = out_flat.reshape((ts_dst.size,) + val_src.shape[1:])
        return out.astype(val_src.dtype, copy=False)
    raise ValueError(f"Unknown mode: {mode!r}")


def common_time_axis(streams_ts: list, rate_hz: float = 50.0) -> np.ndarray:
    """Build a uniform-rate int64-ns time axis covering all given streams.

    The axis spans from max(min(ts)) to min(max(ts)) across non-empty streams
    (the intersection — guarantees every output sample has *some* source).

    Phase K9 (P2-7): intersection 이 비면 silent fallback 하지 않고 **빈 축**을
    반환한다. 이전 동작 ("첫 stream 범위 follow") 은 어떤 워커가 멈췄거나
    ts 가 깨졌을 때 잘못된 에피소드를 조용히 저장하는 위험이 있었다. 호출자
    (align_and_save_episode) 는 ts_axis.size < 2 분기에서 저장을 건너뛴다.
    """
    import logging
    valid = [t for t in streams_ts if isinstance(t, np.ndarray) and t.size > 0]
    if not valid:
        return np.empty(0, dtype=np.int64)
    t_start = max(int(t[0])  for t in valid)
    t_end   = min(int(t[-1]) for t in valid)
    if t_end <= t_start:
        # intersection 비어있음 → 빈 축 + 경고 로그. 저장 측이 명시적 skip.
        try:
            logging.getLogger(__name__).warning(
                f"[align] common_time_axis: stream intersection empty "
                f"(t_start={t_start}, t_end={t_end}) — returning empty axis."
            )
        except Exception:
            pass
        return np.empty(0, dtype=np.int64)
    period_ns = int(1e9 / float(rate_hz))
    n = max(1, (t_end - t_start) // period_ns + 1)
    return t_start + np.arange(n, dtype=np.int64) * period_ns
