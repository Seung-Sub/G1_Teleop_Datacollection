"""LeRobot v2.1 -> Diffusion Policy zarr (replay buffer).

LeRobot 측 입력 (record/<task>/):
    data/chunk-XXX/episode_XXXXXX.parquet
        columns: observation.state, observation.sensor, action,
                 timestamp, episode_index, index, frame_index
    videos/chunk-XXX/observation.images.<view>/episode_XXXXXX.mp4

DP zarr 출력 (<out>.zarr/):
    data/
      state         (N, D_s)  float32
      action        (N, D_a)  float32
      timestamp     (N,)      float64
      camera_0      (N, H, W, 3) uint8   <- view 0 (default: ego_left_view)
      camera_1      (N, H, W, 3) uint8   <- view 1 (default: ego_right_view)
      camera_2      (N, H, W, 3) uint8   <- view 2 (default: ego_realsense)
    meta/
      episode_ends  (E,)      int64       cumulative end indices

zarr 그룹의 invariant:
    all data/* arrays share dim-0 == meta/episode_ends[-1]

사용법:
    python data_refinement/convert_to_dp.py \\
        --src record/pick-apple --out record/pick-apple.zarr
    python data_refinement/convert_to_dp.py \\
        --src record/pick-apple --out record/pick-apple.zarr --views ego_left_view ego_realsense
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import zarr
    from numcodecs import Blosc
except ImportError as e:
    print("ERROR: zarr/numcodecs 가 설치되어 있지 않습니다. `pip install -e '.[dataset_dp]'` 필요.", file=sys.stderr)
    raise

try:
    import cv2
except ImportError as e:
    print("ERROR: opencv-contrib-python 이 설치되어 있지 않습니다.", file=sys.stderr)
    raise


DEFAULT_VIEWS = ["ego", "wrist_l", "wrist_r"]


def _downsample_indices(n_frames: int, src_fps: float, tgt_fps: float) -> np.ndarray:
    """정수배 다운샘플 인덱스. 60→10=6:1 처럼 정수배면 step 샘플링(값 보존).
    정수배가 아니면 균일 간격 근사."""
    src_fps, tgt_fps = float(src_fps), float(tgt_fps)
    if tgt_fps >= src_fps:
        return np.arange(n_frames)  # 다운샘플 불필요
    ratio = src_fps / tgt_fps
    if abs(ratio - round(ratio)) < 1e-6:
        step = int(round(ratio))
        return np.arange(0, n_frames, step)
    n_out = int(round(n_frames * tgt_fps / src_fps))
    return np.linspace(0, n_frames - 1, n_out).round().astype(int)


def _list_episodes(src: str) -> List[int]:
    """Return sorted episode indices found under src/data/chunk-*/"""
    files = glob.glob(os.path.join(src, "data", "chunk-*", "episode_*.parquet"))
    idxs = []
    for f in files:
        base = os.path.basename(f)
        try:
            idxs.append(int(base.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            print(f"  [skip non-conforming] {f}")
    return sorted(set(idxs))


def _episode_paths(src: str, ep: int, views: List[str]):
    chunk_id = ep // 1000
    parquet = os.path.join(src, "data", f"chunk-{chunk_id:03d}", f"episode_{ep:06d}.parquet")
    videos = {}
    for v in views:
        videos[v] = os.path.join(src, "videos", f"chunk-{chunk_id:03d}",
                                 f"observation.images.{v}", f"episode_{ep:06d}.mp4")
    return parquet, videos


def _read_video(path: str, expected_frames: Optional[int] = None) -> np.ndarray:
    """Decode an mp4 to (T,H,W,3) uint8 RGB. Pad/clip to expected_frames if given."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    arr = np.stack(frames, axis=0).astype(np.uint8)
    if expected_frames is not None and arr.shape[0] != expected_frames:
        # Trim or pad-with-last-frame to align with parquet length.
        if arr.shape[0] > expected_frames:
            arr = arr[:expected_frames]
        else:
            pad = np.repeat(arr[-1:], expected_frames - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
    return arr


def _stack_list_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """parquet list-column -> (T, D) float32"""
    return np.stack([np.asarray(x, dtype=np.float32) for x in df[col].tolist()], axis=0)


def main():
    ap = argparse.ArgumentParser(description="Convert LeRobot v2.1 dataset to Diffusion Policy zarr.")
    ap.add_argument("--src",   required=True, help="LeRobot dataset root (record/<task>)")
    ap.add_argument("--out",   required=True, help="Output zarr path (e.g. .../out.zarr)")
    ap.add_argument("--views", nargs="+", default=DEFAULT_VIEWS,
                    help=f"Camera views to embed (default: {DEFAULT_VIEWS})")
    ap.add_argument("--state-key",  default="observation.state",
                    help="Parquet column for proprio (default: observation.state)")
    ap.add_argument("--action-key", default="action",
                    help="Parquet column for action (default: action)")
    ap.add_argument("--include-sensor", action="store_true",
                    help="Also store observation.sensor under data/sensor (if present).")
    ap.add_argument("--src-fps", type=float, default=60.0,
                    help="Source data fps (default: 60).")
    ap.add_argument("--tgt-fps", type=float, default=10.0,
                    help="Target fps for Diffusion Policy (default: 10). 60->10 = 6:1.")
    args = ap.parse_args()

    ep_indices = _list_episodes(args.src)
    if not ep_indices:
        print(f"No episodes found under {args.src}/data/chunk-*/", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(ep_indices)} episodes under {args.src}")

    if os.path.exists(args.out):
        print(f"ERROR: output already exists: {args.out}", file=sys.stderr)
        sys.exit(1)

    store = zarr.DirectoryStore(args.out)
    root  = zarr.group(store=store, overwrite=True)
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.NOSHUFFLE)

    # Probe first episode to allocate shapes.
    pq0, vids0 = _episode_paths(args.src, ep_indices[0], args.views)
    df0 = pd.read_parquet(pq0, engine="pyarrow")
    state0  = _stack_list_col(df0, args.state_key)
    action0 = _stack_list_col(df0, args.action_key)
    T0, D_s = state0.shape
    _,  D_a = action0.shape

    # Probe first video for HxW.
    cam_shapes = {}
    for v in args.views:
        if not os.path.exists(vids0[v]):
            print(f"  [skip view] {v} not found in episode {ep_indices[0]}")
            continue
        a = _read_video(vids0[v])
        cam_shapes[v] = a.shape[1:3]  # (H, W)
        print(f"  view {v} -> shape {a.shape}")

    print(f"state_dim={D_s} action_dim={D_a}")

    # Allocate resizable datasets so add_episode-style append works.
    state_arr  = data_grp.zeros("state",     shape=(0, D_s), chunks=(1024, D_s),
                                dtype="f4", compressor=compressor)
    action_arr = data_grp.zeros("action",    shape=(0, D_a), chunks=(1024, D_a),
                                dtype="f4", compressor=compressor)
    ts_arr     = data_grp.zeros("timestamp", shape=(0,),     chunks=(1024,),
                                dtype="f8", compressor=compressor)
    sensor_arr = None
    if args.include_sensor and "observation.sensor" in df0.columns:
        sensor0 = _stack_list_col(df0, "observation.sensor")
        sensor_arr = data_grp.zeros("sensor", shape=(0, sensor0.shape[1]),
                                    chunks=(1024, sensor0.shape[1]),
                                    dtype="f4", compressor=compressor)

    cam_arrs = {}
    for v, (H, W) in cam_shapes.items():
        # camera index follows DP convention: camera_0, camera_1, ...
        cam_idx = list(args.views).index(v)
        name = f"camera_{cam_idx}"
        cam_arrs[v] = data_grp.zeros(name, shape=(0, H, W, 3),
                                     chunks=(8, H, W, 3),
                                     dtype="u1", compressor=compressor)

    episode_ends: List[int] = []
    cumulative = 0
    for k, ep in enumerate(ep_indices):
        pq, vids = _episode_paths(args.src, ep, args.views)
        df = pd.read_parquet(pq, engine="pyarrow")
        T_full = len(df)
        if T_full == 0:
            print(f"  [skip empty] episode {ep}")
            continue

        # 60->tgt fps 다운샘플 인덱스 (state/action/ts/video 공통 적용 → 정합 보장).
        keep = _downsample_indices(T_full, args.src_fps, args.tgt_fps)
        T = len(keep)

        state  = _stack_list_col(df, args.state_key)[keep]
        action = _stack_list_col(df, args.action_key)[keep]
        # timestamp 는 tgt_fps 기준으로 재계산 (다운샘플 후 0,1/tgt,2/tgt,...).
        ts     = (np.arange(T) / float(args.tgt_fps)).astype(np.float64)

        state_arr.append(state)
        action_arr.append(action)
        ts_arr.append(ts)
        if sensor_arr is not None and "observation.sensor" in df.columns:
            sensor_arr.append(_stack_list_col(df, "observation.sensor")[keep])

        for v, arr in cam_arrs.items():
            if not os.path.exists(vids[v]):
                # Pad with zeros to keep all data/* aligned.
                H, W = cam_shapes[v]
                arr.append(np.zeros((T, H, W, 3), dtype=np.uint8))
            else:
                # 전체 디코드 후 동일 keep 인덱스로 다운샘플 (state 와 정합).
                vid = _read_video(vids[v], expected_frames=T_full)
                arr.append(vid[keep])

        cumulative += T
        episode_ends.append(cumulative)
        print(f"  [{k+1}/{len(ep_indices)}] ep {ep:06d} T={T_full}->{T} (x{args.src_fps:.0f}->{args.tgt_fps:.0f}fps) cum={cumulative}")

    meta_grp.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64),
                            compressor=None, overwrite=True)

    print(f"\nDone. {len(episode_ends)} episodes, {cumulative} frames -> {args.out}")
    print(f"Top-level keys: {list(root.keys())}; data keys: {list(data_grp.keys())}")


if __name__ == "__main__":
    main()
