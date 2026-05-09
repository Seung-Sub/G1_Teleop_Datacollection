"""LeRobot v2.1 -> ACT HDF5 (per-episode).

LeRobot 측 입력 (record/<task>/):
    data/chunk-XXX/episode_XXXXXX.parquet
    videos/chunk-XXX/observation.images.<view>/episode_XXXXXX.mp4

ACT 출력 (<out_dir>/episode_{N}.hdf5; N = sequential 0,1,2,...):
    /                                 attrs.sim = False
    /observations/qpos    (T, D_s)    float32
    /observations/qvel    (T, D_s)    float32   (numerical diff of qpos)
    /observations/images/<view>       (T, H, W, 3) uint8  chunks=(1,H,W,3)
    /action               (T, D_a)    float32

사용법:
    python data_refinement/convert_to_act.py --src record/pick-apple --out record/pick-apple_act
    python data_refinement/convert_to_act.py --src record/pick-apple --out ... \\
        --views ego_left_view ego_realsense
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
    import h5py
except ImportError:
    print("ERROR: h5py 가 설치되어 있지 않습니다. `pip install h5py`.", file=sys.stderr)
    raise

try:
    import cv2
except ImportError:
    print("ERROR: opencv-contrib-python 이 설치되어 있지 않습니다.", file=sys.stderr)
    raise


DEFAULT_VIEWS = ["ego_left_view", "ego_right_view", "ego_realsense"]


def _list_episodes(src: str) -> List[int]:
    files = glob.glob(os.path.join(src, "data", "chunk-*", "episode_*.parquet"))
    idxs = []
    for f in files:
        base = os.path.basename(f)
        try:
            idxs.append(int(base.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
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
        if arr.shape[0] > expected_frames:
            arr = arr[:expected_frames]
        else:
            pad = np.repeat(arr[-1:], expected_frames - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
    return arr


def _stack_list_col(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=np.float32) for x in df[col].tolist()], axis=0)


def _qvel_from_qpos(qpos: np.ndarray, fps: float) -> np.ndarray:
    """Numerical diff. shape (T, D); first frame copies frame 1."""
    if qpos.shape[0] < 2:
        return np.zeros_like(qpos)
    dt = 1.0 / max(fps, 1e-6)
    qvel = np.diff(qpos, axis=0) / dt
    qvel = np.concatenate([qvel[:1], qvel], axis=0)
    return qvel.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Convert LeRobot v2.1 dataset to ACT HDF5 episodes.")
    ap.add_argument("--src",    required=True, help="LeRobot dataset root (record/<task>)")
    ap.add_argument("--out",    required=True, help="Output dir to hold episode_{N}.hdf5")
    ap.add_argument("--views",  nargs="+", default=DEFAULT_VIEWS,
                    help=f"Camera views to embed under /observations/images (default: {DEFAULT_VIEWS})")
    ap.add_argument("--state-key",  default="observation.state",
                    help="Parquet column for qpos (default: observation.state)")
    ap.add_argument("--action-key", default="action",
                    help="Parquet column for action (default: action)")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="Recorder fps used to compute qvel (default: 20Hz)")
    ap.add_argument("--sim", action="store_true",
                    help="Mark root attrs.sim = True (default: False)")
    args = ap.parse_args()

    ep_indices = _list_episodes(args.src)
    if not ep_indices:
        print(f"No episodes found under {args.src}/data/chunk-*/", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(ep_indices)} episodes under {args.src}")

    os.makedirs(args.out, exist_ok=True)
    if any(name.endswith(".hdf5") for name in os.listdir(args.out)):
        print(f"ERROR: output dir not empty: {args.out}", file=sys.stderr)
        sys.exit(1)

    for k, ep in enumerate(ep_indices):
        pq, vids = _episode_paths(args.src, ep, args.views)
        df = pd.read_parquet(pq, engine="pyarrow")
        T = len(df)
        if T == 0:
            print(f"  [skip empty] episode {ep}")
            continue

        qpos   = _stack_list_col(df, args.state_key)
        qvel   = _qvel_from_qpos(qpos, args.fps)
        action = _stack_list_col(df, args.action_key)

        out_path = os.path.join(args.out, f"episode_{k}.hdf5")
        with h5py.File(out_path, "w", rdcc_nbytes=2 * 1024 * 1024) as root:
            root.attrs["sim"] = bool(args.sim)
            obs   = root.create_group("observations")
            imgs  = obs.create_group("images")
            obs.create_dataset("qpos", data=qpos)
            obs.create_dataset("qvel", data=qvel)
            for v in args.views:
                if not os.path.exists(vids[v]):
                    print(f"    [skip view] {v} missing for ep {ep}")
                    continue
                arr = _read_video(vids[v], expected_frames=T)
                H, W = arr.shape[1], arr.shape[2]
                imgs.create_dataset(v, data=arr, dtype="uint8", chunks=(1, H, W, 3))
            root.create_dataset("action", data=action)

        print(f"  [{k+1}/{len(ep_indices)}] ep {ep:06d} T={T} -> {out_path}")

    print(f"\nDone. Wrote {len(ep_indices)} HDF5 files to {args.out}")


if __name__ == "__main__":
    main()
