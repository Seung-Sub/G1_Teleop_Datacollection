#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEEP_SLICE = slice(0, 25)     # 25 DOF 유지
RH_SLICE   = slice(25, 41)    # 오른손 16 DOF -> PCA K로

def list_episode_files(data_dir: str):
    patterns = [
        os.path.join(data_dir, "chunk-*", "episode_*.parquet"),
        os.path.join(data_dir, "episode_*.parquet"),
        os.path.join(data_dir, "**", "episode_*.parquet"),
    ]
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    return sorted(set(files))

def load_basis(npz_path: str):
    z = np.load(npz_path)
    return z["mu"].astype(np.float32), z["W"].astype(np.float32), z["scale"].astype(np.float32)

def encode_synergy(x_rh: np.ndarray, mu: np.ndarray, W: np.ndarray, scale: np.ndarray, clip: float | None):
    x0 = x_rh - mu[None, :]
    s_raw = x0 @ W
    s = s_raw / scale[None, :]
    if clip is not None:
        s = np.clip(s, -clip, clip)
    return s.astype(np.float32)

def replace_list_col(table: pa.Table, col_name: str, new_2d: np.ndarray) -> pa.Table:
    idx = table.schema.get_field_index(col_name)
    if idx < 0:
        raise RuntimeError(f"컬럼 없음: {col_name}")
    new_arr = pa.array([row.tolist() for row in new_2d], type=pa.list_(pa.float32()))
    return table.set_column(idx, col_name, new_arr)

def ensure_videos(src_root: str, dst_root: str, mode: str, overwrite_videos: bool):
    """
    src_root: 원본 dataset 루트 (data/, videos/)
    dst_root: 새 dataset 루트 (data/, videos/)
    mode: skip | symlink | copy
    overwrite_videos: True면 dst_root/videos가 있으면 지우고 다시 생성
    """
    if mode == "skip":
        return

    src_videos = os.path.join(src_root, "videos")
    dst_videos = os.path.join(dst_root, "videos")

    if not os.path.isdir(src_videos):
        print(f"[WARN] src videos 없음: {src_videos} (skip)")
        return

    if os.path.lexists(dst_videos):
        if not overwrite_videos:
            print(f"[INFO] dst videos already exists: {dst_videos} (skip; use --overwrite_videos to replace)")
            return
        # 덮어쓰기: 기존 삭제
        if os.path.islink(dst_videos) or os.path.isfile(dst_videos):
            os.unlink(dst_videos)
        else:
            shutil.rmtree(dst_videos)
        print(f"[INFO] removed existing dst videos: {dst_videos}")

    if mode == "symlink":
        os.symlink(src_videos, dst_videos)
        print(f"[DONE] videos symlink: {dst_videos} -> {src_videos}")
    elif mode == "copy":
        print("[INFO] copying videos... (may take time)")
        shutil.copytree(src_videos, dst_videos)
        print(f"[DONE] videos copied: {dst_videos}")
    else:
        raise ValueError("videos_mode must be one of: skip, symlink, copy")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_data_dir", required=True, help="원본 dataset의 data 폴더 경로")
    ap.add_argument("--output_root", required=True, help="새 dataset 루트 경로. (여기에 data/, videos/ 생성)")
    ap.add_argument("--basis_action", required=True)
    ap.add_argument("--basis_obs", required=True)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true", help="data parquet 덮어쓰기")
    ap.add_argument("--videos_mode", choices=["skip", "symlink", "copy"], default="symlink")
    ap.add_argument("--src_root", required=True, help="원본 dataset 루트(여기서 videos/ 가져옴)")
    ap.add_argument("--overwrite_videos", action="store_true", help="videos도 덮어쓰기")
    args = ap.parse_args()

    files = list_episode_files(args.input_data_dir)
    if not files:
        raise RuntimeError(f"episode parquet 못 찾음: {args.input_data_dir}")

    mu_a, W_a, scale_a = load_basis(args.basis_action)
    mu_o, W_o, scale_o = load_basis(args.basis_obs)
    K = W_a.shape[1]

    out_data_dir = os.path.join(args.output_root, "data")
    os.makedirs(out_data_dir, exist_ok=True)

    print(f"[INFO] episodes: {len(files)}  K={K}")
    print(f"[INFO] output_root: {args.output_root}")
    print(f"[INFO] output_data: {out_data_dir}")

    wrote = 0
    for i, f in enumerate(files):
        rel = os.path.relpath(f, args.input_data_dir)  # chunk-000/episode_XXXX.parquet 유지
        out_path = os.path.join(out_data_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if (not args.overwrite) and os.path.exists(out_path):
            continue

        tab = pq.read_table(f)
        A41 = np.array(tab["action"].to_pylist(), dtype=np.float32)
        S41 = np.array(tab["observation.state"].to_pylist(), dtype=np.float32)

        keepA = A41[:, KEEP_SLICE]      # (T,25)
        rhA   = A41[:, RH_SLICE]        # (T,16)
        sA    = encode_synergy(rhA, mu_a, W_a, scale_a, clip=args.clip)  # (T,K)
        A32   = np.concatenate([keepA, sA], axis=1)

        keepS = S41[:, KEEP_SLICE]
        rhS   = S41[:, RH_SLICE]
        sS    = encode_synergy(rhS, mu_o, W_o, scale_o, clip=args.clip)
        S32   = np.concatenate([keepS, sS], axis=1)

        tab2 = tab
        tab2 = replace_list_col(tab2, "action", A32)
        tab2 = replace_list_col(tab2, "observation.state", S32)

        pq.write_table(tab2, out_path, compression="snappy")
        wrote += 1

        if (i+1) % 20 == 0:
            print(f"  converted {i+1}/{len(files)}")

    print(f"[DONE] wrote {wrote} parquet files under: {out_data_dir}")

    # videos 처리
    ensure_videos(args.src_root, args.output_root, args.videos_mode, args.overwrite_videos)

    # sanity check
    first_out = os.path.join(out_data_dir, os.path.relpath(files[0], args.input_data_dir))
    pf = pq.ParquetFile(first_out)
    t = pf.read_row_group(0, columns=["action","observation.state"]).to_pydict()
    print("[CHECK] first output:", first_out)
    print("[CHECK] action_len =", len(t["action"][0]), "state_len =", len(t["observation.state"][0]))

if __name__ == "__main__":
    main()






# python3 /home/ansur/G1_teleoperation/data_refinement/pca_convert_41_to_32.py \
#   --input_data_dir "/home/ansur/G1_teleoperation/record/0209_apple_pickup_pca/data" \
#   --output_root    "/home/ansur/G1_teleoperation/record/0209_apple_pickup_pca32K7" \
#   --basis_action   "/home/ansur/G1_teleoperation/record/0209_apple_pickup_pcaK7/pca_basis_action_k7.npz" \
#   --basis_obs      "/home/ansur/G1_teleoperation/record/0209_apple_pickup_pcaK7/pca_basis_obs_k7.npz" \
#   --clip 1.0 \
#   --overwrite \
#   --src_root "/home/ansur/G1_teleoperation/record/0209_apple_pickup_pca" \
#   --videos_mode copy \
#   --overwrite_videos