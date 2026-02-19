#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
import numpy as np
import pyarrow.parquet as pq

KEEP_SLICE = slice(0, 25)
RH_SLICE   = slice(25, 41)

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

def read_listcol_as_2d(path: str, col: str) -> np.ndarray:
    tab = pq.read_table(path, columns=[col])
    return np.array(tab[col].to_pylist(), dtype=np.float32)

def fit_pca_svd(X: np.ndarray, k: int, q: float = 0.99):
    N, D = X.shape
    mu = X.mean(axis=0, keepdims=True)
    X0 = X - mu

    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    W = Vt[:k].T  # (D, k)

    eig = (S**2) / max(N - 1, 1)
    total = float(eig.sum()) if eig.size else 1.0
    evr = (eig[:k] / total).astype(np.float32)

    S_raw = X0 @ W
    X_hat = (S_raw @ W.T) + mu
    rmse = float(np.sqrt(np.mean((X - X_hat) ** 2)))

    scale = np.quantile(np.abs(S_raw), q, axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    return mu.squeeze().astype(np.float32), W.astype(np.float32), scale, evr, rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_data_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--max_files", type=int, default=0, help="0이면 전체, 아니면 앞에서 N개만")
    ap.add_argument("--quantile", type=float, default=0.99)
    args = ap.parse_args()

    files = list_episode_files(args.input_data_dir)
    if not files:
        raise RuntimeError(f"episode parquet 못 찾음: {args.input_data_dir}")
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    os.makedirs(args.out_root, exist_ok=True)
    print(f"[INFO] episodes: {len(files)}  K={args.k}")

    X_act, X_obs = [], []
    for i, f in enumerate(files):
        A = read_listcol_as_2d(f, "action")
        S = read_listcol_as_2d(f, "observation.state")
        if A.shape[1] != 41 or S.shape[1] != 41:
            raise RuntimeError(f"dim error: {f} action={A.shape} state={S.shape}")
        X_act.append(A[:, RH_SLICE])  # (T,16)
        X_obs.append(S[:, RH_SLICE])  # (T,16)
        if (i+1) % 20 == 0:
            print(f"  loaded {i+1}/{len(files)}")

    X_act = np.concatenate(X_act, axis=0)
    X_obs = np.concatenate(X_obs, axis=0)

    mu_a, W_a, scale_a, evr_a, rmse_a = fit_pca_svd(X_act, args.k, q=args.quantile)
    mu_o, W_o, scale_o, evr_o, rmse_o = fit_pca_svd(X_obs, args.k, q=args.quantile)

    np.savez(os.path.join(args.out_root, f"pca_basis_action_k{args.k}.npz"),
             mu=mu_a, W=W_a, scale=scale_a,
             keep_start=KEEP_SLICE.start, keep_end=KEEP_SLICE.stop,
             rh_start=RH_SLICE.start, rh_end=RH_SLICE.stop,
             explained_var_ratio=evr_a, recon_rmse=np.float32(rmse_a))

    np.savez(os.path.join(args.out_root, f"pca_basis_obs_k{args.k}.npz"),
             mu=mu_o, W=W_o, scale=scale_o,
             keep_start=KEEP_SLICE.start, keep_end=KEEP_SLICE.stop,
             rh_start=RH_SLICE.start, rh_end=RH_SLICE.stop,
             explained_var_ratio=evr_o, recon_rmse=np.float32(rmse_o))

    print("\n=== PCA summary (RIGHT HAND 16DOF) ===")
    print(f"[ACTION] EVR(sum)={float(evr_a.sum()):.4f}, recon_RMSE={rmse_a:.6f}")
    print(f"[OBS   ] EVR(sum)={float(evr_o.sum()):.4f}, recon_RMSE={rmse_o:.6f}")
    print("[DONE] basis saved to:", args.out_root)

if __name__ == "__main__":
    main()




############
# python3 pca_fit_basis_only.py \
#   --input_data_dir "/home/ansur/G1_teleoperation/record/0129_glue_bimanual/data" \
#   --out_root "/home/ansur/G1_teleoperation/record/0129_glue_bimanual_pcaK7" \
#   --k 7
