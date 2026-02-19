#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np

def parse_slice(s: str) -> slice:
    a, b = s.split(":")
    return slice(int(a), int(b))

def load_basis(npz_path: str):
    d = np.load(npz_path)

    # basis keys (robust)
    if "components" in d:
        W = d["components"]
    elif "W" in d:
        W = d["W"]
    else:
        raise KeyError("npz must contain 'components' or 'W'")

    if "mean" in d:
        mu = d["mean"]
    elif "mu" in d:
        mu = d["mu"]
    else:
        raise KeyError("npz must contain 'mean' or 'mu'")

    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)

    # If stored as (16, k) instead of (k, 16), transpose to (k, 16)
    if W.shape[0] == mu.shape[0] and W.shape[1] != mu.shape[0]:
        W = W.T

    # Expect (k, 16) and mean (16,)
    if W.shape[1] != mu.shape[0]:
        raise ValueError(f"Shape mismatch: W={W.shape}, mu={mu.shape}")

    return W, mu

def iter_parquets(data_dir: Path, max_files: int = 0):
    files = sorted(data_dir.rglob("*.parquet"))
    if max_files and max_files > 0:
        files = files[:max_files]
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_data_dir", required=True, help="ORIGINAL parquet dir (41D).")
    ap.add_argument("--basis_npz", required=True, help="PCA basis npz (k<=7).")
    ap.add_argument("--col", default="action", help="Column name to read (default: action).")
    ap.add_argument("--hand_slice", default="25:41", help="Right-hand slice in 41D (default: 25:41).")
    ap.add_argument("--max_k", type=int, default=7, help="Max k to sweep (default: 7).")
    ap.add_argument("--max_files", type=int, default=0, help="Limit parquet files (0=all).")
    ap.add_argument("--max_rows_per_file", type=int, default=0, help="Limit rows per file (0=all).")
    ap.add_argument("--out_png", default="rmse_vs_k.png", help="Output plot filename.")
    args = ap.parse_args()

    try:
        import pandas as pd
    except Exception as e:
        raise SystemExit("Missing pandas/pyarrow. Install: pip install pandas pyarrow") from e

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("Missing matplotlib. Install: pip install matplotlib") from e

    src_dir = Path(args.src_data_dir)
    if not src_dir.exists():
        raise SystemExit(f"src_data_dir not found: {src_dir}")

    W_full, mu = load_basis(args.basis_npz)
    k_available = W_full.shape[0]
    max_k = min(args.max_k, k_available)

    hand_sl = parse_slice(args.hand_slice)
    hand_dim = hand_sl.stop - hand_sl.start
    if hand_dim != mu.shape[0]:
        raise SystemExit(
            f"hand_slice dim={hand_dim} but basis mean dim={mu.shape[0]}. "
            f"Fix --hand_slice or use matching basis."
        )

    parquet_files = iter_parquets(src_dir, args.max_files)
    if not parquet_files:
        raise SystemExit(f"No parquet files found under: {src_dir}")

    # Accumulate SSE separately for each k
    sse = np.zeros(max_k + 1, dtype=np.float64)   # index 1..max_k used
    cnt = 0

    for fp in parquet_files:
        df = pd.read_parquet(fp)
        if args.col not in df.columns:
            raise SystemExit(f"Column '{args.col}' not in {fp}. Available: {list(df.columns)}")

        arr = df[args.col].to_numpy()
        if args.max_rows_per_file and args.max_rows_per_file > 0:
            arr = arr[:args.max_rows_per_file]

        X = np.stack(arr).astype(np.float64)  # (N, 41)
        if X.ndim != 2 or X.shape[1] < hand_sl.stop:
            raise SystemExit(f"Unexpected shape from {fp}: {X.shape}")

        hand16 = X[:, hand_sl]  # (N,16)
        Xc = hand16 - mu        # center

        # For each k: encode/decode using first k PCs
        for k in range(1, max_k + 1):
            Wk = W_full[:k, :]           # (k,16)
            Z = Xc @ Wk.T                # (N,k)
            X_rec = Z @ Wk + mu          # (N,16)
            diff = hand16 - X_rec
            sse[k] += np.sum(diff * diff)

        cnt += hand16.size  # N*16

    rmse = np.sqrt(sse[1:] / cnt)

    print("==== RMSE vs k ====")
    print(f"basis_npz  : {args.basis_npz}")
    print(f"src_data   : {src_dir}")
    print(f"column     : {args.col}")
    print(f"hand_slice : {args.hand_slice}")
    print(f"samples*d  : {cnt}")
    for i, v in enumerate(rmse, start=1):
        print(f"k={i}: RMSE={v:.8f}")

    # Plot
    ks = np.arange(1, max_k + 1)
    plt.figure()
    plt.plot(ks, rmse, marker="o")
    plt.xlabel("k (number of PCs)")
    plt.ylabel("Reconstruction RMSE")
    plt.title("PCA Reconstruction RMSE vs k")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved plot: {args.out_png}")

if __name__ == "__main__":
    main()



#########################
# python3 sweep_pca_rmse_k1to7.py \
#   --src_data_dir /home/ansur/G1_teleoperation/record/0129_glue_bimanual/data \
#   --basis_npz /home/ansur/G1_teleoperation/record/0129_glue_bimanual_pcaK7/pca_basis_action_k7.npz \
#   --col action \
#   --hand_slice 25:41 \
#   --max_k 7 \
#   --out_png /home/ansur/G1_teleoperation/record/0129_glue_bimanual_pcaK7/rmse_vs_k_action.png
