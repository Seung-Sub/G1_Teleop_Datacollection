#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute PCA reconstruction RMSE for right-hand 16D using saved PCA basis (.npz).

Example:
  python3 check_pca_rmse.py \
    --src_data_dir /home/ansur/G1_teleoperation/record/0129_glue_bimanual/data \
    --basis_npz /home/ansur/G1_teleoperation/record/0129_glue_bimanual_pcaK7/pca_basis_action_k7.npz \
    --col action \
    --hand_slice 25:41 \
    --max_files 50
"""

import argparse
import os
from pathlib import Path
import numpy as np

def parse_slice(s: str) -> slice:
    # format: "start:stop" (stop exclusive), e.g., "25:41"
    if ":" not in s:
        raise ValueError("hand_slice must be like '25:41'")
    a, b = s.split(":")
    return slice(int(a), int(b))

def load_basis(npz_path: str):
    d = np.load(npz_path)
    # common keys: components / mean
    # allow a few variants to be robust
    if "components" in d:
        W = d["components"]
    elif "W" in d:
        W = d["W"]
    else:
        raise KeyError(f"Cannot find 'components' (or 'W') in {npz_path}")

    if "mean" in d:
        mu = d["mean"]
    elif "mu" in d:
        mu = d["mu"]
    else:
        raise KeyError(f"Cannot find 'mean' (or 'mu') in {npz_path}")

    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)

    # Expect: W shape (k, 16). If (16, k), transpose.
    if W.shape[0] == mu.shape[0] and W.shape[1] != mu.shape[0]:
        # (16, k) -> (k, 16)
        W = W.T

    if W.shape[1] != mu.shape[0]:
        raise ValueError(f"Basis shape mismatch: W={W.shape}, mu={mu.shape} (expect W=(k,16), mu=(16,))")

    return W, mu

def iter_parquets(data_dir: Path, max_files: int = 0):
    files = sorted(data_dir.rglob("*.parquet"))
    if max_files and max_files > 0:
        files = files[:max_files]
    for p in files:
        yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_data_dir", required=True, help="Directory containing ORIGINAL parquet data (41D).")
    ap.add_argument("--basis_npz", required=True, help="PCA basis npz (e.g., pca_basis_action_k7.npz).")
    ap.add_argument("--col", default="action", choices=["action", "observation.state"],
                    help="Which column to use. Default: action")
    ap.add_argument("--hand_slice", default="25:41",
                    help="Right-hand slice in the 41D vector, stop is exclusive. Default 25:41 (16D).")
    ap.add_argument("--max_files", type=int, default=0, help="Limit number of parquet files to scan (0 = all).")
    ap.add_argument("--max_rows_per_file", type=int, default=0, help="Limit rows per file (0 = all).")
    args = ap.parse_args()

    # Lazy import so the script fails with a clear message if missing deps.
    try:
        import pandas as pd
    except Exception as e:
        raise SystemExit("Missing pandas. Install: pip install pandas pyarrow") from e

    src_dir = Path(args.src_data_dir)
    if not src_dir.exists():
        raise SystemExit(f"src_data_dir not found: {src_dir}")

    W, mu = load_basis(args.basis_npz)
    k = W.shape[0]
    hand_sl = parse_slice(args.hand_slice)
    hand_dim = hand_sl.stop - hand_sl.start

    if hand_dim != mu.shape[0]:
        raise SystemExit(
            f"hand_slice gives dim={hand_dim}, but basis mean has dim={mu.shape[0]}.\n"
            f"Fix --hand_slice or use the correct basis npz."
        )

    # Accumulate squared error across all samples & dims
    sse = 0.0
    count = 0

    parquet_files = list(iter_parquets(src_dir, args.max_files))
    if not parquet_files:
        raise SystemExit(f"No parquet files found under: {src_dir}")

    for fp in parquet_files:
        df = pd.read_parquet(fp)

        if args.col not in df.columns:
            raise SystemExit(f"Column '{args.col}' not found in {fp}. Available: {list(df.columns)}")

        arr = df[args.col].to_numpy()

        # Optional row limit
        if args.max_rows_per_file and args.max_rows_per_file > 0:
            arr = arr[:args.max_rows_per_file]

        # Convert list/array objects -> stacked 2D array
        try:
            X = np.stack(arr).astype(np.float64)  # (N, 41)
        except Exception as e:
            raise SystemExit(
                f"Failed to stack column '{args.col}' from {fp}. "
                f"Maybe it is not a fixed-size vector per row."
            ) from e

        if X.ndim != 2 or X.shape[1] < hand_sl.stop:
            raise SystemExit(f"Unexpected shape from {fp}: {X.shape}. Check your dataset format and hand_slice.")

        hand16 = X[:, hand_sl]  # (N, 16)

        # PCA encode/decode using orthonormal components:
        # Z = (X - mu) @ W.T
        # X_rec = Z @ W + mu
        Xc = hand16 - mu
        Z = Xc @ W.T
        X_rec = Z @ W + mu

        diff = hand16 - X_rec
        sse += float(np.sum(diff * diff))
        count += int(diff.size)

    rmse = np.sqrt(sse / max(count, 1))

    print("==== PCA Reconstruction RMSE ====")
    print(f"source dir   : {src_dir}")
    print(f"basis npz    : {args.basis_npz}")
    print(f"column       : {args.col}")
    print(f"hand_slice   : {args.hand_slice} (dim={hand_dim})")
    print(f"k (PC count) : {k}")
    print(f"samples dims : {count}")
    print(f"RMSE         : {rmse:.8f}")
    print("Note: RMSE unit matches your joint unit (rad or deg).")

if __name__ == "__main__":
    main()
