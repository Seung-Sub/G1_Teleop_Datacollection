#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import pyarrow.parquet as pq

KEEP = slice(0, 25)

def list_episode_files(data_dir: str):
    pattern = os.path.join(data_dir, "chunk-*", "episode_*.parquet")
    return sorted(glob.glob(pattern))

def main():
    data_dir = "/home/ansur/G1_teleoperation/record/0129_glue_bimanual_pcaK7/data"
    max_files = 20   # ✅ 빠르게 확인(원하면 0으로 바꾸고 전체 돌려도 됨)
    q = 0.99

    files = list_episode_files(data_dir)
    if not files:
        raise RuntimeError(f"no parquet found under: {data_dir}")

    if max_files and max_files > 0:
        files = files[:max_files]

    X = []
    for i, f in enumerate(files):
        tab = pq.read_table(f, columns=["action"])
        A = np.array(tab["action"].to_pylist(), dtype=np.float32)
        X.append(A)
        if (i + 1) % 10 == 0:
            print(f">>> loaded {i+1}/{len(files)}", flush=True)

    X = np.concatenate(X, axis=0)

    D = X.shape[1]
    K = D - 25
    syn = slice(25, D)

    # q-quantile of abs values per-dimension
    q_keep = np.quantile(np.abs(X[:, KEEP]), q, axis=0)
    q_syn  = np.quantile(np.abs(X[:, syn]),  q, axis=0)

    std_keep = X[:, KEEP].std(axis=0)
    std_syn  = X[:, syn].std(axis=0)

    ratio_q = float(np.median(q_keep) / (np.median(q_syn) + 1e-9))
    ratio_std = float(np.median(std_keep) / (np.median(std_syn) + 1e-9))

    print("\n==== SCALE CHECK (ACTION) ====")
    print(f"Total dim: {D} | Synergy K: {K}")
    print(f"q={q}")
    print()
    print("median |x| q99 (keep 25 rad):", float(np.median(q_keep)))
    print("median |x| q99 (synergy K): ", float(np.median(q_syn)))
    print("q99 ratio keep/synergy    :", ratio_q)
    print()
    print("median std keep:", float(np.median(std_keep)))
    print("median std syn :", float(np.median(std_syn)))
    print("std ratio keep/synergy:", ratio_std)
    print()
    print("keep q99 min/med/max:", float(q_keep.min()), float(np.median(q_keep)), float(q_keep.max()))
    print("syn  q99 min/med/max:", float(q_syn.min()),  float(np.median(q_syn)),  float(q_syn.max()))

if __name__ == "__main__":
    main()
