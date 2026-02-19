#!/usr/bin/env python3
import os, glob
import numpy as np
import pyarrow.parquet as pq

data_dir = "/home/ansur/G1_teleoperation/record/0129_glue_bimanual_pcaK7/data"
files = sorted(glob.glob(os.path.join(data_dir, "chunk-*", "episode_*.parquet")))[:20]

X = []
for f in files:
    tab = pq.read_table(f, columns=["action"])
    A = np.array(tab["action"].to_pylist(), dtype=np.float32)
    X.append(A)

X = np.concatenate(X, axis=0)
syn = X[:, 25:]  # (N,K)

# clip=1.0 기준으로 포화된 비율
eps = 1e-6
sat = (np.abs(syn) >= (1.0 - eps))
sat_ratio = sat.mean()

# 축별로도
sat_ratio_dim = sat.mean(axis=0)

print("Total sat ratio (|s|>=1):", float(sat_ratio))
print("Sat ratio per dim:", sat_ratio_dim.tolist())
