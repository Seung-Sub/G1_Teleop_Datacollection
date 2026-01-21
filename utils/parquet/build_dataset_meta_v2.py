import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

import numpy as np
import pandas as pd

def rglob_parquets(data_root: Path):
    return sorted(data_root.rglob("*.parquet"))

# def step1_write_task_index(dataset_root: Path, selected_task_index: int):
#     """
#     For every parquet under data/**, overwrite/add df['task_index'] = selected_task_index.
#     """
#     data_root = dataset_root / "data"
#     files = rglob_parquets(data_root)
#     if not files:
#         print(f"[1] Parquet 없음: {data_root}")
#         return
#     print(f"[1] task_index={selected_task_index} 를 모든 parquet에 기록 중... (총 {len(files)}개)")
#     for pq in files:
#         try:
#             df = pd.read_parquet(pq)
#             df["task_index"] = int(selected_task_index)
#             df.to_parquet(pq, index=False)
#             print(f"  - Updated: {pq}")
#         except Exception as e:
#             print(f"  ! 실패: {pq} -> {e}")
#     print("[1] 완료")

def step1_write_task_index(dataset_root: Path, selected_task_index: int):
    """
    For every parquet under data/**, overwrite/add df['task_index'] = selected_task_index.
    """
    data_root = dataset_root / "data"
    files = rglob_parquets(data_root)
    if not files:
        print(f"[1] Parquet 없음: {data_root}")
        return
    print(f"[1] task_index={selected_task_index} 를 모든 parquet에 기록 중... (총 {len(files)}개)")
    for pq in files:
        try:
            df = pd.read_parquet(pq)
            df["task_index"] = int(selected_task_index)
            df.to_parquet(pq, index=False)
            print(f"  - Updated: {pq}")
        except Exception as e:
            print(f"  ! 실패: {pq} -> {e}")
    print("[1] 완료")