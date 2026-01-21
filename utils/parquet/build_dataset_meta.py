#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build meta files for g1-pick-apple in one go (Steps 1~5) with a simple folder UI.

Outputs (under DATASET_ROOT/meta):
  - info.json
  - stats.json
  - episodes.json
Also updates Parquet files in-place:
  - Adds/overwrites `task_index`
  - Adds/overwrites `observation.img_state_delta`

Requires:
  - Python 3.9+
  - pandas, numpy, pyarrow (or fastparquet), opencv-python
  - (Optional) ffprobe on PATH for richer video meta (falls back to OpenCV)
"""
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

# OpenCV is used in Step 2 and as a fallback for video probing in Step 3
try:
    import cv2
except Exception:
    cv2 = None

# ----------------------------- CLI & UI helpers -----------------------------
def pick_dataset_root_via_ui() -> Optional[Path]:
    """
    폴더 선택 대화상자를 'record/' 기준으로 열어 DATASET_ROOT(예: record/g1-pick-apple)를 고르게 한다.
    사용자가 'record/' 자체를 선택하면 그 안에서 다시 한 번 하위 데이터셋 폴더를 고르게 한다.
    """
    # 스크립트 위치 기준으로 record 후보 경로 추정
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    try:
        base_dir   = Path(__file__).resolve().parent
        parent_dir = (base_dir / ".." / "..").resolve()
        record_hint = (parent_dir / "record").resolve()
    except Exception:
        record_hint = None

    # 1차: record/에서 시작해 폴더 선택
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Select Dataset Root",
                            "DATASET_ROOT를 선택하세요.\n예: record/g1-pick-apple")
        initialdir = str(record_hint) if (record_hint and record_hint.exists()) else None
        sel1 = filedialog.askdirectory(
            title="Select DATASET_ROOT (record/ 하위 데이터셋 폴더)",
            initialdir=initialdir
        )
        if not sel1:
            root.destroy()
            return None
        sel1_path = Path(sel1)

        # 사용자가 record/ 자체를 선택했는지 확인
        if sel1_path.name == "record":
            # record/ 아래에서 하위 폴더(데이터셋) 다시 선택
            messagebox.showinfo("Select Dataset Under record/",
                                "record/ 하위의 데이터셋 폴더를 선택하세요.")
            sel2 = filedialog.askdirectory(
                title="Select a dataset folder under record/",
                initialdir=str(sel1_path)
            )
            root.destroy()
            if not sel2:
                return None
            ds = Path(sel2)
            # 안전검사: 최소한 data/, videos/가 있는 폴더인지 힌트 검사
            if not ((ds / "data").exists() and (ds / "videos").exists()):
                messagebox.showwarning("Not a dataset folder",
                                       "선택한 폴더에 data/ 또는 videos/가 없습니다. 다시 선택해주세요.")
                return None
            return ds

        else:
            root.destroy()
            # record/ 하위가 아니라도 사용자가 직접 정확히 고른 경우 허용
            return sel1_path

    except Exception:
        # UI 실패 시 None 반환
        try:
            root.destroy()
        except Exception:
            pass
        return None

def pick_task_index_via_ui(dataset_root: Path, default: int = 0) -> int:
    """
    Show a minimal input dialog to choose task_index from meta/tasks.jsonl if available.
    Fallback to default if UI unavailable or user cancels.
    """
    tasks_map = load_tasks_map(dataset_root)
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        if tasks_map:
            msg_lines = ["tasks.jsonl:", *[f"- {k}: {v}" for k, v in tasks_map.items()]]
            prompt = "\n".join(msg_lines) + "\n\n사용할 task_index 숫자를 입력하세요:"
        else:
            prompt = "tasks.jsonl을 찾지 못했습니다. 사용할 task_index 숫자를 입력하세요 (기본 0):"
        ans = simpledialog.askstring("task_index 선택", prompt)
        root.destroy()
        if ans is None or ans.strip() == "":
            return default
        return int(ans.strip())
    except Exception:
        return default

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------- Common loaders ------------------------------
def load_tasks_map(dataset_root: Path) -> dict:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    tasks_map = {}
    if tasks_path.exists():
        with open(tasks_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    ti  = obj.get("task_index")
                    ts  = obj.get("task")
                    if ti is not None and ts is not None:
                        tasks_map[int(ti)] = ts
                except json.JSONDecodeError:
                    print(f"⚠️ tasks.jsonl 라인 파싱 실패: {ln[:120]}...")
    else:
        print(f"⚠️ '{tasks_path}' 가 없습니다. task 문자열 없이 진행합니다.")
    return tasks_map

def rglob_parquets(data_root: Path):
    return sorted(data_root.rglob("*.parquet"))

def find_chunk_name(pq_path: Path, data_root: Path) -> str:
    """
    Find nearest ancestor directory named 'chunk-XXX'; else return 'chunk-000'.
    """
    for parent in [pq_path.parent, *pq_path.parents]:
        if parent == data_root.parent or parent == data_root:
            break
        if parent.name.startswith("chunk-"):
            return parent.name
    # Fallback: detect immediate sibling 'chunk-xxx' in path
    for part in pq_path.parts:
        if str(part).startswith("chunk-"):
            return str(part)
    return "chunk-000"

# ----------------------------- Step 1 ---------------------------------------
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

# ----------------------------- Step 2 ---------------------------------------
def step2_write_img_state_delta(dataset_root: Path, primary_view_key: str):
    """
    For every parquet, compute per-frame gray abs diff mean between consecutive frames
    from the matched video and write 'observation.img_state_delta'.
    """
    if cv2 is None:
        print("[2] OpenCV(cv2)가 필요합니다. 설치 후 다시 시도하세요.")
        return

    data_root = dataset_root / "data"
    video_root = dataset_root / "videos"
    files = rglob_parquets(data_root)
    if not files:
        print(f"[2] Parquet 없음: {data_root}")
        return

    def pq_to_video(pq_path: Path) -> Path:
        episode_name = pq_path.stem  # e.g., "episode_000123"
        chunk_name = find_chunk_name(pq_path, data_root)
        return video_root / chunk_name / primary_view_key / f"{episode_name}.mp4"

    print(f"[2] '{primary_view_key}' 비디오 기반 img_state_delta 계산 중... (총 {len(files)}개)")
    for pq in files:
        df = None
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"  ! Parquet 읽기 실패: {pq} -> {e}")
            continue

        vid_path = pq_to_video(pq)
        if not vid_path.exists():
            print(f"  ! 비디오 없음: {vid_path}")
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"  ! 비디오 열기 실패: {vid_path}")
            continue

        deltas = []
        ok, prev_frame = cap.read()
        if not ok:
            print(f"  ! 첫 프레임 읽기 실패: {vid_path}")
            cap.release(); continue

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            diff = np.abs(gray - prev_gray)
            deltas.append(float(diff.mean()))
            prev_gray = gray
        cap.release()

        # length alignment
        if len(deltas) != len(df):
            print(f"  ~ 길이 불일치: frames={len(deltas)} rows={len(df)} -> 보정")
            if len(deltas) < len(df):
                deltas += [0.0] * (len(df) - len(deltas))
            else:
                deltas = deltas[:len(df)]

        df["observation.img_state_delta"] = np.asarray(deltas, dtype=np.float32)
        try:
            df.to_parquet(pq, index=False)
            print(f"  - Updated img_state_delta: {pq.name}")
        except Exception as e:
            print(f"  ! 저장 실패: {pq} -> {e}")
    print("[2] 완료")

# ----------------------------- Step 3 ---------------------------------------
STATE_ACTION_NAMES = [
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "camera_yaw_joint","camera_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
    "L_pinky_proximal_joint","L_ring_proximal_joint","L_middle_proximal_joint","L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint","L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint","R_ring_proximal_joint","R_middle_proximal_joint","R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint","R_thumb_proximal_yaw_joint"
]

def probe_video_meta(sample_path: Path, default_fps: float):
    meta = {
        "width": None, "height": None, "fps": default_fps,
        "codec": None, "pix_fmt": None, "has_audio": False, "channels": 3
    }
    # 1) ffprobe
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(sample_path)],
            stderr=subprocess.STDOUT
        )
        info = json.loads(out)
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if vstreams:
            s = vstreams[0]
            meta["width"] = s.get("width")
            meta["height"] = s.get("height")
            meta["codec"] = s.get("codec_name")
            meta["pix_fmt"] = s.get("pix_fmt")
            r = s.get("r_frame_rate") or s.get("avg_frame_rate")
            if r and r != "0/0":
                try:
                    num, den = map(int, r.split("/"))
                    if den:
                        meta["fps"] = num / den
                except Exception:
                    pass
        meta["has_audio"] = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        if meta["width"] and meta["height"]:
            return meta
    except Exception:
        pass
    # 2) OpenCV
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(str(sample_path))
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_read = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if w and h:
                    meta["width"], meta["height"] = w, h
                if fps_read and fps_read > 0:
                    meta["fps"] = float(fps_read)
        except Exception:
            pass
    # 3) Defaults
    if not meta["width"]:   meta["width"]  = 640
    if not meta["height"]:  meta["height"] = 480
    if not meta["codec"]:   meta["codec"]  = "h264"
    if not meta["pix_fmt"]: meta["pix_fmt"]= "yuv420p"
    return meta

def step3_write_info_json(dataset_root: Path, view_keys: list, fps: int = 20, version: str = "v2.1"):
    data_root = dataset_root / "data"
    video_root = dataset_root / "videos"

    parquet_paths = rglob_parquets(data_root)
    total_episodes = len(parquet_paths)

    total_frames = 0
    task_indices = set()
    dtypes = {}
    sample_vectors = {}

    for pq in parquet_paths:
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"[3] Parquet 읽기 실패: {pq} -> {e}")
            continue
        total_frames += len(df)
        if "task_index" in df.columns:
            try:
                task_indices.update(df["task_index"].dropna().astype(int).unique())
            except Exception:
                pass
        for col, dt in df.dtypes.items():
            if col not in dtypes:
                dtypes[col] = str(dt)
            if col in ("observation.state", "action") and col not in sample_vectors:
                s = df[col].dropna()
                if len(s):
                    sample_vectors[col] = s.iloc[0]

    # video files for all view_keys
    total_videos = 0
    all_video_paths = []
    for view_key in view_keys:
        video_paths = list((video_root).rglob(f"{view_key}/*.mp4"))
        total_videos += len(video_paths)
        all_video_paths.extend(video_paths)

    # chunk info
    chunk_dirs = [d for d in (data_root.iterdir() if data_root.exists() else []) if d.is_dir() and d.name.startswith("chunk-")]
    total_chunks = len(chunk_dirs)
    chunks_size = total_episodes // total_chunks if total_chunks else total_episodes

    # features schema
    features = {}
    for col, dt in dtypes.items():
        feat = {"dtype": dt, "shape": [1], "names": col.split(".")[-1] if "." in col else None}
        if col in ("observation.state", "action"):
            target_len = None
            if col in sample_vectors:
                try:
                    arr = np.asarray(sample_vectors[col]).reshape(-1)
                    if arr.size > 1:
                        target_len = int(arr.size)
                except Exception:
                    pass
            if target_len is None:
                target_len = len(STATE_ACTION_NAMES)
            feat = {"dtype": "float64", "shape": [target_len], "names": STATE_ACTION_NAMES}
        features[col] = feat

    # Add features for each view_key
    for view_key in view_keys:
        view_video_paths = list((video_root).rglob(f"{view_key}/*.mp4"))
        if len(view_video_paths) > 0:
            vm = probe_video_meta(view_video_paths[0], default_fps=fps)
        else:
            vm = {"width": 640, "height": 480, "fps": fps, "codec": "h264", "pix_fmt": "yuv420p", "has_audio": False, "channels": 3}

        features[view_key] = {
            "dtype": "video",
            "shape": [int(vm["height"]), int(vm["width"]), int(vm.get("channels", 3))],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": int(vm["height"]),
                "video.width": int(vm["width"]),
                "video.codec": vm.get("codec") or "h264",
                "video.pix_fmt": vm.get("pix_fmt") or "yuv420p",
                "video.is_depth_map": False,
                "video.fps": float(vm.get("fps", fps)),
                "video.channels": int(vm.get("channels", 3)),
                "has_audio": bool(vm.get("has_audio", False))
            }
        }

    meta = {
        "codebase_version": version,
        "robot_type": None,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_indices),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": f"videos/chunk-{{episode_chunk:03d}}/{view_keys[0]}/episode_{{episode_index:06d}}.mp4",
        "video_paths": {view_key: f"videos/chunk-{{episode_chunk:03d}}/{view_key}/episode_{{episode_index:06d}}.mp4" for view_key in view_keys},
        "features": features,
        "discarded_episode_indices": []
    }

    out_path = dataset_root / "meta" / "info.json"
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"[3] ✅ info.json 생성: {out_path}")

# ----------------------------- Step 4 ---------------------------------------
def compute_stats_1d(arr: np.ndarray):
    return {
        "mean": [float(arr.mean())],
        "std":  [float(arr.std())],
        "min":  [float(arr.min())],
        "max":  [float(arr.max())],
        "q01":  [float(np.quantile(arr, 0.01))],
        "q99":  [float(np.quantile(arr, 0.99))],
    }

def compute_stats_nd(x: np.ndarray, axis=0):
    return {
        "mean": np.mean(x, axis=axis).tolist(),
        "std":  np.std(x, axis=axis).tolist(),
        "min":  np.min(x, axis=axis).tolist(),
        "max":  np.max(x, axis=axis).tolist(),
        "q01":  np.quantile(x, 0.01, axis=axis).tolist(),
        "q99":  np.quantile(x, 0.99, axis=axis).tolist(),
    }

def step4_write_stats_json(dataset_root: Path):
    data_root = dataset_root / "data"
    out_path = dataset_root / "meta" / "stats.json"
    ensure_dir(out_path.parent)

    files = rglob_parquets(data_root)
    if not files:
        print(f"[4] Parquet 없음: {data_root}")
        return

    acc = {
        "observation.state":    [],
        "action":               [],
        "observation.img_state_delta": [],
        "timestamp":            [],
        "frame_index":          [],
        "episode_index":        [],
        "index":                [],
        "task_index":           []
    }

    for pq in files:
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"[4] 읽기 실패: {pq} -> {e}")
            continue

        # multi-dim
        for col in ("observation.state", "action"):
            if col in df.columns:
                try:
                    acc[col].append(np.stack(df[col].values).astype(np.float32))
                except Exception:
                    # 혼합 길이/결측 등은 건너뜀
                    pass

        # scalars
        for key in ["observation.img_state_delta", "timestamp",
                    "frame_index", "episode_index", "index", "task_index"]:
            if key in df.columns:
                try:
                    acc[key].append(df[key].to_numpy())
                except Exception:
                    pass

    # concat
    for k, lst in list(acc.items()):
        if not lst:
            del acc[k]
            continue
        try:
            acc[k] = np.concatenate(lst, axis=0)
        except Exception:
            # 차원 불일치 등은 1D로 강제 시도
            try:
                acc[k] = np.concatenate([np.ravel(x) for x in lst], axis=0)
            except Exception:
                del acc[k]

    stats = {}
    for feature, data in acc.items():
        if data.ndim == 1:
            stats[feature] = compute_stats_1d(data)
        else:
            stats[feature] = compute_stats_nd(data, axis=0)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"[4] ✅ stats.json 생성: {out_path}")

# ----------------------------- Step 5 ---------------------------------------
# 파일 상단 import 근처에 추가
from typing import Optional, DefaultDict, Set

# ... 중략 ...

def step5_write_episodes_json(dataset_root: Path):
    """
    Save meta/episodes.json as a JSON ARRAY.
    Each item: {"episode_index": int, "tasks": [str,...], "length": int}
    """
    import re
    from collections import defaultdict

    data_root = dataset_root / "data"
    meta_dir  = dataset_root / "meta"
    ensure_dir(meta_dir)
    out_path  = meta_dir / "episodes.json"

    tasks_map = load_tasks_map(dataset_root)
    files = rglob_parquets(data_root)
    if not files:
        raise FileNotFoundError(f"[5] '{data_root}' 아래에서 Parquet 파일을 찾지 못했습니다.")

    def parse_episode_idx_from_filename(p: Path) -> Optional[int]:
        m = re.search(r"episode_(\d+)", p.stem)
        return int(m.group(1)) if m else None

    frames_by_ep: DefaultDict[int, Set[int]] = defaultdict(set)
    tasks_by_ep:  DefaultDict[int, Set[int]] = defaultdict(set)

    needed = ["episode_index", "task_index", "frame_index"]
    for pq in files:
        # 1) 필요한 컬럼만 먼저 시도 → 실패 시 전체 읽기
        try:
            df = pd.read_parquet(pq, columns=needed)
        except Exception:
            try:
                df = pd.read_parquet(pq)  # fallback
            except Exception as e2:
                print(f"[5] 읽기 실패: {pq} -> {e2}")
                continue

        # 2) episode_index 기준 집계 (없으면 파일명에서 폴백)
        if "episode_index" in df.columns:
            for ep_idx, g in df.groupby("episode_index"):
                try:
                    ep_idx = int(ep_idx)
                except Exception:
                    continue

                # 프레임 길이: frame_index가 있으면 고유값, 없으면 행 길이
                if "frame_index" in g.columns:
                    frames = pd.to_numeric(g["frame_index"], errors="coerce").dropna().astype(int).tolist()
                    frames_by_ep[ep_idx].update(frames)
                else:
                    frames_by_ep[ep_idx].update(range(len(g)))

                # 태스크 인덱스
                if "task_index" in g.columns:
                    tis = pd.to_numeric(g["task_index"], errors="coerce").dropna().astype(int).tolist()
                    tasks_by_ep[ep_idx].update(tis)
        else:
            ep_idx = parse_episode_idx_from_filename(pq)
            if ep_idx is None:
                print(f"[5] episode_index 컬럼/파일명 둘 다 없음: {pq} → 건너뜀")
                continue

            if "frame_index" in df.columns:
                frames = pd.to_numeric(df["frame_index"], errors="coerce").dropna().astype(int).tolist()
                frames_by_ep[ep_idx].update(frames)
            else:
                frames_by_ep[ep_idx].update(range(len(df)))

            if "task_index" in df.columns:
                tis = pd.to_numeric(df["task_index"], errors="coerce").dropna().astype(int).tolist()
                tasks_by_ep[ep_idx].update(tis)

    # 3) episodes.json 저장 (JSON 배열)
    rows = []
    for ep_idx in sorted(frames_by_ep.keys()):
        unique_task_idxs = sorted(tasks_by_ep.get(ep_idx, set()))
        tasks = [tasks_map.get(ti, f"task_index:{ti}") for ti in unique_task_idxs]
        rows.append({
            "episode_index": ep_idx,
            "tasks": tasks,
            "length": int(len(frames_by_ep[ep_idx])),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"[5] ✅ episodes.json 생성: {out_path} (총 {len(rows)} 에피소드)")

# ----------------------------- Step 6 ---------------------------------------
def step6_copy_modality_json(dataset_root: Path):
    """
    Copy modality.json from utils/parquet/ to dataset meta folder.
    """
    script_dir = Path(__file__).parent
    source_modality = script_dir / "modality.json"
    target_modality = dataset_root / "meta" / "modality.json"
    
    if source_modality.exists():
        import shutil
        shutil.copy2(source_modality, target_modality)
        print(f"[6] ✅ modality.json 복사: {target_modality}")
    else:
        print(f"[6] ⚠️ modality.json을 찾지 못했습니다: {source_modality}")

# ----------------------------- Orchestrator ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run steps 1~6 for dataset meta building.")
    parser.add_argument("--root", type=str, help="DATASET_ROOT (e.g., /abs/path/to/record/g1-pick-apple)")
    parser.add_argument("--task-index", type=int, help="Selected task_index to write in Step 1")
    parser.add_argument("--view-keys", type=str, nargs='+', 
                        default=["observation.images.ego_left_view", "observation.images.ego_right_view"],
                        help="Video view keys under videos/chunk-xxx/<VIEW_KEY> (supports multiple)")
    parser.add_argument("--primary-view-key", type=str, default="observation.images.ego_left_view",
                        help="Primary view key for img_state_delta calculation")
    parser.add_argument("--fps", type=int, default=20, help="FPS used in info.json probe fallback")
    parser.add_argument("--version", type=str, default="v2.1", help="codebase_version for info.json")
    args = parser.parse_args()

    dataset_root = None
    # 1) precedence: CLI arg
    if args.root:
        dataset_root = Path(args.root)
    # 2) env var
    if dataset_root is None:
        env = os.environ.get("DATASET_ROOT")
        if env:
            dataset_root = Path(env)
    # 3) UI
    if dataset_root is None:
        dataset_root = pick_dataset_root_via_ui()
    if dataset_root is None:
        print("DATASET_ROOT을 지정하지 않아 종료합니다.")
        sys.exit(1)
    if not dataset_root.exists():
        print(f"경로가 존재하지 않습니다: {dataset_root}")
        sys.exit(1)

    # task index
    if args.task_index is not None:
        selected_task_index = int(args.task_index)
    else:
        selected_task_index = pick_task_index_via_ui(dataset_root, default=0)
    print(f"DATASET_ROOT: {dataset_root}")
    print(f"VIEW_KEYS: {args.view_keys}")
    print(f"PRIMARY_VIEW_KEY: {args.primary_view_key}")
    print(f"Step1 task_index: {selected_task_index}")

    # Run steps
    step1_write_task_index(dataset_root, selected_task_index)
    step2_write_img_state_delta(dataset_root, args.primary_view_key)
    step3_write_info_json(dataset_root, args.view_keys, fps=args.fps, version=args.version)
    step4_write_stats_json(dataset_root)
    step5_write_episodes_json(dataset_root)
    step6_copy_modality_json(dataset_root)

    print("🎉 모든 단계 완료!")

if __name__ == "__main__":
    main()
