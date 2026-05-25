#!/usr/bin/env python3
"""
convert_to_gr00t_multitask.py — 여러 task 폴더를 하나의 GR00T N1.7 데이터셋으로 병합.

배경 (실제 GR00T 코드 정독):
  - 본인 annotation key = "human.task_description", original_key="task_index".
    로더(lerobot_episode_loader.py L380-381)는
        original_df["task_index"].apply(lambda x: tasks_map[x])
    로 *각 프레임의 parquet task_index* 를 tasks.jsonl 의 instruction 으로 매핑.
  - 따라서 다중 task 학습은:
      1) task 별 고유 task_index (0,1,2...) 부여
      2) 각 폴더 parquet 의 task_index 를 그 값으로 설정
      3) tasks.jsonl 에 모든 task 줄별 기록 ({task_index, task})
      4) episode_index 를 폴더 간 전역 연속 번호로 재부여 (충돌 방지)
      5) chunk 는 전역 episode_index // chunks_size 로 계산 (로더 L360 과 일치)

입력: tasks 정의 JSON (또는 --task-spec 반복). 각 항목:
    {"src": "record/pick_apple", "task": "Pick the apple and place it on the plate."}

사용:
  conda activate teleop
  # 방법 1: JSON 스펙 파일
  python convert_to_gr00t_multitask.py --spec tasks_spec.json \
      --out record_gr00t/multitask --src-fps 60 --tgt-fps 20
  # tasks_spec.json:
  #   [
  #     {"src": "record/pick_apple",  "task": "Pick the apple and place it on the plate."},
  #     {"src": "record/pick_pear",   "task": "Pick the pear and place it on the plate."},
  #     {"src": "record/open_drawer", "task": "Open the drawer."}
  #   ]

  # 방법 2: CLI 반복 (--task-spec "폴더::instruction")
  python convert_to_gr00t_multitask.py \
      --task-spec "record/pick_apple::Pick the apple and place it on the plate." \
      --task-spec "record/pick_pear::Pick the pear and place it on the plate." \
      --out record_gr00t/multitask --src-fps 60 --tgt-fps 20

  이후 (GR00T 레포):
  python -m gr00t.data.stats --dataset-path <out> \
      --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/G1_DEX3/g1_dex3_config.py
"""
import os
import sys
import glob
import json
import argparse
import subprocess
import numpy as np
import pandas as pd

CHUNK_SIZE = 1000


def hr(t=""):
    print("\n" + "=" * 76)
    if t:
        print(t); print("=" * 76)


def list_episode_parquets(src):
    return sorted(glob.glob(os.path.join(src, "data", "chunk-*", "episode_*.parquet")))


def ep_idx_from_path(p):
    import re
    m = re.search(r"episode_(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else None


def load_modality(src):
    mp = os.path.join(src, "meta", "modality.json")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"modality.json 없음: {mp}")
    with open(mp) as f:
        return json.load(f)


def downsample_indices(n, src_fps, tgt_fps):
    if src_fps % tgt_fps != 0:
        n_out = int(round(n * tgt_fps / src_fps))
        return np.linspace(0, n - 1, n_out).round().astype(int)
    return np.arange(0, n, src_fps // tgt_fps)


def video_role_dirs(src):
    roles = {}
    for chunk in glob.glob(os.path.join(src, "videos", "chunk-*")):
        for d in sorted(os.listdir(chunk)):
            if d.startswith("observation.images."):
                roles.setdefault(d, os.path.join(chunk, d))
    return roles


def probe_video(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate",
             "-of", "json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        s = json.loads(out.stdout)["streams"][0]
        return int(s["width"]), int(s["height"])
    except Exception:
        return None, None


def downsample_video(src_mp4, dst_mp4, keep_idx, tgt_fps):
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    diffs = np.diff(keep_idx)
    if len(diffs) > 0 and np.all(diffs == diffs[0]):
        step = int(diffs[0])
        vf = f"select='not(mod(n\\,{step}))',setpts=N/{tgt_fps}/TB"
    else:
        expr = "+".join(f"eq(n\\,{i})" for i in keep_idx)
        vf = f"select='{expr}',setpts=N/{tgt_fps}/TB"
    cmd = ["ffmpeg", "-y", "-i", src_mp4, "-vf", vf, "-vsync", "0",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-x264-params", "keyint=1", "-r", str(tgt_fps), dst_mp4]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"  ⚠️ ffmpeg 실패: {src_mp4}\n{r.stderr[-300:]}")
        return False
    return True


def parse_specs(args):
    """tasks 스펙 → [(src, instruction), ...]"""
    specs = []
    if args.spec:
        with open(args.spec, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            specs.append((item["src"], item["task"]))
    for ts in (args.task_spec or []):
        if "::" not in ts:
            print(f"[오류] --task-spec 형식은 '폴더::instruction': {ts}"); sys.exit(1)
        src, instr = ts.split("::", 1)
        specs.append((src.strip(), instr.strip()))
    if not specs:
        print("[오류] --spec 또는 --task-spec 으로 task 를 하나 이상 지정하세요."); sys.exit(1)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="tasks 정의 JSON 파일 ([{src, task}, ...])")
    ap.add_argument("--task-spec", action="append",
                    help="'폴더::instruction' 형식, 반복 가능")
    ap.add_argument("--out", required=True, help="출력 GR00T 데이터셋 루트")
    ap.add_argument("--src-fps", type=float, default=60.0)
    ap.add_argument("--tgt-fps", type=float, default=20.0)
    ap.add_argument("--robot-type", default="Unitree_G1")
    ap.add_argument("--codebase-version", default="v2.1")
    ap.add_argument("--slim-cols", action="store_true",
                    help="GR00T 핵심 컬럼만 남겨 경량화(선택).")
    args = ap.parse_args()

    specs = parse_specs(args)
    src_fps, tgt_fps = int(args.src_fps), int(args.tgt_fps)
    out = args.out
    out_meta = os.path.join(out, "meta")
    os.makedirs(out_meta, exist_ok=True)

    hr(f"GR00T 다중 task 병합 → {out}  ({src_fps}→{tgt_fps}fps)")
    for ti, (src, instr) in enumerate(specs):
        print(f"  task_index {ti}: {src}  ::  {instr!r}")

    # 모든 task 의 modality.json 이 동일해야 함 (같은 로봇 구성) — 첫 것 기준 + 검증
    modality0 = load_modality(specs[0][0])

    episodes_meta = []
    tasks_jsonl = []
    total_frames = 0
    state_dim = action_dim = None
    state_dtype = action_dtype = "float32"
    global_index = 0      # 전체 프레임 통합 index
    global_ep = 0         # 전역 episode_index
    roles_union = {}      # 카메라 role (첫 task 기준)

    for ti, (src, instr) in enumerate(specs):
        tasks_jsonl.append({"task_index": ti, "task": instr})

        # modality.json 일치 검증 (다른 로봇 구성 섞임 방지)
        mod = load_modality(src)
        if mod.get("state") != modality0.get("state") or mod.get("action") != modality0.get("action"):
            print(f"  ⚠️ {src} 의 modality state/action 분할이 첫 task 와 다릅니다. "
                  f"같은 로봇 구성인지 확인 필요.")

        parquets = list_episode_parquets(src)
        if not parquets:
            print(f"  [오류] {src} 에 parquet 없음"); sys.exit(1)
        roles = video_role_dirs(src)
        if ti == 0:
            roles_union = roles

        print(f"\n  [task {ti}] {src}: 에피소드 {len(parquets)}개")
        for pq in parquets:
            local_ep = ep_idx_from_path(pq)
            df = pd.read_parquet(pq, engine="pyarrow")
            n = len(df)
            keep = downsample_indices(n, src_fps, tgt_fps)
            df2 = df.iloc[keep].reset_index(drop=True)
            m = len(df2)

            # 메타 컬럼 재계산 — task_index = ti, episode_index = 전역
            df2["timestamp"]     = (np.arange(m) / tgt_fps).astype(np.float32)
            df2["frame_index"]   = np.arange(m, dtype=np.int64)
            df2["episode_index"] = np.int64(global_ep)
            df2["index"]         = np.arange(global_index, global_index + m, dtype=np.int64)
            df2["task_index"]    = np.int64(ti)
            global_index += m

            if state_dim is None:
                state_dim  = int(np.asarray(df2["observation.state"].iloc[0]).size)
                action_dim = int(np.asarray(df2["action"].iloc[0]).size)
                state_dtype  = str(np.asarray(df2["observation.state"].iloc[0]).dtype)
                action_dtype = str(np.asarray(df2["action"].iloc[0]).dtype)

            if args.slim_cols:
                cols = ["observation.state", "action", "timestamp",
                        "task_index", "episode_index", "index", "frame_index"]
                df2 = df2[[c for c in cols if c in df2.columns]]

            # 출력 (전역 episode_index 기반 chunk)
            chunk_idx = global_ep // CHUNK_SIZE
            out_data = os.path.join(out, "data", f"chunk-{chunk_idx:03d}")
            os.makedirs(out_data, exist_ok=True)
            df2.to_parquet(os.path.join(out_data, f"episode_{global_ep:06d}.parquet"),
                           engine="pyarrow", index=False)

            # 비디오 (동일 keep, 전역 episode_index, src 의 local_ep 에서 읽음)
            for role, rdir in roles.items():
                src_mp4 = os.path.join(rdir, f"episode_{local_ep:06d}.mp4")
                if not os.path.exists(src_mp4):
                    print(f"    [skip] {role} local ep{local_ep:06d}.mp4 없음")
                    continue
                dst_mp4 = os.path.join(out, "videos", f"chunk-{chunk_idx:03d}", role,
                                       f"episode_{global_ep:06d}.mp4")
                downsample_video(src_mp4, dst_mp4, keep, tgt_fps)

            episodes_meta.append({"episode_index": global_ep, "tasks": [instr], "length": m})
            total_frames += m
            print(f"    local ep{local_ep:06d} → 전역 ep{global_ep:06d}: {n}→{m} 프레임 (task_index={ti})")
            global_ep += 1

    # ---- meta 파일들 --------------------------------------------------------
    with open(os.path.join(out_meta, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for t in tasks_jsonl:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(out_meta, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for e in episodes_meta:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(out_meta, "modality.json"), "w", encoding="utf-8") as f:
        json.dump(modality0, f, indent=4, ensure_ascii=False)

    # video features (첫 에피소드 프로브)
    vid_features = {}
    for role in roles_union:
        sample = os.path.join(out, "videos", "chunk-000", role, "episode_000000.mp4")
        w, h = probe_video(sample) if os.path.exists(sample) else (640, 360)
        vid_features[role] = {
            "dtype": "video", "shape": [h or 360, w or 640, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.height": h or 360, "video.width": w or 640,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "video.fps": float(tgt_fps),
                     "video.channels": 3, "has_audio": False},
        }

    def flat_names(section):
        names = []
        for key, rng in section.items():
            for i in range(rng["start"], rng["end"]):
                names.append(f"{key}_{i - rng['start']}")
        return names

    features = {
        "observation.state": {"dtype": state_dtype, "shape": [state_dim],
                              "names": flat_names(modality0["state"])},
        "action":            {"dtype": action_dtype, "shape": [action_dim],
                              "names": flat_names(modality0["action"])},
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index":         {"dtype": "int64", "shape": [1], "names": None},
        "task_index":    {"dtype": "int64", "shape": [1], "names": None},
    }
    features.update(vid_features)

    info = {
        "codebase_version": args.codebase_version,
        "robot_type": args.robot_type,
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(tasks_jsonl),
        "total_videos": len(episodes_meta) * len(roles_union),
        "total_chunks": (max(e["episode_index"] for e in episodes_meta) // CHUNK_SIZE) + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": tgt_fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(os.path.join(out_meta, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    hr("완료")
    print(f"  출력: {out}")
    print(f"  task {len(tasks_jsonl)}개, 에피소드 {len(episodes_meta)}개, 총 {total_frames}프레임")
    print(f"  state_dim={state_dim}, action_dim={action_dim}, fps={tgt_fps}")
    print(f"  tasks.jsonl:")
    for t in tasks_jsonl:
        print(f"    task_index {t['task_index']}: {t['task']!r}")
    print()
    print("  ⚠️ 다음 — stats 생성 (GR00T 레포):")
    print(f"     python -m gr00t.data.stats --dataset-path {out} \\")
    print(f"         --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/G1_DEX3/g1_dex3_config.py")


if __name__ == "__main__":
    main()
