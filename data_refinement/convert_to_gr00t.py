#!/usr/bin/env python3
"""
convert_to_gr00t.py — G1_Teleop_Datacollection (60Hz LeRobot v2) → GR00T N1.7 학습 형식.

GR00T N1.7 공식 요구(getting_started/data_preparation.md + gr00t/data/dataset/
lerobot_episode_loader.py 정독 기반):
  meta/info.json  (features, data_path, video_path, chunks_size, fps)
  meta/episodes.jsonl  (줄별 {"episode_index", "tasks":[...], "length"})
  meta/tasks.jsonl     (줄별 {"task_index", "task"})
  meta/modality.json   (이미 수집 시 생성됨 — GR00T 형식과 일치 확인)
  meta/stats.json      (GR00T 의 gr00t/data/stats.py generate_stats 로 생성 — 본 스크립트는 안내만)
  parquet: observation.state/action(concat), timestamp, task_index, episode_index, index, frame_index
  videos: observation.images.<role>/episode_XXXXXX.mp4

핵심 설계 (사실 기반):
  - hz: 60→20fps 다운샘플 (3:1 정수배). action 은 각 시점의 *절대 관절각* 이라
    3프레임마다 1개 샘플링해도 각 프레임이 유효한 목표값 → 보간 없이 값 보존.
    (GR00T 가 학습 시 arm 을 RELATIVE 로 자동 변환, 배포 시 절대 복원 →
     Teleop 의 절대 관절각 제어와 정합. state_action_processor.py 정독 확인.)
  - 해상도: 640x360 유지 (N1.7 Cosmos-Reason2 백본이 native aspect ratio 지원, 패딩 없음).
  - video 다운샘플: ffmpeg select 필터로 parquet 와 *동일 인덱스* 프레임만 추출 → 영상-상태 정합.
  - 28D layout 은 modality.json 에서 동적으로 읽음 (수집 인자 무관 자동 대응).
  - 절대 관절각 그대로 저장 (relative 변환은 GR00T 가 학습/추론 시 처리).

사용법:
  conda activate teleop   # pyarrow/pandas/numpy 필요. ffmpeg(ffprobe) 필요.
  python convert_to_gr00t.py --src record/pick_test --out record_gr00t/pick_test \
      --task "Pick the object and place it on the plate." --src-fps 60 --tgt-fps 20

  이후 GR00T 레포에서:
  python -c "from gr00t.data.stats import generate_stats; generate_stats('record_gr00t/pick_test')"
  # (stats.json 생성. GR00T 로더가 필수로 요구.)
"""
import os
import sys
import glob
import json
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
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
        raise FileNotFoundError(f"modality.json 없음: {mp} (수집 시 자동 생성됐어야 함)")
    with open(mp) as f:
        return json.load(f)


def downsample_indices(n_frames, src_fps, tgt_fps):
    """정수배 다운샘플 인덱스. src_fps 가 tgt_fps 의 정수배여야 깔끔(예 60→20=3)."""
    if src_fps % tgt_fps != 0:
        print(f"  ⚠️ src_fps({src_fps})가 tgt_fps({tgt_fps})의 정수배가 아님. "
              f"가장 가까운 비율로 근사 샘플링합니다.")
        # 근사: 균일 간격 인덱스
        n_out = int(round(n_frames * tgt_fps / src_fps))
        return np.linspace(0, n_frames - 1, n_out).round().astype(int)
    step = src_fps // tgt_fps
    return np.arange(0, n_frames, step)


def video_role_dirs(src):
    """videos/chunk-000/observation.images.<role> 디렉토리 목록."""
    base = glob.glob(os.path.join(src, "videos", "chunk-*"))
    roles = {}
    for chunk in base:
        for d in sorted(os.listdir(chunk)):
            if d.startswith("observation.images."):
                roles.setdefault(d, os.path.join(chunk, d))
    return roles


def probe_video(path):
    """ffprobe 로 (w,h,fps,nb_frames)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames,avg_frame_rate",
             "-of", "json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        s = json.loads(out.stdout)["streams"][0]
        num, den = s.get("avg_frame_rate", "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        nb = s.get("nb_frames", "0")
        nb = int(nb) if str(nb).isdigit() else None
        return int(s["width"]), int(s["height"]), fps, nb
    except Exception as e:
        print(f"  ffprobe 실패({e})")
        return None, None, None, None


def downsample_video(src_mp4, dst_mp4, keep_idx, tgt_fps):
    """ffmpeg select 필터로 keep_idx 프레임만 추출 → tgt_fps 로 재인코딩.
    parquet 와 동일 인덱스를 뽑아 영상-상태 정합 보장."""
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    keep_set = set(int(i) for i in keep_idx)
    # select='eq(n,0)+eq(n,3)+...' 는 너무 길어질 수 있어, not(mod) 패턴 사용 (정수배일 때).
    # 일반화: keep_idx 가 등간격 step 이면 not(mod(n,step)) 으로 충분.
    diffs = np.diff(keep_idx)
    if len(diffs) > 0 and np.all(diffs == diffs[0]):
        step = int(diffs[0])
        vf = f"select='not(mod(n\\,{step}))',setpts=N/{tgt_fps}/TB"
    else:
        # 비등간격: 명시적 eq 나열 (프레임 수가 많지 않을 때만)
        expr = "+".join(f"eq(n\\,{i})" for i in keep_idx)
        vf = f"select='{expr}',setpts=N/{tgt_fps}/TB"
    cmd = [
        "ffmpeg", "-y", "-i", src_mp4,
        "-vf", vf, "-vsync", "0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-x264-params", "keyint=1",     # 모든 프레임 키프레임(랜덤 액세스, 학습 디코딩 안정)
        "-r", str(tgt_fps),
        dst_mp4,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"  ⚠️ ffmpeg 실패: {src_mp4}\n{r.stderr[-400:]}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="입력 LeRobot 데이터셋 루트 (record/<task>)")
    ap.add_argument("--out", required=True, help="출력 GR00T 데이터셋 루트")
    ap.add_argument("--task", required=True, help="language task description (tasks.jsonl 에 기록)")
    ap.add_argument("--src-fps", type=float, default=60.0)
    ap.add_argument("--tgt-fps", type=float, default=20.0)
    ap.add_argument("--robot-type", default="Unitree_G1")
    ap.add_argument("--codebase-version", default="v2.1")
    ap.add_argument("--slim-cols", action="store_true",
                    help="GR00T 학습 핵심 컬럼만 남겨 경량화(선택). 기본은 원본 전체 컬럼 유지 "
                         "(NVIDIA GR00T G1 데이터도 부가 컬럼 보유, 로더가 modality 키만 읽음).")
    args = ap.parse_args()

    src, out = args.src, args.out
    src_fps, tgt_fps = int(args.src_fps), int(args.tgt_fps)

    parquets = list_episode_parquets(src)
    if not parquets:
        print(f"[오류] {src}/data/chunk-*/ 에 parquet 없음"); sys.exit(1)
    modality = load_modality(src)
    roles = video_role_dirs(src)

    hr(f"GR00T 변환: {src} → {out}  ({src_fps}→{tgt_fps}fps)")
    print(f"  에피소드 {len(parquets)}개, 카메라 {list(roles.keys())}")
    print(f"  task: {args.task!r}")

    # 출력 디렉토리
    out_data = os.path.join(out, "data", "chunk-000")
    out_meta = os.path.join(out, "meta")
    os.makedirs(out_data, exist_ok=True)
    os.makedirs(out_meta, exist_ok=True)

    episodes_meta = []      # episodes.jsonl 행
    total_frames = 0
    state_dim = action_dim = None
    state_dtype = action_dtype = "float32"  # 첫 에피소드에서 실제값으로 갱신
    global_index = 0        # 전체 프레임 통합 index

    for pq in parquets:
        ep = ep_idx_from_path(pq)
        df = pd.read_parquet(pq, engine="pyarrow")
        n = len(df)
        keep = downsample_indices(n, src_fps, tgt_fps)
        df2 = df.iloc[keep].reset_index(drop=True)

        # timestamp/frame_index/index/episode_index/task_index 재계산
        m = len(df2)
        df2["timestamp"]     = (np.arange(m) / tgt_fps).astype(np.float32)
        df2["frame_index"]   = np.arange(m, dtype=np.int64)
        df2["episode_index"] = np.int64(ep)
        df2["index"]         = np.arange(global_index, global_index + m, dtype=np.int64)
        df2["task_index"]    = np.int64(0)   # 단일 task. 다중 task 면 매핑 필요.
        global_index += m

        if state_dim is None:
            state_dim  = int(np.asarray(df2["observation.state"].iloc[0]).size)
            action_dim = int(np.asarray(df2["action"].iloc[0]).size)
            # 실제 저장 dtype 기록 (info.json features 에 반영 — 실데이터와 일치).
            state_dtype  = str(np.asarray(df2["observation.state"].iloc[0]).dtype)
            action_dtype = str(np.asarray(df2["action"].iloc[0]).dtype)

        # ---- GR00T 형식 정리 (옵션) -----------------------------------------
        # dtype: 원본(float64) 유지. 근거 — NVIDIA GR00T G1 데이터셋
        #   (nvidia/Arena-G1-Loco-Manipulation-Task, nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1)
        #   은 observation.state/action 을 FP64(float64) 로 저장. data_preparation.md 의
        #   "float32" 는 권장 표현이며 강제 아님(로더 lerobot_episode_loader.py 는 info dtype 을
        #   검증하지 않고 L427 에서 video 키만 확인). 따라서 float64 유지가 GR00T G1 실데이터와 일치.
        #
        # 컬럼: 기본은 *원본 전체 컬럼 유지* (GR00T G1 데이터도 annotation 등 부가 컬럼 보유).
        #   로더는 modality 키만 선택적으로 읽어 나머지는 무시하므로 유지/제거 모두 학습 무해.
        #   --slim-cols 지정 시 GR00T 학습용 핵심 컬럼만 남겨 경량화(선택).
        if args.slim_cols:
            gr00t_cols = ["observation.state", "action", "timestamp",
                          "task_index", "episode_index", "index", "frame_index"]
            keep_cols = [c for c in gr00t_cols if c in df2.columns]
            df2 = df2[keep_cols]

        # parquet 저장
        out_pq = os.path.join(out_data, f"episode_{ep:06d}.parquet")
        df2.to_parquet(out_pq, engine="pyarrow", index=False)

        # video 다운샘플 (동일 keep 인덱스)
        for role, rdir in roles.items():
            src_mp4 = os.path.join(rdir, f"episode_{ep:06d}.mp4")
            if not os.path.exists(src_mp4):
                print(f"  [skip] {role} episode_{ep:06d}.mp4 없음")
                continue
            dst_mp4 = os.path.join(out, "videos", "chunk-000", role, f"episode_{ep:06d}.mp4")
            downsample_video(src_mp4, dst_mp4, keep, tgt_fps)

        episodes_meta.append({"episode_index": ep, "tasks": [args.task], "length": m})
        total_frames += m
        print(f"  ep {ep:06d}: {n} → {m} 프레임")

    # ---- meta/tasks.jsonl (줄별) --------------------------------------------
    with open(os.path.join(out_meta, "tasks.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}, ensure_ascii=False) + "\n")

    # ---- meta/episodes.jsonl (줄별) -----------------------------------------
    with open(os.path.join(out_meta, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for row in episodes_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- meta/modality.json (복사) ------------------------------------------
    with open(os.path.join(out_meta, "modality.json"), "w", encoding="utf-8") as f:
        json.dump(modality, f, indent=4, ensure_ascii=False)

    # ---- meta/info.json -----------------------------------------------------
    # video 메타 프로브 (출력본 1개)
    vid_features = {}
    first_ep = ep_idx_from_path(parquets[0])
    for role in roles:
        sample = os.path.join(out, "videos", "chunk-000", role, f"episode_{first_ep:06d}.mp4")
        if os.path.exists(sample):
            w, h, fps, nb = probe_video(sample)
        else:
            w, h, fps = 640, 360, tgt_fps
        vid_features[role] = {
            "dtype": "video",
            "shape": [h or 360, w or 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": h or 360, "video.width": w or 640,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": float(tgt_fps),
                "video.channels": 3, "has_audio": False,
            },
        }

    # state/action names: modality.json 의 키 순서대로 평탄화 (GR00T 는 names 를 강제하지 않으나
    # LeRobot info.json 관례상 기록. 차원만 정확하면 로더는 modality.json 으로 분할함.)
    def flat_names(section):
        names = []
        for key, rng in section.items():
            for i in range(rng["start"], rng["end"]):
                names.append(f"{key}_{i - rng['start']}")
        return names

    features = {
        "observation.state": {"dtype": state_dtype, "shape": [state_dim],
                              "names": flat_names(modality["state"])},
        "action":            {"dtype": action_dtype, "shape": [action_dim],
                              "names": flat_names(modality["action"])},
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
        "total_tasks": 1,
        "total_videos": len(episodes_meta) * len(roles),
        "total_chunks": 1,
        "chunks_size": 1000,
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
    print(f"  state_dim={state_dim}, action_dim={action_dim}, fps={tgt_fps}")
    print(f"  에피소드 {len(episodes_meta)}개, 총 {total_frames}프레임")
    print(f"  생성: info.json, episodes.jsonl, tasks.jsonl, modality.json")
    print()
    print("  ⚠️ 다음 단계 — stats.json 생성 (GR00T 로더 필수):")
    print("     GR00T 레포에서:")
    print(f"     python -c \"from gr00t.data.stats import generate_stats; generate_stats('{out}')\"")
    print("     (mean/std/min/max/q01/q99 형식. GR00T 가 정규화에 사용.)")


if __name__ == "__main__":
    main()
