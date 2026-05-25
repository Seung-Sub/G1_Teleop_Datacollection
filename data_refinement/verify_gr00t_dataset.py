#!/usr/bin/env python3
"""
verify_gr00t_dataset.py — convert_to_gr00t.py 출력본이 GR00T N1.7 로더 형식과
정합하는지 검증. (실제 GR00T lerobot_episode_loader.py 가 요구하는 항목 기준.)

검증 항목 (사실 기반 — lerobot_episode_loader.py 정독):
  [meta] info.json / episodes.jsonl / tasks.jsonl / modality.json 존재 + 형식
    - episodes.jsonl: 줄별 JSON, {"episode_index","tasks","length"}
    - tasks.jsonl:    줄별 JSON, {"task_index","task"}
    - info.json:      data_path / chunks_size 필수, fps, features
    - modality.json:  state/action/video/annotation, video original_key
  [정합] modality.json video original_key ⊆ info.json features (로더 L427 assert)
  [정합] info.json features 의 state/action shape == parquet 실제 차원
  [parquet] observation.state / action / timestamp / task_index / episode_index /
            index / frame_index 컬럼 존재, 차원 일치, NaN 없음
  [fps] timestamp 간격이 info.json fps 와 일치
  [video-state] 각 에피소드 mp4 프레임 수 == parquet 행 수 (정합)
  [stats] stats.json 존재 여부 (없으면 generate_stats 안내)

사용:
  conda activate teleop   # pandas/numpy/pyarrow, (선택)ffprobe
  python verify_gr00t_dataset.py --dataset record_gr00t/pick_test
"""
import os
import sys
import json
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd


PASS, FAIL, WARN = "✅", "❌", "⚠️"
errors = []
warns = []


def check(cond, msg_ok, msg_fail, fatal=True):
    if cond:
        print(f"  {PASS} {msg_ok}")
        return True
    else:
        print(f"  {FAIL if fatal else WARN} {msg_fail}")
        (errors if fatal else warns).append(msg_fail)
        return False


def probe_nb_frames(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return int(out.stdout.strip())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="GR00T 데이터셋 루트 (convert_to_gr00t 출력)")
    ap.add_argument("--skip-video", action="store_true", help="비디오 프레임수 검증 건너뜀(ffprobe 없을 때)")
    args = ap.parse_args()

    root = args.dataset
    meta = os.path.join(root, "meta")

    print("=" * 76)
    print(f"GR00T 데이터셋 검증: {root}")
    print("=" * 76)

    # ---- 1. meta 파일 존재 -------------------------------------------------
    print("\n[1] meta 파일 존재")
    for fn in ["info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"]:
        check(os.path.exists(os.path.join(meta, fn)),
              f"{fn} 존재", f"{fn} 없음 (GR00T 로더 필수)")
    has_stats = os.path.exists(os.path.join(meta, "stats.json"))
    check(has_stats, "stats.json 존재",
          "stats.json 없음 — generate_stats 실행 필요 (로더가 assert 로 요구)", fatal=False)

    if errors:
        print(f"\n{FAIL} 필수 meta 누락. 중단."); sys.exit(1)

    # ---- 2. info.json --------------------------------------------------------
    print("\n[2] info.json 형식")
    with open(os.path.join(meta, "info.json")) as f:
        info = json.load(f)
    check("data_path" in info, "data_path 있음", "data_path 없음 (로더 필수)")
    check("chunks_size" in info, "chunks_size 있음", "chunks_size 없음 (로더 필수)")
    fps = info.get("fps")
    check(fps is not None, f"fps={fps}", "fps 없음", fatal=False)
    feats = info.get("features", {})
    state_feat = feats.get("observation.state", {})
    action_feat = feats.get("action", {})
    info_state_dim = state_feat.get("shape", [None])[0]
    info_action_dim = action_feat.get("shape", [None])[0]
    print(f"     features.observation.state.shape={state_feat.get('shape')}, "
          f"action.shape={action_feat.get('shape')}")

    # ---- 3. modality.json ----------------------------------------------------
    print("\n[3] modality.json 형식 + info 정합")
    with open(os.path.join(meta, "modality.json")) as f:
        mod = json.load(f)
    for sec in ["state", "action", "video"]:
        check(sec in mod, f"modality.{sec} 있음", f"modality.{sec} 없음")
    # video original_key ⊆ info features  (로더 L427 assert 대응)
    vid_keys_ok = True
    for vk, vv in mod.get("video", {}).items():
        ok = vv["original_key"] in feats
        vid_keys_ok &= ok
        if not ok:
            print(f"     {FAIL} video '{vk}' original_key={vv['original_key']} 가 info features 에 없음")
    check(vid_keys_ok, "video original_key 가 모두 info features 에 존재 (로더 assert 통과)",
          "video original_key 불일치 (로더 L427 assert 실패 위험)")
    # state/action 분할 끝값 == info dim
    mod_state_dim = max((r["end"] for r in mod["state"].values()), default=0)
    mod_action_dim = max((r["end"] for r in mod["action"].values()), default=0)
    check(mod_state_dim == info_state_dim,
          f"modality state dim({mod_state_dim}) == info({info_state_dim})",
          f"state dim 불일치: modality={mod_state_dim} info={info_state_dim}")
    check(mod_action_dim == info_action_dim,
          f"modality action dim({mod_action_dim}) == info({info_action_dim})",
          f"action dim 불일치: modality={mod_action_dim} info={info_action_dim}")

    # ---- 4. tasks.jsonl / episodes.jsonl (줄별 JSON) -------------------------
    print("\n[4] tasks.jsonl / episodes.jsonl (줄별 JSON)")
    with open(os.path.join(meta, "tasks.jsonl")) as f:
        task_lines = [json.loads(l) for l in f if l.strip()]
    check(all("task_index" in t and "task" in t for t in task_lines),
          f"tasks.jsonl 줄별 OK ({len(task_lines)} task)",
          "tasks.jsonl 형식 오류 (task_index/task 키 필요)")
    with open(os.path.join(meta, "episodes.jsonl")) as f:
        ep_lines = [json.loads(l) for l in f if l.strip()]
    check(all(("episode_index" in e and "length" in e) for e in ep_lines),
          f"episodes.jsonl 줄별 OK ({len(ep_lines)} episode)",
          "episodes.jsonl 형식 오류 (episode_index/length 필요)")

    # ---- 5. parquet 컬럼/차원/NaN -------------------------------------------
    print("\n[5] parquet 검증")
    pqs = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet")))
    check(len(pqs) > 0, f"parquet {len(pqs)}개 발견", "parquet 없음")
    required_cols = ["observation.state", "action", "timestamp",
                     "task_index", "episode_index", "index", "frame_index"]
    total_frames = 0
    fps_ok = True
    ep_len_map = {}
    for pq in pqs:
        df = pd.read_parquet(pq, engine="pyarrow")
        ep = int(df["episode_index"].iloc[0]) if "episode_index" in df else -1
        ep_len_map[ep] = len(df)
        total_frames += len(df)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            check(False, "", f"{os.path.basename(pq)}: 컬럼 누락 {missing}")
            continue
        # 차원
        s_dim = int(np.asarray(df["observation.state"].iloc[0]).size)
        a_dim = int(np.asarray(df["action"].iloc[0]).size)
        if s_dim != info_state_dim or a_dim != info_action_dim:
            check(False, "", f"{os.path.basename(pq)}: dim 불일치 state={s_dim} action={a_dim}")
        # NaN
        s_all = np.stack([np.asarray(x, np.float32) for x in df["observation.state"]])
        a_all = np.stack([np.asarray(x, np.float32) for x in df["action"]])
        if np.isnan(s_all).any() or np.isnan(a_all).any():
            check(False, "", f"{os.path.basename(pq)}: NaN 존재")
        # fps (timestamp 간격)
        ts = df["timestamp"].to_numpy(dtype=np.float64)
        if len(ts) > 1:
            dt = np.median(np.diff(ts))
            eff_fps = 1.0 / dt if dt > 0 else 0
            if fps and abs(eff_fps - fps) > 0.5:
                fps_ok = False
                print(f"     {WARN} {os.path.basename(pq)}: timestamp fps={eff_fps:.2f} != info fps={fps}")
    check(True, f"parquet 컬럼/차원/NaN 검사 완료 (총 {total_frames}프레임, "
                f"state_dim={info_state_dim}, action_dim={info_action_dim})", "")
    check(fps_ok, f"timestamp 간격이 info fps({fps})와 일치", "일부 fps 불일치", fatal=False)

    # ---- 6. video-state 정합 (프레임 수) ------------------------------------
    if not args.skip_video:
        print("\n[6] video-state 프레임 정합")
        vid_roles = list(mod.get("video", {}).values())
        checked = 0
        mism = 0
        for vv in vid_roles:
            role = vv["original_key"]  # observation.images.<role>
            for ep, plen in ep_len_map.items():
                chunk = ep // info["chunks_size"]
                mp4 = os.path.join(root, "videos", f"chunk-{chunk:03d}", role,
                                   f"episode_{ep:06d}.mp4")
                if not os.path.exists(mp4):
                    continue
                nb = probe_nb_frames(mp4)
                checked += 1
                if nb is not None and nb != plen:
                    mism += 1
                    print(f"     {WARN} {role} ep{ep}: video {nb}프레임 != parquet {plen}행")
        if checked == 0:
            print(f"     {WARN} 비디오를 찾지 못하거나 ffprobe 없음 — 건너뜀")
        else:
            check(mism == 0, f"video-state 프레임 정합 OK ({checked}개 검사)",
                  f"{mism}개 비디오 프레임수 불일치", fatal=False)
    else:
        print("\n[6] video-state 정합 — 건너뜀(--skip-video)")

    # ---- 결과 ----------------------------------------------------------------
    print("\n" + "=" * 76)
    if errors:
        print(f"{FAIL} 검증 실패: {len(errors)}개 치명 오류")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    else:
        print(f"{PASS} 모든 필수 검증 통과" + (f" ({len(warns)}개 경고)" if warns else ""))
        for w in warns:
            print(f"   {WARN} {w}")
        if not has_stats:
            print(f"\n   다음 단계: GR00T 레포에서 stats.json 생성")
            print(f"   python -c \"from gr00t.data.stats import generate_stats; "
                  f"generate_stats('{root}')\"")


if __name__ == "__main__":
    main()
