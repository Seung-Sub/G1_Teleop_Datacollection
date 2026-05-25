#!/usr/bin/env python3
"""
저장된 에피소드 종합 검증 — 녹화 1회 후 parquet + MP4 가 의도대로 저장됐는지 정량 점검.

검증 항목:
  1) parquet 구조: 행 수, 컬럼, observation.state / action / observation.sensor 차원
  2) 60Hz 정렬: timestamp 간격이 ~16.67ms(60Hz) 인지, 행 수 ≈ 60 × 길이
  3) raw_ts 메타: axis_ts_ns + raw_ts_* 존재, 각 모달리티 raw_ts 로 실제 수집 hz 역산
     (raw 가 60Hz 이상이었는지 = 업샘플 없이 다운/매칭됐는지 확인)
  4) MP4: ego/wrist_l/wrist_r 3개 view 존재, 해상도 640x360, fps 60,
     프레임 수 = parquet 행 수 (영상-상태 정합)
  5) 보간 sanity: NaN/inf 없음, 연속신호(arm) 부드러움 vs action 특성, 값 범위 타당성

사용법:
  cd ~/G1_Teleop_Datacollection
  conda activate teleop
  python verify_episode.py                          # record/ 에서 최신 task/episode 자동탐색
  python verify_episode.py record/<task>/data/chunk-000/episode_000000.parquet
  python verify_episode.py --task <task_name> --ep 0

ffprobe(ffmpeg) 가 있으면 MP4 메타를 정확히 읽고, 없으면 imageio 로 대체.
"""
import os
import sys
import glob
import json
import subprocess
import argparse
import numpy as np
import pandas as pd


def hr(title=""):
    print("\n" + "=" * 76)
    if title:
        print(title)
        print("=" * 76)


def find_latest_parquet(base="record"):
    """record/<task>/data/chunk-*/episode_*.parquet 중 최신(mtime) 반환."""
    cands = glob.glob(os.path.join(base, "*", "data", "chunk-*", "episode_*.parquet"))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def parquet_to_video_dir(parquet_path):
    """data/chunk-XXX/episode_YYY.parquet → videos/chunk-XXX (같은 task 아래)."""
    # .../<task>/data/chunk-000/episode_000000.parquet
    p = os.path.abspath(parquet_path)
    chunk_dir = os.path.dirname(p)                 # chunk-000
    chunk_name = os.path.basename(chunk_dir)       # chunk-000
    data_dir = os.path.dirname(chunk_dir)          # data
    task_dir = os.path.dirname(data_dir)           # <task>
    ep_name = os.path.basename(p)                  # episode_000000.parquet
    ep_mp4 = ep_name.replace(".parquet", ".mp4")
    video_chunk = os.path.join(task_dir, "videos", chunk_name)
    return video_chunk, ep_mp4


def probe_video(path):
    """ffprobe 로 (width, height, nb_frames, avg_fps) 반환. 실패 시 imageio fallback."""
    # ffprobe 시도
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate",
             "-of", "json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        info = json.loads(out.stdout)["streams"][0]
        w = int(info.get("width", 0))
        h = int(info.get("height", 0))
        nb = info.get("nb_frames", "0")
        nb = int(nb) if str(nb).isdigit() else None
        # avg_frame_rate 는 "60/1" 형태
        def _parse_fps(s):
            try:
                num, den = s.split("/")
                return float(num) / float(den) if float(den) != 0 else 0.0
            except Exception:
                return 0.0
        fps = _parse_fps(info.get("avg_frame_rate", "0/1")) or _parse_fps(info.get("r_frame_rate", "0/1"))
        return w, h, nb, fps, "ffprobe"
    except Exception:
        pass
    # imageio fallback
    try:
        import imageio
        rdr = imageio.get_reader(path, format="FFMPEG")
        meta = rdr.get_meta_data()
        fps = float(meta.get("fps", 0))
        # 프레임 수 직접 카운트 (느릴 수 있음)
        nb = rdr.count_frames() if hasattr(rdr, "count_frames") else None
        first = rdr.get_data(0)
        h, w = first.shape[0], first.shape[1]
        rdr.close()
        return w, h, nb, fps, "imageio"
    except Exception as e:
        return None, None, None, None, f"실패({e})"


def col_dim(df, col):
    """list 컬럼의 차원 (첫 유효 행 기준)."""
    try:
        v = df[col].dropna().iloc[0]
        return np.asarray(v).shape
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="?", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--base", default="record")
    ap.add_argument("--target-hz", type=float, default=60.0)
    args = ap.parse_args()

    # parquet 경로 결정
    if args.parquet:
        pq = args.parquet
    elif args.task is not None and args.ep is not None:
        chunk = args.ep // 1000
        pq = os.path.join(args.base, args.task, "data", f"chunk-{chunk:03d}",
                          f"episode_{args.ep:06d}.parquet")
    else:
        pq = find_latest_parquet(args.base)
        if pq is None:
            print(f"[오류] {args.base}/ 에서 parquet 을 못 찾음. 녹화를 먼저 하거나 경로를 지정하세요.")
            sys.exit(1)
        print(f"[자동탐색] 최신 parquet: {pq}")

    if not os.path.exists(pq):
        print(f"[오류] 파일 없음: {pq}")
        sys.exit(1)

    df = pd.read_parquet(pq, engine="pyarrow")
    n = len(df)

    # ========================================================================
    hr("1) parquet 구조")
    print(f"  경로: {pq}")
    print(f"  행 수(프레임): {n}")
    print(f"  컬럼 ({len(df.columns)}개): {list(df.columns)}")
    for c in ["observation.state", "action", "observation.sensor"]:
        if c in df.columns:
            print(f"  - {c:22s} dim={col_dim(df, c)}")
        else:
            print(f"  - {c:22s} ❌ 없음")

    # ========================================================================
    hr(f"2) {args.target_hz:.0f}Hz 정렬 정확성 (timestamp 간격)")
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy(dtype=np.float64)
        if n >= 2:
            dt = np.diff(ts)
            dt_ms = dt * 1e3
            eff_hz = 1.0 / np.mean(dt) if np.mean(dt) > 0 else 0.0
            expected_ms = 1000.0 / args.target_hz
            print(f"  timestamp 범위: {ts[0]:.3f} ~ {ts[-1]:.3f} s (길이 {ts[-1]-ts[0]:.2f}s)")
            print(f"  평균 간격: {np.mean(dt_ms):.3f}ms (기대 {expected_ms:.3f}ms @ {args.target_hz:.0f}Hz)")
            print(f"  간격 표준편차: {np.std(dt_ms):.4f}ms (정렬축은 균일해야 하므로 ~0 이어야 정상)")
            print(f"  역산 hz: {eff_hz:.2f}Hz")
            n_expected = (ts[-1] - ts[0]) * args.target_hz + 1
            ok_hz = abs(eff_hz - args.target_hz) < 1.0 and np.std(dt_ms) < 0.5
            print(f"  행 수 {n} vs 기대 ~{n_expected:.0f}  →  {'✓ 정렬 정확' if ok_hz else '⚠️ 확인 필요'}")
        else:
            print("  ⚠️ 행이 2개 미만 — 너무 짧은 에피소드.")
    else:
        print("  ❌ timestamp 컬럼 없음")

    # ========================================================================
    hr("3) raw_ts 메타 — 업샘플 여부 판정")
    print("  raw_ts_* 는 60Hz 축의 각 프레임에서 보간에 쓰인 직전 raw 샘플 ts.")
    print("  60Hz 축에 투영된 값이라, raw 가 60 이상이면 unique 가 축 수에 saturate(~60Hz로 보임),")
    print("  raw 가 60 미만이면 그 값 그대로 보임 → '업샘플 여부' 판정에 사용 (정확한 raw hz 는")
    print("  check_pipeline_live.py 의 SHM 직접측정이 담당: hand=100/robot=300/camera=60 확인됨).")
    raw_cols = [c for c in df.columns if c.startswith("raw_ts_")] + (["axis_ts_ns"] if "axis_ts_ns" in df.columns else [])
    T_ep = float(ts[-1] - ts[0]) if ("timestamp" in df.columns and n >= 2) else 0.0
    if not raw_cols:
        print("  ❌ raw_ts_* / axis_ts_ns 메타 컬럼 없음")
    for c in sorted(raw_cols):
        arr = df[c].to_numpy()
        try:
            arr = arr.astype(np.int64)
        except Exception:
            print(f"  - {c}: (파싱 실패)")
            continue
        uniq = np.unique(arr)
        # unique개수/길이 = 60Hz 축에서 관측된 raw 갱신 빈도. raw≥60 이면 ~60 saturate.
        obs_hz = (uniq.size - 1) / T_ep if T_ep > 0 else 0.0
        if c == "axis_ts_ns":
            print(f"  - {c:24s} unique={uniq.size:5d}  (정렬축 자체, 모든 프레임 고유여야 정상)")
        else:
            # 업샘플 판정: 축(60Hz)에서 본 갱신빈도가 60 근처면 raw≥60(업샘플X), 미만이면 업샘플O.
            tag = "✓ raw≥60 (업샘플 없음)" if obs_hz >= 59.0 else f"⚠️ raw≈{obs_hz:.0f}Hz <60 → 60Hz 축에서 업샘플됨"
            print(f"  - {c:24s} unique={uniq.size:5d}  축관측빈도≈{obs_hz:6.1f}Hz  {tag}")

    # ========================================================================
    hr("4) MP4 view 정합 (해상도/fps/프레임수)")
    video_chunk, ep_mp4 = parquet_to_video_dir(pq)
    if not os.path.isdir(video_chunk):
        print(f"  ❌ videos 폴더 없음: {video_chunk}")
    else:
        views = sorted(os.listdir(video_chunk))
        print(f"  videos chunk: {video_chunk}")
        print(f"  발견된 view: {views}")
        for v in views:
            mp4 = os.path.join(video_chunk, v, ep_mp4)
            if not os.path.exists(mp4):
                print(f"  - {v}: ❌ {ep_mp4} 없음")
                continue
            w, h, nb, fps, src = probe_video(mp4)
            res_ok = (w == 640 and h == 360)
            fps_ok = fps is not None and abs(fps - args.target_hz) < 1.0
            nb_ok = (nb == n) if nb is not None else None
            flags = []
            flags.append("해상도 ✓" if res_ok else f"해상도 ⚠️({w}x{h})")
            flags.append(f"fps ✓({fps:.1f})" if fps_ok else f"fps ⚠️({fps})")
            if nb_ok is True:   flags.append(f"프레임 ✓({nb}=행수)")
            elif nb_ok is False: flags.append(f"프레임 ⚠️({nb}≠{n})")
            else:                flags.append("프레임 ?(메타없음)")
            print(f"  - {v}: {w}x{h} fps={fps} frames={nb} [{src}]  {'  '.join(flags)}")

    # ========================================================================
    hr("5) 값 sanity (NaN/inf, 범위, 보간 특성)")
    for c in ["observation.state", "action"]:
        if c not in df.columns:
            continue
        try:
            mat = np.stack(df[c].to_numpy())  # (N, D)
        except Exception as e:
            print(f"  - {c}: stack 실패 {e}")
            continue
        nan_cnt = int(np.isnan(mat).sum())
        inf_cnt = int(np.isinf(mat).sum())
        # 차원별 변동성 (보간 특성): action 은 ZOH 라 계단형(차분 대부분 0 구간), state 는 linear 라 연속.
        diffs = np.diff(mat, axis=0)
        frac_zero = float(np.mean(np.all(np.isclose(diffs, 0.0), axis=1))) if mat.shape[0] > 1 else 0.0
        print(f"  - {c:18s} shape={mat.shape}  NaN={nan_cnt} inf={inf_cnt}  "
              f"min={mat.min():.3f} max={mat.max():.3f}")
        print(f"      연속프레임 완전동일 비율={frac_zero*100:.1f}% "
              f"({'ZOH=계단형 특성' if c=='action' else 'linear=연속(낮아야 정상)'})")
        if nan_cnt or inf_cnt:
            print(f"      ⚠️ NaN/inf 존재 — 보간/수집 오류 의심")

    hr("요약")
    print("  - 1: state/action 차원이 학습 layout(dex3=33 / waist·head 토글에 따라 가변)과 맞는지 확인.")
    print(f"  - 2: 간격 std≈0 + 역산 {args.target_hz:.0f}Hz → 정렬축 정확.")
    print("  - 3: raw_ts 축관측빈도가 모두 ~60(saturate) → 업샘플 없음. <60 인 모달리티는 업샘플된 것.")
    print("  - 4: 3개 view 모두 640x360 / fps 60 / 프레임수=행수 → 영상-상태 정합.")
    print("  - 5: NaN/inf 0 + action 계단형 / state 연속 → 보간 정상.")
    print("  위 항목에 ⚠️/❌ 가 있으면 알려주세요. 모두 ✓ 면 replay/변환 단계로 진행.")


if __name__ == "__main__":
    main()
