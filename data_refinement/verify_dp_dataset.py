#!/usr/bin/env python3
"""
verify_dp_dataset.py — convert_to_dp.py 출력 zarr 가 Diffusion Policy(ReplayBuffer)
형식과 정합하는지 검증.

검증 항목 (사실 기반 — real-stanford/diffusion_policy ReplayBuffer 형식):
  [구조] data/ 그룹: state, action, timestamp, camera_N (+ optional sensor)
         meta/ 그룹: episode_ends (int64, 누적 끝 인덱스)
  [invariant] 모든 data/* 의 dim-0 == episode_ends[-1] == 총 프레임 수
  [episode_ends] 단조 증가, 양수, 마지막 == 총 길이
  [차원] state (N,Ds) action (N,Da) camera (N,H,W,3) uint8
  [NaN] state/action NaN 없음
  [fps] timestamp 간격 == 1/tgt_fps (다운샘플 정합)
  [에피소드 복원] episode_ends 로 자른 각 에피소드 길이 > 0

사용:
  conda activate teleop   # zarr, numcodecs, numpy
  python verify_dp_dataset.py --zarr record/pick_test.zarr --tgt-fps 10
"""
import sys
import argparse
import numpy as np

try:
    import zarr
except ImportError:
    print("ERROR: zarr 미설치. pip install zarr numcodecs", file=sys.stderr)
    raise

PASS, FAIL, WARN = "✅", "❌", "⚠️"
errors, warns = [], []


def check(cond, ok, fail, fatal=True):
    if cond:
        print(f"  {PASS} {ok}")
    else:
        print(f"  {FAIL if fatal else WARN} {fail}")
        (errors if fatal else warns).append(fail)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True, help="DP zarr 경로 (convert_to_dp 출력)")
    ap.add_argument("--tgt-fps", type=float, default=10.0, help="기대 다운샘플 fps")
    args = ap.parse_args()

    print("=" * 76)
    print(f"Diffusion Policy zarr 검증: {args.zarr}")
    print("=" * 76)

    root = zarr.open(args.zarr, mode="r")

    # ---- 1. 그룹 구조 -------------------------------------------------------
    print("\n[1] zarr 그룹 구조")
    top = list(root.keys())
    check("data" in top, "data 그룹 있음", "data 그룹 없음")
    check("meta" in top, "meta 그룹 있음", "meta 그룹 없음")
    if errors:
        print(f"\n{FAIL} 구조 오류. 중단."); sys.exit(1)

    data = root["data"]
    meta = root["meta"]
    data_keys = list(data.keys())
    print(f"     data keys: {data_keys}")
    check("state" in data_keys, "data/state 있음", "data/state 없음")
    check("action" in data_keys, "data/action 있음", "data/action 없음")
    check("episode_ends" in list(meta.keys()), "meta/episode_ends 있음", "meta/episode_ends 없음")

    # ---- 2. episode_ends ----------------------------------------------------
    print("\n[2] episode_ends")
    ends = meta["episode_ends"][:]
    check(ends.dtype == np.int64, f"episode_ends dtype int64", f"dtype={ends.dtype} (int64 기대)", fatal=False)
    check(len(ends) > 0, f"에피소드 {len(ends)}개", "에피소드 0개")
    monotonic = np.all(np.diff(ends) > 0)
    check(monotonic, "episode_ends 단조 증가", "episode_ends 단조 증가 아님 (빈 에피소드?)")
    total = int(ends[-1]) if len(ends) else 0
    print(f"     총 프레임(episode_ends[-1])={total}")

    # ---- 3. data invariant (모든 배열 dim-0 == total) -----------------------
    print("\n[3] data 배열 길이 정합 (invariant)")
    for k in data_keys:
        n = data[k].shape[0]
        check(n == total, f"data/{k} 길이={n} == total({total})",
              f"data/{k} 길이={n} != total({total}) — 정합 깨짐")

    # ---- 4. 차원/dtype ------------------------------------------------------
    print("\n[4] 차원 / dtype")
    s = data["state"]
    a = data["action"]
    print(f"     state shape={s.shape} dtype={s.dtype}")
    print(f"     action shape={a.shape} dtype={a.dtype}")
    check(s.ndim == 2, f"state 2D (N,Ds)", f"state ndim={s.ndim}")
    check(a.ndim == 2, f"action 2D (N,Da)", f"action ndim={a.ndim}")
    check(s.dtype == np.float32, "state float32", f"state dtype={s.dtype}", fatal=False)
    for k in data_keys:
        if k.startswith("camera"):
            c = data[k]
            print(f"     {k} shape={c.shape} dtype={c.dtype}")
            check(c.ndim == 4 and c.shape[-1] == 3, f"{k} (N,H,W,3)", f"{k} shape 이상={c.shape}", fatal=False)
            check(c.dtype == np.uint8, f"{k} uint8", f"{k} dtype={c.dtype}", fatal=False)

    # ---- 5. NaN -------------------------------------------------------------
    print("\n[5] NaN 검사 (state/action)")
    s_arr = s[:]
    a_arr = a[:]
    check(not np.isnan(s_arr).any(), "state NaN 없음", "state 에 NaN 존재")
    check(not np.isnan(a_arr).any(), "action NaN 없음", "action 에 NaN 존재")

    # ---- 6. fps (timestamp) -------------------------------------------------
    print("\n[6] timestamp fps (다운샘플 정합)")
    if "timestamp" in data_keys:
        ts = data["timestamp"][:]
        # 에피소드 경계 제외하고 간격 측정 (경계에서 0으로 리셋되므로)
        starts = np.concatenate([[0], ends[:-1]])
        dts = []
        for st, en in zip(starts, ends):
            seg = ts[st:en]
            if len(seg) > 1:
                dts.extend(np.diff(seg).tolist())
        if dts:
            med = np.median(dts)
            eff = 1.0 / med if med > 0 else 0
            check(abs(eff - args.tgt_fps) < 0.5,
                  f"timestamp fps={eff:.2f} == tgt({args.tgt_fps})",
                  f"timestamp fps={eff:.2f} != tgt({args.tgt_fps}) — 다운샘플 확인 필요", fatal=False)
    else:
        print(f"     {WARN} timestamp 없음")

    # ---- 7. 에피소드 복원 ---------------------------------------------------
    print("\n[7] 에피소드 분할 복원")
    starts = np.concatenate([[0], ends[:-1]])
    lens = ends - starts
    check(np.all(lens > 0), f"모든 에피소드 길이>0 (min={lens.min()}, max={lens.max()}, 평균={lens.mean():.0f})",
          "길이 0인 에피소드 존재")

    # ---- 결과 ---------------------------------------------------------------
    print("\n" + "=" * 76)
    if errors:
        print(f"{FAIL} 검증 실패: {len(errors)}개 치명 오류")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    print(f"{PASS} 모든 필수 검증 통과" + (f" ({len(warns)}개 경고)" if warns else ""))
    for w in warns:
        print(f"   {WARN} {w}")
    print(f"\n   요약: {len(ends)}개 에피소드, {total}프레임, "
          f"state{s.shape[1]}D action{a.shape[1]}D, "
          f"{sum(1 for k in data_keys if k.startswith('camera'))}개 카메라")


if __name__ == "__main__":
    main()
