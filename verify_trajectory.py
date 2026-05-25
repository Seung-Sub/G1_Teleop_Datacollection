#!/usr/bin/env python3
"""
에피소드 시계열 궤적 정밀 진단 — verify_episode(통계/정합) 의 보완(궤적/물리적 타당성).

목적:
  - NaN guard 가 튐을 잡았는지 (action 궤적에 비정상 점프 없는지)
  - home 복귀(recovery) 가 부드러운 cosine ease 로 들어갔는지 (수렴 구간 식별)
  - teleop 추종 품질 (obs vs action 오차 분포)
  - 관절각이 물리적 범위(rad) 내인지

출력:
  - 터미널: 차원별 통계 + 점프 탐지 + 추종오차 + 범위 체크 (복사해서 공유 가능)
  - PNG: action/obs 궤적 + 추종오차 플롯 (헤드리스 환경 대비 파일 저장; 업로드해 공유 가능)

사용법:
  cd ~/G1_Teleop_Datacollection
  conda activate teleop
  python verify_trajectory.py                  # 최신 에피소드 자동탐색
  python verify_trajectory.py record/<task>/data/chunk-000/episode_000000.parquet
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # 헤드리스: 창 대신 파일 저장
import matplotlib.pyplot as plt


def find_latest_parquet(base="record"):
    cands = glob.glob(os.path.join(base, "*", "data", "chunk-*", "episode_*.parquet"))
    return max(cands, key=os.path.getmtime) if cands else None


def stack_col(df, col):
    return np.stack(df[col].to_numpy()).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="?", default=None)
    ap.add_argument("--base", default="record")
    ap.add_argument("--target-hz", type=float, default=60.0)
    args = ap.parse_args()

    pq = args.parquet or find_latest_parquet(args.base)
    if pq is None or not os.path.exists(pq):
        print(f"[오류] parquet 없음: {pq}")
        sys.exit(1)
    print(f"[대상] {pq}\n")

    df = pd.read_parquet(pq, engine="pyarrow")
    n = len(df)
    state = stack_col(df, "observation.state")    # (N, D)
    action = stack_col(df, "action")              # (N, D)
    D = state.shape[1]
    dt = 1.0 / args.target_hz

    # ── 1) 차원별 통계 + 범위 ──────────────────────────────────────────
    print("=" * 76)
    print(f"1) 차원별 통계 (D={D}, N={n})  — 관절각(rad) 물리범위 체크")
    print("=" * 76)
    # G1 arm joint 대략적 범위 (rad): 어깨~손목 대부분 ±3.14 이내. 넘으면 의심.
    PHYS_LIMIT = np.pi * 1.05  # 약 ±3.30 rad 초과 시 경고
    print(f"  {'dim':>3} {'state_min':>10}{'state_max':>10}{'act_min':>10}{'act_max':>10}  범위")
    for d in range(D):
        smn, smx = state[:, d].min(), state[:, d].max()
        amn, amx = action[:, d].min(), action[:, d].max()
        over = max(abs(smn), abs(smx), abs(amn), abs(amx)) > PHYS_LIMIT
        flag = "⚠️ 범위초과?" if over else "✓"
        print(f"  {d:>3} {smn:>10.3f}{smx:>10.3f}{amn:>10.3f}{amx:>10.3f}  {flag}")

    # ── 2) 점프(불연속) 탐지 — NaN guard 흔적/이상 ────────────────────
    print("\n" + "=" * 76)
    print("2) action 점프 탐지 (인접 프레임 차이)")
    print("=" * 76)
    da = np.abs(np.diff(action, axis=0))   # (N-1, D)
    # 각 프레임의 최대 차원 변화 (rad)
    max_jump_per_frame = da.max(axis=1)
    # 60Hz 에서 한 프레임(16.7ms) 사이 관절이 급변하면 점프. 0.5rad/frame ≈ 30rad/s = 비정상.
    JUMP_THRESH = 0.5
    jump_idx = np.where(max_jump_per_frame > JUMP_THRESH)[0]
    print(f"  프레임당 최대 관절변화: 평균 {max_jump_per_frame.mean():.4f} rad, "
          f"최대 {max_jump_per_frame.max():.4f} rad")
    print(f"  점프(>{JUMP_THRESH}rad/frame) 발생: {len(jump_idx)}회")
    if len(jump_idx) > 0:
        print(f"    → 발생 프레임(상위 10): {jump_idx[:10].tolist()}")
        print(f"    ⚠️ 점프가 많으면 NaN guard 가 잡은 흔적이거나 teleop 급동작. "
              f"궤적 PNG 로 확인 권장.")
    else:
        print(f"    ✓ 큰 점프 없음 — 궤적 연속적 (NaN guard 정상 작동 또는 NaN 미발생).")

    # ── 3) home 복귀(수렴) 구간 식별 ──────────────────────────────────
    print("\n" + "=" * 76)
    print("3) home/정지 구간 식별 (action 변화가 거의 없는 구간)")
    print("=" * 76)
    # 프레임당 변화가 매우 작은(<0.005rad) 연속 구간 = 정지/수렴(home ready pose 도달 등)
    still = max_jump_per_frame < 0.005
    # 연속 still 구간 길이
    segs = []
    i = 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]:
                j += 1
            if (j - i) >= int(args.target_hz * 0.5):  # 0.5초 이상 정지
                segs.append((i, j))
            i = j
        else:
            i += 1
    if segs:
        print(f"  0.5초 이상 정지/수렴 구간 {len(segs)}개:")
        for s, e in segs[:8]:
            print(f"    프레임 {s}~{e} ({(e-s)*dt:.2f}s)  t={s*dt:.2f}~{e*dt:.2f}s")
        print(f"    → 작업 끝 + home 복귀 후 ready pose 정지 등이 여기 해당될 수 있음.")
    else:
        print(f"  뚜렷한 정지 구간 없음 (계속 움직인 에피소드).")

    # ── 4) teleop 추종 오차 (obs vs action) ───────────────────────────
    print("\n" + "=" * 76)
    print("4) teleop 추종: obs(현재) vs action(목표) 오차")
    print("=" * 76)
    err = np.abs(state - action)   # (N, D)
    print(f"  전체 |obs-action| 평균 {err.mean():.4f} rad, 최대 {err.max():.4f} rad")
    print(f"  (작을수록 로봇이 목표를 잘 추종. arm 은 보통 작고, hand 는 grip 토글로 클 수 있음.)")
    # 차원별 평균 오차 상위
    dim_err = err.mean(axis=0)
    worst = np.argsort(dim_err)[::-1][:5]
    print(f"  추종오차 큰 차원 top5: " + ", ".join(f"dim{d}={dim_err[d]:.3f}" for d in worst))

    # ── 5) PNG 저장 ───────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("5) 궤적 PNG 저장")
    print("=" * 76)
    t = np.arange(n) * dt
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    # (a) action 전체 차원 궤적
    for d in range(D):
        axes[0].plot(t, action[:, d], lw=0.8)
    axes[0].set_title(f"action trajectory (all {D} dims)  — jumps={len(jump_idx)}")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("rad"); axes[0].grid(True, alpha=0.3)
    # (b) obs vs action (대표 차원 — 추종오차 큰 top3)
    for d in worst[:3]:
        line, = axes[1].plot(t, action[:, d], lw=1.0, label=f"action[{d}]")
        axes[1].plot(t, state[:, d], lw=1.0, ls="--", color=line.get_color(), label=f"obs[{d}]")
    axes[1].set_title("obs(--) vs action(—) for top-3 tracking-error dims")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("rad"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    # (c) 프레임당 최대 점프
    axes[2].plot(t[1:], max_jump_per_frame, lw=0.8, color="crimson")
    axes[2].axhline(JUMP_THRESH, color="gray", ls=":", label=f"jump thresh {JUMP_THRESH}")
    axes[2].set_title("max joint change per frame (jump detector)")
    axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("rad/frame"); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.splitext(pq)[0] + "_trajectory.png"
    plt.savefig(out_png, dpi=110)
    print(f"  저장: {out_png}")
    print(f"  (이 PNG 를 업로드하면 궤적을 직접 분석해 드립니다.)")

    print("\n" + "=" * 76)
    print("요약")
    print("=" * 76)
    print(f"  - 점프 {len(jump_idx)}회: 0 또는 소수면 정상. 많으면 NaN guard 흔적/급동작 → PNG 확인.")
    print(f"  - 정지구간 {len(segs)}개: home 복귀 후 ready pose 정지가 잡히면 의도대로 수집된 것.")
    print(f"  - 추종오차 평균 {err.mean():.3f} rad: 작으면 teleop 추종 양호 → replay 적합.")
    print(f"  - 범위초과 ⚠️ 있으면 그 차원 확인 (관절 limit 넘으면 IK/수집 이상).")


if __name__ == "__main__":
    main()
