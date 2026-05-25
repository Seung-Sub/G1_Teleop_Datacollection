#!/usr/bin/env python3
"""
전체 데이터 수집 파이프라인 라이브 진단.

main.py 가 *실행 중인 상태에서 함께* 돌려, 모든 SHM 을 직접 읽어 각 장치/스트림의
- 실제 동작 hz (타임스탬프 변화 추적 기반, freq_shm 의 단일 camera_freq 한계 우회)
- 신선도 (마지막 갱신 후 경과 — 멈춘 스트림 탐지)
- 안정성 (hz 표준편차/지터)
를 측정한다. 카메라 3대(ego/wrist_l/wrist_r)를 *개별* 측정하는 게 핵심.

데이터 수집 파이프라인 구축 전 디버깅 단계용 — "모든 모달리티가 안정적으로 들어오는가"
를 한 화면에서 정량 확인.

⚠️ 반드시 main.py 가 *실행 중이고 Teleop 이 RUN 상태* 일 때 실행 (SHM 이 이미 생성+갱신 중).
   main 보다 먼저 켜면 빈 SHM 을 새로 만들어 0 만 읽힌다.

사용법:
  cd ~/G1_Teleop_Datacollection
  conda activate teleop
  python check_pipeline_live.py            # 10초 측정
  python check_pipeline_live.py 20         # 20초 측정
"""
import sys
import time
import numpy as np
from multiprocessing import Lock

sys.path.insert(0, '.')

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    WORKER_FREQ, ROBOT_OBS, ROBOT_ACTION, CAMERA_VIEW,
    TELEVISION, QUEST_CONTROLLER, RECORD_MODE_LAYOUT,
)

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
POLL_HZ = 500.0   # SHM 폴링 주파수 (장치 최대 hz 보다 충분히 높게 — 변화 놓치지 않게)


def attach(schema, name):
    """기존 SHM 에 attach. main 이 이미 생성했어야 함. lock 은 자체 생성(읽기 전용).

    이 프로세스는 SHM 의 owner 가 아니라 *attach(읽기)만* 한다. 그러나 Python
    multiprocessing.shared_memory 는 attach 만 해도 resource_tracker 에 이름을
    등록하고, main(owner)이 unlink 한 뒤 이 프로세스 종료 시 정리하려다
    'leaked / No such file or directory' 경고를 낸다(CPython 알려진 동작 #116849).
    → attach 직후 resource_tracker 에서 unregister 하여 경고를 없앤다. 읽기 전용
       이므로 unregister 해도 안전(우리가 unlink 책임이 없음).
    """
    try:
        mgr = SharedMemoryManager(schema, Lock(), name)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(mgr.shm._name, "shared_memory")
        except Exception:
            pass
        return mgr
    except Exception as e:
        print(f"  [attach 실패] {name}: {e}")
        return None


# ── 측정 대상 정의: (표시이름, shm핸들, ts필드명) ────────────────────────
print("=" * 74)
print("전체 파이프라인 라이브 진단 — SHM attach 중...")
print("=" * 74)

shms = {}
targets = []   # (label, shm, ts_field, expected_hz)

# 카메라 3대 (개별)
for role, shm_name, exp in [('ego', 'rs_ego_shm', 60), ('wrist_l', 'rs_wrist_l_shm', 60), ('wrist_r', 'rs_wrist_r_shm', 60)]:
    s = attach(CAMERA_VIEW, shm_name)
    if s is not None:
        shms[shm_name] = s
        targets.append((f'camera:{role}', s, 'frame_ts', exp))

# robot obs (body + hand 는 같은 ROBOT_OBS SHM 의 다른 ts)
s_obs = attach(ROBOT_OBS, 'robot_obs_shm')
if s_obs is not None:
    shms['robot_obs_shm'] = s_obs
    targets.append(('robot_obs(body)', s_obs, 'obs_body_ts', 300))
    targets.append(('robot_obs(hand)', s_obs, 'obs_hand_ts', 100))

# robot action
s_act = attach(ROBOT_ACTION, 'robot_action_shm')
if s_act is not None:
    shms['robot_action_shm'] = s_act
    targets.append(('action(body)', s_act, 'action_body_ts', 60))
    targets.append(('action(hand)', s_act, 'action_hand_ts', 100))

# VR / television
s_tv = attach(TELEVISION, 'television_shm')
if s_tv is not None:
    shms['television_shm'] = s_tv
    targets.append(('television(VR)', s_tv, 'television_ts', 60))

# controller
s_ctrl = attach(QUEST_CONTROLLER, 'quest_controller_shm')
if s_ctrl is not None:
    shms['quest_controller_shm'] = s_ctrl
    targets.append(('controller', s_ctrl, 'controller_ts', 60))

# freq_shm (워커 자가보고 — 참고용)
s_freq = attach(WORKER_FREQ, 'freq_shm')

if not targets:
    print("\n  측정 대상 SHM 을 하나도 attach 못 했습니다.")
    print("  main.py 가 실행 중인지, 같은 디렉토리에서 실행하는지 확인하세요.")
    sys.exit(1)

print(f"\n  측정 대상 {len(targets)} 개 스트림. {DURATION:.0f}초 측정 시작...\n")

# ── 측정 상태 ──────────────────────────────────────────────────────────
# label → {last_ts, count, intervals(deque), first_seen_t, last_change_t}
import collections
M = {}
for label, shm, field, exp in targets:
    M[label] = {
        'last_ts': None, 'count': 0,
        'intervals_ns': collections.deque(maxlen=2000),
        'last_change_wall': None, 'expected': exp,
        'ts_field': field, 'shm': shm,
    }

# ── 폴링 루프 ──────────────────────────────────────────────────────────
poll_period = 1.0 / POLL_HZ
t_start = time.perf_counter()
t_next = t_start
last_print = t_start

def fmt_ts(d, field):
    v = d.get(field, None)
    if v is None:
        return None
    try:
        return int(np.asarray(v).reshape(-1)[0])
    except Exception:
        return None

while time.perf_counter() - t_start < DURATION:
    nowp = time.perf_counter()
    # 각 SHM 한 번씩 read (같은 SHM 은 1회 read 로 여러 ts 추출)
    cache = {}
    for label, info in M.items():
        shm = info['shm']
        if id(shm) not in cache:
            try:
                cache[id(shm)] = shm.read_data()
            except Exception:
                cache[id(shm)] = None
        d = cache[id(shm)]
        if d is None:
            continue
        ts = fmt_ts(d, info['ts_field'])
        if ts is None or ts <= 0:
            continue
        if info['last_ts'] is None:
            info['last_ts'] = ts
            info['last_change_wall'] = nowp
        elif ts != info['last_ts']:
            dt_ns = ts - info['last_ts']
            if dt_ns > 0:
                info['intervals_ns'].append(dt_ns)
                info['count'] += 1
            info['last_ts'] = ts
            info['last_change_wall'] = nowp

    # 진행 표시 (1초마다)
    if nowp - last_print >= 1.0:
        last_print = nowp
        el = nowp - t_start
        live = sum(1 for i in M.values() if i['count'] > 0)
        print(f"    t={el:4.1f}s  활성 스트림 {live}/{len(M)}", end='\r')

    t_next += poll_period
    sl = t_next - time.perf_counter()
    if sl > 0:
        time.sleep(sl)

# ── 결과 ──────────────────────────────────────────────────────────────
print("\n")
print("=" * 74)
print(f"결과 ({DURATION:.0f}초 측정)")
print("=" * 74)
print(f"  {'스트림':<18}{'실측Hz':>9}{'기대':>6}{'지터ms':>9}{'신선도':>9}   상태")
print(f"  {'-'*18}{'-'*9}{'-'*6}{'-'*9}{'-'*9}   {'-'*12}")

now_end = time.perf_counter()
any_issue = False
for label, info in M.items():
    iv = np.array(info['intervals_ns'], dtype=np.float64)
    exp = info['expected']
    if iv.size == 0:
        print(f"  {label:<18}{'—':>9}{exp:>6}{'—':>9}{'—':>9}   ❌ 갱신 없음(멈춤/미연결)")
        any_issue = True
        continue
    mean_dt_s = np.mean(iv) * 1e-9
    hz = 1.0 / mean_dt_s if mean_dt_s > 0 else 0.0
    jitter_ms = np.std(iv) * 1e-6
    # 신선도 = 마지막 변화 후 경과 (ms)
    stale_ms = (now_end - info['last_change_wall']) * 1e3 if info['last_change_wall'] else 9e9
    # 상태 판정
    ratio = hz / exp if exp > 0 else 1.0
    if stale_ms > 500:
        status = f"⚠️ {stale_ms:.0f}ms 정지"
        any_issue = True
    elif ratio < 0.8:
        status = f"⚠️ 기대 대비 낮음({ratio*100:.0f}%)"
        any_issue = True
    elif jitter_ms > (1000.0/exp) * 1.5 if exp > 0 else False:
        status = "⚠️ 지터 큼"
        any_issue = True
    else:
        status = "✓ 안정"
    print(f"  {label:<18}{hz:>9.1f}{exp:>6}{jitter_ms:>9.2f}{stale_ms:>8.0f}m   {status}")

# freq_shm 자가보고 (참고)
if s_freq is not None:
    try:
        fd = s_freq.read_data()
        def g(k):
            try: return float(np.asarray(fd[k]).reshape(-1)[0])
            except Exception: return 0.0
        print("\n  [참고] freq_shm 워커 자가보고:")
        print(f"    g1={g('g1_freq'):.1f}  hand={g('hand_freq'):.1f}  vr={g('vr_freq'):.1f}  "
              f"camera={g('camera_freq'):.1f}  record={g('record_freq'):.1f}")
        print(f"    ik_solve_ms: avg={g('ik_solve_ms_avg'):.2f} p95={g('ik_solve_ms_p95'):.2f} "
              f"max={g('ik_solve_ms_max'):.2f} (20ms 예산=50Hz)")
    except Exception as e:
        print(f"  freq_shm 읽기 실패: {e}")

print("\n  해석:")
print("  - camera:ego/wrist_l/wrist_r 모두 ~60Hz + 지터 작음 + 신선도 낮음 → 3대 카메라 안정.")
print("  - robot_obs(hand) ~100Hz → hand 100Hz 전환 성공. (50 근처면 Rate 수정 미반영/병목)")
print("  - 특정 스트림 '갱신 없음' → 그 워커 미동작 or RUN 상태 아님 (Teleop START 했는지 확인).")
print("  - action 계열은 teleop 입력 있을 때만 갱신될 수 있음 (가만히 있으면 낮을 수 있음).")
print("  - 신선도(정지 ms)가 크면 그 스트림이 측정 중간에 멈춘 것 → 워커 크래시 의심.")

for s in shms.values():
    try: s.worker_close()
    except Exception: pass
if s_freq is not None:
    try: s_freq.worker_close()
    except Exception: pass
