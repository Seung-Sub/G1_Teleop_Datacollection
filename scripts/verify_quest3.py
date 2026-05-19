#!/usr/bin/env python3
"""Quest3 connectivity + IK pipeline verifier.

main.py --no-robot 와 함께 동작. SHM 을 attach 만 하고 (owner X) 주기적으로
주요 SHM 값을 print. Quest3 입력이 SHM 으로 흘러가는지 + worker_g1_ik 의 IK
계산이 정상인지 + 컨트롤러 버튼이 record_mode_shm 으로 매핑되는지 검증.

Usage:
    Terminal 1:  cd /home/user/G1_Teleoperation && conda activate teleop
                 python main.py --no-robot --vr-input controller --camera none \\
                                --hand dex3 --waist fixed --head off

    Terminal 2:  python scripts/verify_quest3.py [--rate 2.0] [--watch]

검증 시나리오:
    1) GUI 띄워지면 'VR' 버튼 클릭 → adb reverse tcp:8012 실행.
    2) 'START' 클릭 (set_start 트리거 → worker_g1_ik 가 RUN 진입).
    3) Quest3 HMD 쓰고 https://<pro4000-ip>:8012 페이지에서 'enter VR' 클릭.
    4) HMD/컨트롤러를 움직이면 verify_quest3.py 의 출력이 변화하는지 확인.
       - HMD trans 가 사용자 머리 움직임에 따라 변화
       - L/R wrist 가 컨트롤러 움직임 반영
       - trig/grip 값이 0~1 사이 입력에 반응
    5) Left grip 누르면 'Action arm' 값이 컨트롤러 변위에 따라 변화 (clutch).
       Left grip 떼면 freeze.
    6) Right-A 누르면 3초 동안 Action arm 이 ready pose 로 부드럽게 수렴.
       이 동안 grip/trigger 입력은 lockout (action 변화 없음).
    7) Left X / Left Y / Right B 누르면 RecMode flag 가 토글 (start/reset 등).
"""
from __future__ import annotations
import argparse
import sys
import time
import multiprocessing as mp

import numpy as np

# 프로젝트 루트를 path 에 추가
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    TELEVISION, QUEST_CONTROLLER, ROBOT_OBS, ROBOT_ACTION,
    RECORD_MODE_LAYOUT, WORKER_FREQ, TELEOP_CONFIG,
    HAND_MAPPING_INV, CAMERA_MAPPING_INV, VR_INPUT_MAPPING_INV,
    WAIST_MAPPING_INV, HEAD_MAPPING_INV,
)


def _fmt_vec(v, decimals=3):
    return np.array2string(np.asarray(v), precision=decimals, suppress_small=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rate',  type=float, default=2.0, help='Print rate Hz (default 2)')
    ap.add_argument('--watch', action='store_true',
                    help='controller 버튼 / record_mode rising-edge 가 발생할 때만 추가 로그')
    ap.add_argument('--full',  action='store_true', help='full 4x4 행렬 출력 (기본은 translation 만)')
    args = ap.parse_args()

    # main.py 가 owner — 여기는 attach only. lock 은 로컬 multiprocessing.Lock 으로
    # 각 SHM 당 하나씩. (실제 SHM 메모리는 OS-wide 공유이므로 다른 프로세스 lock 과 분리.)
    locks = {k: mp.Lock() for k in (
        'television_lock', 'quest_controller_lock', 'robot_obs_lock',
        'robot_action_lock', 'record_lock', 'freq_lock',
    )}

    print("== Quest3 / IK pipeline verifier ==")
    print("main.py --no-robot 이 먼저 실행 중이어야 합니다.")
    print("(아직 안 돌고 있으면 SHM attach 실패로 종료됨)")
    print()

    try:
        shm = {
            'tv':   SharedMemoryManager(TELEVISION,         locks['television_lock'],       'television_shm'),
            'ctrl': SharedMemoryManager(QUEST_CONTROLLER,   locks['quest_controller_lock'], 'quest_controller_shm'),
            'obs':  SharedMemoryManager(ROBOT_OBS,          locks['robot_obs_lock'],        'robot_obs_shm'),
            'act':  SharedMemoryManager(ROBOT_ACTION,       locks['robot_action_lock'],     'robot_action_shm'),
            'mode': SharedMemoryManager(RECORD_MODE_LAYOUT, locks['record_lock'],           'record_mode_shm'),
            'freq': SharedMemoryManager(WORKER_FREQ,        locks['freq_lock'],             'freq_shm'),
            'cfg':  SharedMemoryManager(TELEOP_CONFIG,      locks['record_lock'],           'teleop_config_shm'),
        }
    except FileNotFoundError as e:
        print(f"[FATAL] SHM attach 실패: {e}")
        print("→ 먼저 main.py 를 실행하세요: python main.py --no-robot ...")
        sys.exit(1)

    try:
        cfg = shm['cfg'].read_data()
        print(f"TELEOP_CONFIG: hand={HAND_MAPPING_INV.get(int(cfg['hand_type']), cfg['hand_type'])} "
              f"cam={CAMERA_MAPPING_INV.get(int(cfg['camera_type']), cfg['camera_type'])} "
              f"vr={VR_INPUT_MAPPING_INV.get(int(cfg['vr_input']), cfg['vr_input'])} "
              f"waist={WAIST_MAPPING_INV.get(int(cfg['waist_mode']), cfg['waist_mode'])} "
              f"head={HEAD_MAPPING_INV.get(int(cfg['head_mode']), cfg['head_mode'])}")
    except Exception as e:
        print(f"[WARN] read teleop_config failed: {e}")
    print()

    period = max(1e-3, 1.0 / args.rate)
    prev_btns = {'lx': False, 'ly': False, 'rb': False, 'ra': False}
    prev_mode = None

    try:
        while True:
            t0 = time.perf_counter()
            try:
                tv   = shm['tv'].read_data()
                ctrl = shm['ctrl'].read_data()
                obs  = shm['obs'].read_data()
                act  = shm['act'].read_data()
                mode = shm['mode'].read_data()
                freq = shm['freq'].read_data()
            except Exception as e:
                print(f"[ERR] SHM read: {e}")
                time.sleep(period); continue

            head_t = tv['head_rmat'][:3, 3]
            lw_t   = tv['left_wrist_mat'][:3, 3]
            rw_t   = tv['right_wrist_mat'][:3, 3]

            l_trig, r_trig = float(ctrl['left_trigger']),  float(ctrl['right_trigger'])
            l_grip, r_grip = float(ctrl['left_squeeze']),  float(ctrl['right_squeeze'])
            l_btn = np.asarray(ctrl['left_buttons']).astype(float)
            r_btn = np.asarray(ctrl['right_buttons']).astype(float)
            ctrl_conn = bool(ctrl['connected'])
            ts_tv  = int(tv['television_ts'])
            ts_ct  = int(ctrl['controller_ts'])
            ts_ob  = int(obs['obs_body_ts'])
            ts_act = int(act['action_body_ts'])

            act_waist = act['action_waist']
            act_arm   = act['action_arm']
            act_head  = act['action_head']

            rm = {k: bool(mode[k]) for k in ('start','reset','replay','done','home','deploy')}

            print(f"[{time.strftime('%H:%M:%S')}] ctrl_connected={ctrl_conn}")
            print(f"  HMD head trans   : {_fmt_vec(head_t)}")
            print(f"  L wrist trans    : {_fmt_vec(lw_t)}    R wrist trans : {_fmt_vec(rw_t)}")
            print(f"  L: trig={l_trig:.2f} grip={l_grip:.2f} btn(X,Y,thumb)={_fmt_vec(l_btn, 0)}")
            print(f"  R: trig={r_trig:.2f} grip={r_grip:.2f} btn(A,B,thumb)={_fmt_vec(r_btn, 0)}")
            print(f"  Action waist={_fmt_vec(act_waist)}  head={_fmt_vec(act_head)}")
            print(f"         arm[L]={_fmt_vec(act_arm[:7])}")
            print(f"         arm[R]={_fmt_vec(act_arm[7:])}")
            print(f"  RecMode: {rm}")
            print(f"  Freq   : g1={float(freq['g1_freq']):.1f} hand={float(freq['hand_freq']):.1f} vr={float(freq['vr_freq']):.1f} cam={float(freq['camera_freq']):.1f}")
            print(f"  TS(ns) : tv={ts_tv}  ctrl={ts_ct}  obs={ts_ob}  act={ts_act}")

            # rising-edge 별 추가 로그 (--watch)
            if args.watch:
                cur = {
                    'lx': l_btn[0] > 0.5,
                    'ly': l_btn[1] > 0.5,
                    'ra': r_btn[0] > 0.5,
                    'rb': r_btn[1] > 0.5,
                }
                for k, v in cur.items():
                    if v and not prev_btns[k]:
                        print(f"  >>> RISING-EDGE: {k.upper()}")
                prev_btns = cur

                if prev_mode is not None:
                    for k in rm:
                        if rm[k] != prev_mode[k]:
                            print(f"  >>> RecMode change: {k}={prev_mode[k]} → {rm[k]}")
                prev_mode = rm

            print()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nverifier exit.")


if __name__ == '__main__':
    main()
