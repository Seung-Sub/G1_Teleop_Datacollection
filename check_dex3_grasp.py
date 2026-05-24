#!/usr/bin/env python3
"""
DEX3 grasp 거동 모니터 — "물체 쥔 자세까지만 접힘 / 부하 시 죽음" 원인 진단.

main.py 실행 중(Hand connect 후) 동시 실행. grasp/release 를 반복하는 동안
양손 각 모터의 q(위치) / tau_est(추정토크) / temperature / mode 를 추적한다.

확인 포인트:
  - grasp 시 q 가 목표(±1.74)까지 도달하는가, 아니면 중간에 멈추는가
  - 멈춘다면 그 시점 tau_est 가 큰가 (= 위치오차로 토크 포화 → 보호)
  - temperature 가 상승하는가 (= 과열 보호)
  - mode 가 1(control)→0(stop) 으로 바뀌는가 (= 펌웨어가 모터 정지)
  - cmd 발행이 멈추는가 (= hand controller 프로세스 다운)

사용법:
  1. 터미널 A: main.py 실행 (Hand connect 까지)
  2. 터미널 B: conda activate teleop; python check_dex3_grasp.py
  3. 모니터링 중 컨트롤러 trigger 로 grasp↔release 2~3회 (물체 없이),
     그 다음 물체를 쥐어보고, 다시 물체 없이 grasp 해보며 차이 관찰.
"""
import time
import yaml
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, HandCmd_

cfg = yaml.safe_load(open("utils/lan_config.yaml"))
ChannelFactoryInitialize(0, cfg["network_interface"])

latest = {"l_state": None, "r_state": None, "l_cmd": [0, None], "r_cmd": [0, None]}

def _scalar(x):
    """unitree_hg MotorState 필드가 스칼라일 수도, 배열(int8[2] 등)일 수도 있어
    방어적으로 첫 원소를 꺼낸다. (예: temperature 는 int8[2] = [모터온도, 보드온도])"""
    try:
        if isinstance(x, (list, tuple)) or hasattr(x, "__len__"):
            return x[0] if len(x) > 0 else 0
    except Exception:
        pass
    return x

def mk_state_cb(key):
    def cb(msg):
        try:
            latest[key] = {
                "q":    [round(float(_scalar(msg.motor_state[i].q)), 2) for i in range(7)],
                "tau":  [round(float(_scalar(msg.motor_state[i].tau_est)), 2) for i in range(7)],
                "temp": [int(_scalar(msg.motor_state[i].temperature)) for i in range(7)],
                "mode": [int(_scalar(msg.motor_state[i].mode)) for i in range(7)],
            }
        except Exception as e:
            latest[key] = {"err": str(e)}
    return cb

def mk_cmd_cb(key):
    def cb(msg):
        latest[key][0] += 1
        latest[key][1] = [round(float(msg.motor_cmd[i].q), 2) for i in range(7)]
    return cb

ChannelSubscriber("rt/dex3/left/state",  HandState_).Init(mk_state_cb("l_state"), 1)
ChannelSubscriber("rt/dex3/right/state", HandState_).Init(mk_state_cb("r_state"), 1)
ChannelSubscriber("rt/dex3/left/cmd",    HandCmd_).Init(mk_cmd_cb("l_cmd"), 1)
ChannelSubscriber("rt/dex3/right/cmd",   HandCmd_).Init(mk_cmd_cb("r_cmd"), 1)

print(">>> 모니터링 시작 (Ctrl+C 로 종료). trigger 로 grasp/release + 물체 쥐기 테스트.\n")
t0 = time.time()
last = 0.0
prev_cmd = {"l": 0, "r": 0}
# 관측 중 tau 절대값 최대 추적 (안전마진 판단용). 7모터 각각.
max_abs_tau = {"l": [0.0]*7, "r": [0.0]*7}
max_temp    = {"l": [0]*7,   "r": [0]*7}
saw_mode0   = {"l": False, "r": False}

def _update_extrema(side, d):
    if not d or "err" in d:
        return
    for i in range(7):
        max_abs_tau[side][i] = max(max_abs_tau[side][i], abs(d["tau"][i]))
        max_temp[side][i]    = max(max_temp[side][i], d["temp"][i])
        if d["mode"][i] == 0:
            saw_mode0[side] = True

try:
    while True:
        now = time.time()
        # extrema 는 매 루프(50ms) 갱신해 순간 피크도 포착
        _update_extrema("l", latest["l_state"])
        _update_extrema("r", latest["r_state"])
        if now - last >= 1.0:
            last = now
            dl = latest["l_state"]; dr = latest["r_state"]
            lc = latest["l_cmd"][0] - prev_cmd["l"]; rc = latest["r_cmd"][0] - prev_cmd["r"]
            prev_cmd["l"] = latest["l_cmd"][0]; prev_cmd["r"] = latest["r_cmd"][0]
            print(f"--- t={now-t0:5.1f}s | cmd/s L={lc} R={rc} ---")
            if dr and "err" not in dr:
                print(f"  R q   ={dr['q']}")
                print(f"  R tau ={dr['tau']}  temp={dr['temp']}  mode={dr['mode']}")
            elif dr:
                print(f"  R state parse err: {dr['err']}")
            if dl and "err" not in dl:
                print(f"  L q   ={dl['q']}")
                print(f"  L tau ={dl['tau']}  temp={dl['temp']}  mode={dl['mode']}")
            elif dl:
                print(f"  L state parse err: {dl['err']}")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\n\n[Ctrl+C] 측정 종료. 관측 구간 요약:\n")
    print(f"  RIGHT max|tau| (7모터): {[round(x) for x in max_abs_tau['r']]}")
    print(f"  RIGHT max temp        : {max_temp['r']}")
    print(f"  LEFT  max|tau| (7모터): {[round(x) for x in max_abs_tau['l']]}")
    print(f"  LEFT  max temp        : {max_temp['l']}")
    print(f"  mode=0(보호정지) 관측: RIGHT={saw_mode0['r']}  LEFT={saw_mode0['l']}")

print("\n================= 해석 =================")
print("  - cmd/s 가 도중 0 으로 떨어짐  → hand controller 프로세스 다운 (부하로 죽음)")
print("  - mode 가 1→0 으로 바뀜        → 펌웨어가 모터를 stop (보호)")
print("  - grasp 시 |tau| 가 큰 값에서 포화 → 위치오차 토크 포화 (kp 과대 or 물체 저항)")
print("  - temp 가 계속 상승            → 과열 보호 임박")
print("  - q 가 목표(±1.74) 못 미치고 멈춤 → 위 중 하나로 더 못 감")
print("========================================")
