#!/usr/bin/env python3
"""
DEX3 state DDS 진단 스크립트 (단독 실행).

목적:
  - main.py 를 *끈 상태* 에서, host 가 DEX3 의 state 토픽을 실제로 DDS 로
    받을 수 있는지 직접 확인한다.
  - rt/dex3/{left,right}/state (고주파) 와 rt/lf/dex3/{left,right}/state (저주파)
    네 토픽을 모두 구독해 어느 것이 데이터를 주는지 비교한다.

사용법:
  $ cd ~/G1_Teleop_Datacollection
  $ conda activate teleop
  $ python check_dex3_state.py

해석:
  - rt/dex3/right/state 가 OK  -> 통신 정상. 원인은 main.py 내부(도메인/순서)일 가능성.
  - rt/lf/dex3/... 만 OK        -> 코드의 구독 토픽을 rt/lf/dex3/... 로 바꿔야 함.
  - 전부 NO DATA               -> DEX3 보드가 publish 안 함 (모드/케이블/펌웨어).
"""

import time
import yaml

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

# ---- main.py 와 동일한 방식으로 도메인/인터페이스 초기화 --------------------
cfg = yaml.safe_load(open("utils/lan_config.yaml"))
NET_IFACE = cfg["network_interface"]          # 예: enp129s0
print(f"[init] ChannelFactoryInitialize(0, '{NET_IFACE}')")
ChannelFactoryInitialize(0, NET_IFACE)

TOPICS = [
    "rt/dex3/right/state",
    "rt/lf/dex3/right/state",
    "rt/dex3/left/state",
    "rt/lf/dex3/left/state",
]

WAIT_SEC   = 3.0      # 토픽당 최대 대기 시간
POLL_SLEEP = 0.02     # 20ms 폴링

print(f"[info] 각 토픽을 최대 {WAIT_SEC:.0f}초씩 대기하며 첫 메시지를 기다립니다.\n")

results = {}
for topic in TOPICS:
    sub = ChannelSubscriber(topic, HandState_)
    sub.Init()
    msg = None
    t0 = time.time()
    n_polls = 0
    while time.time() - t0 < WAIT_SEC:
        msg = sub.Read()
        n_polls += 1
        if msg is not None:
            break
        time.sleep(POLL_SLEEP)

    if msg is not None:
        # 7개 모터 q 값 출력
        try:
            qs = [round(float(msg.motor_state[i].q), 3) for i in range(7)]
        except Exception as e:
            qs = f"(motor_state 읽기 실패: {e})"
        results[topic] = ("OK", qs)
        print(f"  [OK]      {topic:28s}  q(7)= {qs}")
    else:
        results[topic] = ("NO DATA", None)
        print(f"  [NO DATA] {topic:28s}  ({n_polls} polls, {WAIT_SEC:.0f}s)")

print("\n================= 요약 =================")
any_ok = any(v[0] == "OK" for v in results.values())
for topic, (status, _) in results.items():
    print(f"  {status:8s}  {topic}")

print("\n================= 해석 =================")
if results["rt/dex3/right/state"][0] == "OK":
    print("  -> rt/dex3/right/state 정상 수신. DDS 통신 OK.")
    print("     원인은 main.py 내부(init 순서/도메인 충돌 등)일 가능성. 결과를 알려주세요.")
elif results["rt/lf/dex3/right/state"][0] == "OK":
    print("  -> 고주파(rt/dex3/...)는 안 오고 저주파(rt/lf/dex3/...)만 옵니다.")
    print("     robot_hand_dex3.py 의 구독 토픽을 rt/lf/dex3/... 로 바꾸면 해결될 수 있습니다.")
elif not any_ok:
    print("  -> 네 토픽 모두 데이터 없음. DEX3 보드가 publish 안 하는 상태입니다.")
    print("     (1) G1 을 control/debug mode 로 바꾼 *후* 다시 이 스크립트를 돌려보세요.")
    print("     (2) DEX3 케이블/전원/펌웨어를 점검하세요.")
else:
    print("  -> 일부 토픽만 수신. 위 요약을 그대로 알려주세요.")
print("========================================")
