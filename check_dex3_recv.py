#!/usr/bin/env python3
"""
DEX3 hand state DDS 수신 진단.

증상: main 실행 시 [Dex3_Controller] "Waiting for hand state DDS ... 미수신: ['left','right']"
      에서 멈춤 → DEX3 보드가 rt/dex3/{left,right}/state 를 publish 하지 않는 상태.

이 스크립트는 main 과 *독립적으로* 두 state 토픽을 직접 구독해, 실제로 메시지가
도착하는지 / 몇 Hz 인지 / 관절 값이 정상인지 확인한다. 하드웨어/통신 문제인지
소프트웨어 문제인지 구분이 목적.

⚠️ main.py 를 끈 상태에서 단독 실행 (DDS 토픽 충돌 방지).
   DEX3 핸드가 G1 에 연결되고 전원/통신이 켜져 있어야 한다.

사용법:
  cd ~/G1_Teleop_Datacollection
  conda activate teleop
  python check_dex3_recv.py [net_interface]
    net_interface 생략 시 utils/lan_config.yaml 의 network_interface 사용.
"""
import sys
import time
import threading

# ── network interface ──────────────────────────────────────────────────
def get_iface():
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        import yaml, os
        p = os.path.join(os.path.dirname(__file__), 'utils', 'lan_config.yaml')
        with open(p) as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict) and 'network_interface' in cfg:
            return str(cfg['network_interface'])
    except Exception:
        pass
    return 'enp129s0'


iface = get_iface()
print("=" * 70)
print(f"DEX3 state 수신 진단 (net interface = {iface})")
print("=" * 70)

# ── DDS init ────────────────────────────────────────────────────────────
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
except Exception as e:
    print(f"[FAIL] unitree_sdk2py import 실패: {e}")
    sys.exit(1)

try:
    ChannelFactoryInitialize(0, iface)
    print(f"  ChannelFactoryInitialize(0, '{iface}') OK\n")
except Exception as e:
    print(f"[FAIL] DDS 초기화 실패: {e}")
    print(f"       인터페이스를 인자로 지정: python check_dex3_recv.py <iface>")
    sys.exit(1)

kTopicDex3LeftState  = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

# ── 수신 카운터 ──────────────────────────────────────────────────────────
state = {
    'left':  {'count': 0, 'last_msg': None, 'first_t': None, 'last_t': None},
    'right': {'count': 0, 'last_msg': None, 'first_t': None, 'last_t': None},
}
lock = threading.Lock()


def make_cb(side):
    def cb(msg):
        now = time.perf_counter()
        with lock:
            s = state[side]
            s['count'] += 1
            s['last_msg'] = msg
            if s['first_t'] is None:
                s['first_t'] = now
            s['last_t'] = now
    return cb


sub_l = ChannelSubscriber(kTopicDex3LeftState, HandState_)
sub_l.Init(make_cb('left'), 10)
sub_r = ChannelSubscriber(kTopicDex3RightState, HandState_)
sub_r.Init(make_cb('right'), 10)

print("  구독 시작:")
print(f"    {kTopicDex3LeftState}")
print(f"    {kTopicDex3RightState}")
print("\n  5초간 수신 측정 중...\n")

# ── 측정 ──────────────────────────────────────────────────────────────────
DURATION = 5.0
t0 = time.perf_counter()
while time.perf_counter() - t0 < DURATION:
    time.sleep(0.2)
    with lock:
        cl = state['left']['count']
        cr = state['right']['count']
    elapsed = time.perf_counter() - t0
    print(f"    t={elapsed:4.1f}s  left={cl:5d} msgs  right={cr:5d} msgs", end='\r')

print("\n")
print("=" * 70)
print("결과")
print("=" * 70)

for side in ('left', 'right'):
    s = state[side]
    if s['count'] == 0:
        print(f"  [{side:5s}] ❌ 수신 0 — {('rt/dex3/'+side+'/state')} 가 publish 되지 않음.")
        continue
    dur = (s['last_t'] - s['first_t']) if (s['first_t'] and s['last_t']) else 0
    hz = s['count'] / dur if dur > 0 else 0
    print(f"  [{side:5s}] ✓ 수신 {s['count']} msgs, ~{hz:.0f}Hz")
    # 관절 값 정상성 확인
    msg = s['last_msg']
    try:
        qs = [round(float(msg.motor_state[i].q), 3) for i in range(7)]
        modes = [int(msg.motor_state[i].mode) for i in range(7)]
        temps = []
        for i in range(7):
            t = msg.motor_state[i].temperature
            try:
                temps.append(int(t[0]) if hasattr(t, '__len__') else int(t))
            except Exception:
                temps.append('?')
        print(f"           q   = {qs}")
        print(f"           mode= {modes}  (1=정상, 0=보호정지)")
        print(f"           temp= {temps}")
        if all(m == 0 for m in modes):
            print(f"           ⚠️ 전 모터 mode=0 (보호정지). G1/DEX3 재부팅 필요할 수 있음.")
    except Exception as e:
        print(f"           (관절 값 파싱 실패: {e})")

print("\n  해석:")
print("  - 양쪽 다 수신 0 → DEX3 보드가 state 를 안 보냄. 아래 점검:")
print("      1) DEX3 케이블/전원 (USB/통신선 연결)")
print("      2) G1 + DEX3 전원 재부팅 (이전 세션에서 모터 보호정지 상태로 남았을 수 있음)")
print("      3) 같은 net interface(enp129s0) + DDS domain 0 인지")
print("      4) main.py 가 정말 꺼져 있는지 (토픽 점유 충돌)")
print("  - 한쪽만 수신 → 그쪽 케이블/보드 점검.")
print("  - 양쪽 다 정상 수신(~수십~수백Hz) → 하드웨어 정상. 그렇다면 main 의 hand")
print("    초기화 멈춤은 다른 원인 (init 타이밍/네트워크) → 알려주세요.")
