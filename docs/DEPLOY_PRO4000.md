# Pro4000 배포 + Quest3 검증 가이드

이 워크스페이스를 pro4000 머신에 옮긴 후 Meta Quest 3 만 유선 연결해서
SHM/IK/컨트롤러 매핑이 정상 동작하는지 검증하는 단계.

> 전제: pro4000 에 conda `teleop` env 가 이미 존재. unitree_sdk2py /
> inspire_sdkpy / pyzed / pyrealsense2 / vuer 등은 이미 설치됨 (이전 작업).
> 본 단계에서 신규로 추가된 dependency 는 *없음* — utils/raw_stream.py,
> utils/align.py, utils/record_collectors.py, utils/camera_discovery.py 는
> numpy / typing 만 의존하고 pyzed/pyrealsense2 는 lazy import.

## 1. 코드베이스 옮기기

### 옵션 A: git pull (권장)
```bash
# 로컬에서
git push origin main

# pro4000 에서 (~/G1_Teleoperation_clean 에서 작업한 가정)
ssh kist@161.122.114.90
cd ~/G1_Teleoperation_clean    # 또는 새 위치
git fetch origin
git checkout main
git pull --ff-only origin main
```

### 옵션 B: rsync (네트워크 issue 또는 unpushed branch)
```bash
# 로컬에서
rsync -av --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'record/' \
    --exclude '.claude/' \
    /home/user/G1_Teleoperation/ \
    kist@161.122.114.90:~/G1_Teleoperation_clean/
```

> `.git` 포함하려면 `--exclude .git` 빼면 됨. 단 pro4000 측 git history 가
> 덮어써질 수 있으니 신중히.

## 2. conda env 확인

```bash
ssh kist@161.122.114.90
conda activate teleop
cd ~/G1_Teleoperation_clean

# Python 3.8.20 인지 확인
python --version

# 신규 모듈 import 확인 (numpy 만 의존)
python -c "from utils.raw_stream import RawStreamBuffer; print('raw_stream OK')"
python -c "from utils.align import interp_to_axis, common_time_axis; print('align OK')"
python -c "from utils.record_collectors import RecordCollectors, align_and_save_episode; print('record_collectors OK')"
python -c "from utils.camera_discovery import discover_realsense, discover_zed, auto_select; print('camera_discovery OK')"

# main.py --help 가 우리 CLI 표시하는지 (vuer argparse hijack 회피 확인)
python main.py --help
```

`--help` 가 표시해야 하는 핵심 flags:
- `--hand {inspire,dex3}`
- `--camera <auto|zed|realsense|none|serial>`
- `--cameras-config <path>`    ← Phase K7 (멀티 카메라 yaml)
- `--vr-input {hand,controller}`
- `--waist {hmd,fixed}`  ← Phase F
- `--head  {dxl,off}`    ← Phase F
- `--tactile {off,on}`   ← Phase K8 (DEX3 press_sensor_state)
- `--lower-body {hoist,loco}`  ← Phase N
- `--gait {off,thumbstick}` / `--gait-stick {split,left,right}`  ← Phase N (loco 보행)
- `--no-robot`            ← Phase F
- `--zed-mode {direct,stream}`
- `--thumb-bend / --thumb-yaw`

## 3. Quest 3 USB 연결

1. Quest 3 를 USB-C 케이블로 pro4000 에 연결
2. Quest 3 헤드셋 안에서 "이 PC 를 신뢰" 다이얼로그 → 허용
3. pro4000 에서 확인:
```bash
adb devices
# List of devices attached
# 1WMHHxxxxxxxxx    device         ← Quest 3 가 보여야 함
```
4. Vuer 포트(8012) 를 Quest 3 로 reverse:
```bash
adb reverse tcp:8012 tcp:8012
```
※ 위 명령은 main.py GUI 의 'VR' 버튼이 자동 호출 (gui/ui_launcher.py:func_vr).
   adb 가 PATH 에 있어야 함.

## 4. Quest3-only 동작 검증 (G1 없이)

### Terminal 1: main.py 시작
```bash
conda activate teleop
cd ~/G1_Teleoperation_clean
python main.py \
    --no-robot \
    --vr-input controller \
    --camera none \
    --hand dex3 \
    --waist fixed \
    --head off
```

옵션 설명:
- `--no-robot`: G1 / hand 워커 spawn 생략. set_g1/set_hand pre-set.
- `--camera none`: 카메라 없이도 가능. 카메라 연결됐으면 `--camera auto` 사용 가능.
- `--hand dex3 --waist fixed --head off`: 가장 안전한 조합 (waist 고정, head DXL skip).

기대 결과:
- GUI 띄워짐
- `[main] hand=dex3 camera=none ... waist=fixed head=off no_robot=True` 로그
- `[VR] start. vr_input=controller` 로그
- worker_g1_ik: `FSM start: WAIT_CONNECT (vr_input=controller)` → set_g1 pre-set 이므로 `WAIT_CONNECT → WAIT_START` 이행 → IK solver 초기화 후 WAIT_START

### Terminal 2: GUI 에서 START 누르기 + Quest3 enter VR

GUI 화면에서:
1. **VR** 버튼 클릭 → adb reverse 자동 실행
2. **START** 버튼 클릭 → `set_start` set → worker_g1_ik FSM RUN 진입

Quest 3 헤드셋:
1. 헤드셋 안 브라우저로 `https://127.0.0.1:8012` 열기 (https 자기 서명 cert → "고급" → "안전하지 않음 진입")
2. "Enter VR" 클릭
3. Vuer 가 CONTROLLER_MOVE 이벤트 publish 시작

### Terminal 3: verify_quest3.py 실행
```bash
conda activate teleop
cd ~/G1_Teleoperation_clean
python scripts/verify_quest3.py --rate 2.0 --watch
```

기대 출력 (예시):
```
[14:23:01] ctrl_connected=True
  HMD head trans   : [0.012 0.451 -0.034]
  L wrist trans    : [-0.123 0.250 -0.180]    R wrist trans : [0.134 0.244 -0.179]
  L: trig=0.00 grip=0.00 btn(X,Y,thumb)=[0. 0. 0.]
  R: trig=0.00 grip=0.00 btn(A,B,thumb)=[0. 0. 0.]
  Action waist=[0. 0. 0.]  head=[0.    -0.175]
         arm[L]=[ 0.    0.349 0.   -0.175 0.    0.    0.   ]
         arm[R]=[ 0.   -0.349 0.   -0.175 0.    0.    0.   ]
  RecMode: {'start': True, 'reset': False, 'replay': False, 'done': False, 'home': False, 'deploy': False}
  Freq   : g1=0.0 hand=0.0 vr=50.0 cam=0.0
  TS(ns) : tv=12345... ctrl=12345... obs=0  act=12345...
```

### 검증 체크리스트

| 체크 | 확인 방법 |
|---|---|
| ✅ Quest3 연결됨 | `ctrl_connected=True` |
| ✅ HMD 움직임 반영 | 머리 움직이면 `HMD head trans` 변화 |
| ✅ 컨트롤러 움직임 반영 | 컨트롤러 움직이면 `L/R wrist trans` 변화 |
| ✅ Trigger | trigger 누르면 0→1 |
| ✅ Grip | grip 누르면 0→1 |
| ✅ Buttons | X/Y/A/B 누르면 해당 btn 비트 1 |
| ✅ Clutch 동작 | Left grip 누르고 컨트롤러 움직이면 `arm[L]` 값 변화. grip 떼면 freeze (값 변화 X) |
| ✅ Recovery | Right-A 누르면 3초 동안 `arm[L/R]` 값이 부드럽게 ready-pose (위 예시) 로 수렴. 이 동안 grip 입력 무시 |
| ✅ Record buttons | Left X → `RecMode.start` 토글. Left Y → `RecMode.reset=True`. Right B → 모든 mode False |
| ✅ TS update | `TS(ns)` 의 tv/ctrl/act 가 매 출력마다 증가 (obs 는 0 유지 — G1 없음) |
| ✅ vr_freq ≈ 50Hz | worker_vr 루프 주파수 |

### 주의

- **waist=fixed** 설정이므로 grip 누른 채 머리 돌려도 waist action 은 0 으로 고정.
  HMD→waist 매핑을 테스트하려면 `--waist hmd` 로 실행.
- **head=off** 설정이므로 Dynamixel 명령 publish 안 함. 그러나 worker_g1_ik 는
  여전히 `action_head = HEAD_READY_RAD = [0, -0.175]` 를 ROBOT_ACTION SHM 에 publish.
- G1 없이 IK solver 가 동작하려면 reduced URDF 가 로딩되어야 함. 만약 실패 로그가
  나오면 `g1_control/assets/g1/g1_body29_inspire_zed.urdf` 가 pro4000 에 있는지 확인.

## 5. 트러블슈팅

### `SHM attach 실패`
verify_quest3.py 가 SHM 을 attach 못 함 → main.py 가 owner 로 생성 중이어야 함.
main.py 가 먼저 실행되었는지 확인. 또는 이전 세션의 SHM 잔존 시:
```bash
ls /dev/shm | grep -E '(camera|television|quest|robot|record|freq|teleop)'
# 필요 시 unlink
python -c "
import multiprocessing.shared_memory as shm
for n in ['camera_shm','television_shm','quest_controller_shm','robot_obs_shm','robot_action_shm','record_mode_shm','freq_shm','teleop_config_shm']:
    try:
        s = shm.SharedMemory(name=n); s.close(); s.unlink(); print(f'unlinked {n}')
    except FileNotFoundError: pass
"
```

### Vuer 페이지가 안 열림
- pro4000 의 8012 포트 방화벽 확인
- adb reverse 가 정상이면 Quest3 안 browser 에서 `https://127.0.0.1:8012`
- 자기 서명 cert 경고 → "고급 → 안전하지 않음 사이트로 이동"
- pro4000 에 `cert.pem` / `key.pem` 이 작업 디렉터리에 있는지 (Vuer 가 https 용으로 사용)

### controller 가 안 잡힘 (hand-tracking 만 잡힘)
- Vuer 가 `CONTROLLER_MOVE` 이벤트를 발생시키려면 사용자가 컨트롤러를 들고 있어야 함
- 헤드셋 내부 옵션: 'Controllers' 활성, 'Hands' 비활성 (또는 controller mode 우선)
- 본 코드는 `--vr-input controller` 시 `MotionControllers` 컴포넌트만 upsert (television.py:236)
- **vuer 0.0.60 client JS 가 hand-tracking 을 hardcode 요청 → Quest 가 controller demote**. 본 워크스페이스의 `scripts/patch_vuer_xr.py disable` 적용 필요 (`status` 로 `PATCHED` 확인). pro4000 측에도 동일 적용

### vr_freq = 0
- worker_vr 가 죽었거나 Vuer 가 event 못 받는 상태
- main.py 로그에 `[VR] start. vr_input=controller` 가 떴는지 확인
- Vuer 페이지를 Quest3 에서 열고 "Enter VR" 까지 했는지

## 6. 다음 단계

이 검증이 통과하면:
1. G1 본체 + Inspire/DEX3 손 연결 후 `--no-robot` 빼고 정식 실행
2. 데이터 수집 시 `python main.py --hand dex3 --camera auto --vr-input controller --waist fixed --head off` 같이 실 환경 옵션
3. 정책 eval 은 별도 `python evaluate.py --mode gr00t_zed --model-path ... --lag-compensate`
