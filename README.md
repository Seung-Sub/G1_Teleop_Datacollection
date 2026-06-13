# G1 Teleoperation & Data Collection

Unitree **G1** 휴머노이드 (양손 **Inspire** 또는 **DEX3**) 를 **Meta Quest 3** 컨트롤러
로 원격 조작하고, **다중 modality 시간 정렬된 에피소드** 를 수집하며, 학습된 정책을
**inference-lag 보상** 과 함께 평가하는 워크스페이스.

> 본 README 는 2026-05 cleanup 후 코드베이스 기준입니다. 이전 버전과 달리 KISTAR
> hand / AMO whole-body 코드가 제거되었고, controller-mode + multi-modal alignment
> pipeline 이 추가되었습니다.

---

## 목차

1. [핵심 기능](#1-핵심-기능)
2. [시스템 개요](#2-시스템-개요)
3. [요구사항](#3-요구사항)
4. [설치](#4-설치)
5. [실행 모드 + CLI](#5-실행-모드--cli)
6. [실 운용 시나리오](#6-실-운용-시나리오)
7. [컨트롤러 매핑](#7-컨트롤러-매핑)
8. [데이터 수집 파이프라인](#8-데이터-수집-파이프라인)
9. [정책 평가 (eval)](#9-정책-평가-eval)
10. [검증 절차](#10-검증-절차)
11. [트러블슈팅](#11-트러블슈팅)
12. [관련 문서 / 파일 인덱스](#12-관련-문서--파일-인덱스)

---

## 1. 핵심 기능

- **Controller-mode SE(3) clutch teleoperation**
  - Grip 누른 동안만 컨트롤러 변위가 G1 양손 EE 에 반영, 떼면 freeze
  - 다시 잡으면 그 자리에서 anchor 재캡처 → 사용자 자세 무관, 부드럽게 이어 조작
- **HMD-on-neck 운용 (절대좌표 미사용)**
  - HMD 회전 변위만 G1 waist 에 매핑 (`--waist hmd`) — grip 누른 동안만 추종
  - `--waist fixed` 옵션으로 waist 잠금 (HMD 가 흔들려도 영향 없음)
- **Trigger toggle 양손 grasp**
  - Inspire 6 DOF / DEX3 7 DOF 어느 쪽이든 trigger rising-edge 한 번에 5 손가락 close↔open
  - 엄지 자세는 `--thumb-bend / --thumb-yaw` 사전 세팅 (CLI)
- **Right-A Smooth Recovery (3 초 cosine ease) + 입력 lockout**
  - Ready pose 로 부드럽게 복귀, 복귀 도중 grip/trigger/모든 버튼 무시 → 안전 보장
- **Controller record buttons**
  - 좌 X = Start toggle (Record 시작 / early-stop+save)
  - 좌 Y = Drop (현 에피소드 폐기)
  - 우 B = SET (task_name 은 GUI 입력)
- **Multi-modal time alignment**
  - 8 개 streaming SHM 마다 `*_ts` (monotonic ns) 필드
  - 모든 stream 의 raw timestamp + 데이터를 episode 동안 background poller 가 collect
  - 에피소드 close 시 60 Hz 공통 시간축 → linear (continuous) / ZOH (image, discrete) 보간
- **External-policy 평가 + inference-lag 보상**
  - `evaluate.py` 가 별도 conda env (예: gr00t) 에서 ROBOT_OBS 읽어 policy.get_action
  - chunk 앞부분을 `t_publish - t_obs` 만큼 trim → robot 도달 시각이 정렬됨
- **`--no-robot` 검증 모드**
  - G1 / hand 하드웨어 없이 Quest3 + IK 만으로 파이프라인 동작 확인 가능
- **카메라 자동감지**
  - RealSense (D435i → D455 → D405) 우선 → ZED (2i → Mini) fallback
  - serial 직접 지정 (`--camera <serial>`) + 역할 명명 stub (`utils/cameras.yaml`)
- **하드웨어 토글**
  - `--head off` Dynamixel 미사용
  - `--waist fixed` waist 매핑 비활성
  - `--camera none` 카메라 없이 운용

## 2. 시스템 개요

### 2-1. 데이터 흐름

```
┌──────────────────────┐
│  Quest 3 (USB wired) │
│  - HMD pose          │
│  - L/R controller    │  ──► (Vuer HTTPS:8012, WebXR)
│    pose + trigger    │      │
│    + grip + buttons  │      ▼
└──────────────────────┘   worker_vr  ──►  TELEVISION SHM
                            (60Hz)         QUEST_CONTROLLER SHM
                                              │
                                              ▼
                                  worker_g1_ik (60Hz)
                                  - SE(3) clutch on grip-hold
                                  - HMD R_delta → waist (옵션)
                                  - Right-A 3s cosine recovery
                                  - DEX3/Inspire common interface
                                              │
                          ROBOT_ACTION SHM ◄──┘
                                  │
                                  ▼
                          worker_g1_ctrl (60Hz cmd / 300Hz obs)
                          - LowCmd_ / LowState_ DDS
                          - Dynamixel head (--head dxl)
                          - ROBOT_OBS SHM (300Hz)

                  worker_hand_ctrl (100Hz)
                  ├─ Inspire DDS rt/inspire_hand/ctrl/{l,r}
                  └─ DEX3 DDS    rt/dex3/{left,right}/{cmd,state}

                  worker_zed (~30Hz)  or  worker_camera (RealSense 60fps 640x360)
                          │
                          ▼
                      CAMERA_VIEW SHM ─► ego / wrist_l / wrist_r (각 640x360, 60fps) + ts

                  worker_record (FSM 외부 20Hz)
                  ├─ start: RecordCollectors 시작 (3 poller thread)
                  │   - robot poller 1kHz : obs_body / obs_hand / action_body/hand
                  │   - tv poller 200Hz   : television + controller
                  │   - camera poller 100Hz : zed left/right or 3× realsense view
                  ├─ stop: stop_and_dump → align_and_save_episode
                  │   - common_time_axis 60Hz (intersection)
                  │   - linear-interp continuous, ZOH images
                  │   - ParquetSink + extra_columns(raw_ts_*) + VideoSink (각 view mp4)
                  └─ LeRobot v2.1 dataset layout — modality.json 자동 생성 (utils/modality_layout)
```

### 2-2. 모듈 책임 한 줄 요약

| 영역 | 모듈 | 역할 |
|---|---|---|
| Entry | `main.py` | CLI parse → SHM owner-create → worker spawn |
| Eval entry (GR00T) | `evaluate.py` | 별도 conda env 에서 GR00T N1.7 Gr00tPolicy 평가 (lag-compensate) |
| Eval entry (DP) | `evaluate_dp.py` | 별도 conda env 에서 Diffusion Policy .ckpt 평가 |
| VR 입력 | `open_television/television.py`, `tv_wrapper.py`, `workers/worker_vr.py` | Vuer 이벤트 → SHM, OpenXR→Robot 기저 변환 |
| IK | `g1_control/g1_ik.py`, `workers/worker_g1_ik.py` | Pinocchio+CasADi IPOPT, clutch + recovery + waist clutch |
| G1 본체 | `g1_control/g1_whole_control.py`, `workers/worker_g1_ctrl.py` | LowCmd_/LowState_ DualRate 50/300Hz, `damp_to_release` 안전 종료 |
| 하반신 (loco) | `workers/worker_loco.py` | `--lower-body loco` 시 LocoClient (rt/arm_sdk + thumbstick→Move) |
| 머리 | `g1_control/g1_head_dynamixel.py` | XL 시리즈 2모터 syncwrite (`--head dxl`) |
| 손 | `hand_control/robot_hand_inspire.py`, `robot_hand_dex3.py`, `workers/worker_hand_ctrl.py` | DDS, controller-mode trigger toggle, hand-mode retargeting, DEX3 rate-limit + 좌우 거울대칭 |
| 손 터치 | `workers/worker_hand_dds.py` | Inspire Modbus (DEX3 미해당) |
| 카메라 | `workers/worker_zed.py`, `worker_camera.py`, `utils/camera_discovery.py` | direct/stream, RealSense first auto-detect |
| 정렬 | `utils/raw_stream.py`, `utils/align.py`, `utils/record_collectors.py` | RawStreamBuffer dedup + interp_to_axis + common_time_axis |
| 저장 | `utils/parquet_sink.py`, `utils/video_sink.py` | LeRobot v2.1 + raw_ts_* extra |
| 정책 평가 (GR00T) | `workers/worker_deploy_policy.py` | N1.7 Gr00tPolicy slow(20Hz)/fast(60Hz) + cross-fade + lag trim |
| 정책 평가 (DP) | `workers/worker_deploy_dp.py` | DP slow(10Hz)/fast(60Hz), n_obs_steps deque, 평탄 obs dict |
| UI | `gui/ui_launcher.py` | PyQt5 — 카메라뷰, 진행률, 터치맵, 모드 토글 |
| 키보드 | `workers/keyboard_listener.py` | `q`=shutdown, `h`=go_home |
| 보조 | `utils/mat_tool.py` | cosine_ease, se3_interp (quat slerp), fast_mat_inv |

## 3. 요구사항

### 3-1. OS / Python
- Ubuntu 22.04 (테스트), 다른 Linux 도 가능
- Python 3.8 (conda env)

### 3-2. 하드웨어 (선택적, --no-robot 으로 부분 검증 가능)
- **Unitree G1** 본체 + Ethernet (DDS)
- **양손 hand**: Inspire RH56 또는 Unitree DEX3-1
- **머리** (optional): Dynamixel XL 2모터 + USB serial
- **카메라** (optional): RealSense D435i/455/405 또는 ZED 2i/Mini
- **Meta Quest 3** + USB-C 데이터 케이블
- **Meta Horizon Link 데스크톱 앱은 Linux 에 필요 없음** (Quest 헤드셋 안 브라우저
  + Vuer 서버만 사용)

### 3-3. 외부 SDK
- ZED SDK + `pyzed` (ZED 사용 시)
- `pyrealsense2` (RealSense 사용 시)
- `unitree_sdk2_python` (G1 + DEX3)
- `inspire_sdkpy` (Inspire DDS + Modbus touch)
- `dynamixel_sdk` (`--head dxl` 사용 시)
- `vuer` (Quest3 WebXR)
- `pinocchio`, `casadi`, `cyclonedds`

## 4. 설치

**전체 설치 절차는 [`docs/INSTALL.md`](docs/INSTALL.md) 참고.** 2026-05-20 에 검증된
정확한 명령 시퀀스 + 버전 + 자주 만나는 오류 대처법까지 포함되어 있습니다.

요약:
```bash
# 1) conda env
conda create -n teleop python=3.8 -c conda-forge -y && conda activate teleop

# 2) core (pinocchio + casadi 는 conda-forge, 나머지는 pip)
conda install -c conda-forge -y pinocchio casadi
pip install 'numpy<2' scipy 'opencv-contrib-python-headless<4.11' pyarrow pandas pyyaml 'imageio[ffmpeg]' pyqt5
# opencv-contrib-python-headless: cv2.aruco 포함 + Qt 번들 없음 (PyQt5 와의 충돌 회피)

# 3) Vuer (params_proto 는 2.x 라인으로 pin — 3.x 에서 vuer 0.0.60 이 필요로 하는 `Flag` symbol 제거됨)
pip install --no-deps 'vuer==0.0.60' 'params_proto>=2.12,<3.0'
pip install aiohttp aiohttp-cors websockets msgpack dotvar pillow

# 4) Unitree SDK + Dynamixel + logging (cyclonedds 는 unitree_sdk2py 가 dep 으로 자동 설치)
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git ~/unitree_sdk2_python
pip install -e ~/unitree_sdk2_python
pip install dynamixel_sdk logging_mp
# logging_mp 에 get_logger/basic_config alias 추가 — INSTALL.md §4-5

# 5) 카메라 SDK (옵션)
pip install pyrealsense2                       # RealSense
# ZED: ZED SDK 설치 후 python /usr/local/zed/get_python_api.py

# 6) 워크스페이스 (setup.py 의 torch / scikit-learn / nlopt / 등 함께 설치 — 수십 분)
pip install -e .

# 7) Vuer WebXR hand-tracking 기본값 OFF 패치 (controller 모드 필수)
#    vuer 0.0.60 client JS 가 hand-tracking 을 hardcode 로 요청 → Quest 가 hand 우선 모드로
#    전환하면서 controller 입력이 demote 됨. 또 WebSocket URL 의 port 가 누락된 bug 도 함께 패치.
python scripts/patch_vuer_xr.py disable
python scripts/patch_vuer_xr.py status   # PATCHED 표시 확인

# 8) 검증
QT_QPA_PLATFORM=offscreen python scripts/verify_offline.py
# → SUMMARY  PASS=9  FAIL=0
```

> 평가 (`evaluate.py`) 는 별도 conda env (예: `gr00t`) 에서 실행 권장. main.py 의
> teleop env 와 충돌 회피.

Quest 3 USB 연결은 별도 단계 — [`docs/QUEST3_SETUP.md`](docs/QUEST3_SETUP.md) 참고.

## 5. 실행 모드 + CLI

### 5-1. `main.py` 핵심 옵션

```text
--hand        {inspire, dex3}            손 하드웨어 (default: inspire)
--camera      {auto, zed, realsense,     'auto' 면 RealSense 우선 자동감지 → ZED fallback.
               none, <serial>}            cameras.yaml 비어있을 때만 사용 (단일 카메라).
--camera-role ego (default)              cameras.yaml 없이 운용 시 role 라벨
--zed-mode    {direct, stream}           direct=USB, stream=set_from_stream(legacy)
--cameras-config <path>                  multi-camera 매핑 yaml (default: utils/cameras.yaml)
--vr-input    {hand, controller}         (default: hand)
--waist       {hmd, fixed}               HMD→waist 매핑 활성 / 고정
--head        {dxl, off}                 Dynamixel head 사용 / 미사용
--tactile     {off, on}                  손 촉각 로깅. on=DEX3 press_sensor_state / Inspire 17점 촉각.
                                         (Inspire 는 off 일 때 손 상태 Hz↑ — 브리지가 촉각 미read)
--lower-body  {hoist, loco}              hoist=rt/lowcmd (호이스트 현수 전제),
                                         loco=rt/arm_sdk (motion mode, 내장 LocoClient 가 leg/waist 제어)
--gait        {off, thumbstick}          (loco 전용) thumbstick→LocoClient.Move 보행
--gait-stick  {split, left, right}       (gait=thumbstick) stick 매핑 방식
--thumb-bend, --thumb-yaw  (0.0~1.0)     (Inspire) controller-mode 엄지 사전 자세 override
--grip-profile  <name>                   (Inspire) 그립 프로파일 {full_oppose,tripod,pinch,lateral,hook}
                                         (hand_control/inspire_grip_profiles.yaml). 상세는 USAGE.md
--grasp-fingers, --close-depth           (Inspire) 파지 손가락 subset / 깊이 override
--grip-force, --grip-speed  (0~1000)     (Inspire) force_set(g)/speed_set 상한 — 과부하 차단·파지속도 (deploy 에도 적용)
--no-robot                               G1/hand worker 생략 (set_g1/set_hand 자동 set)
```

### 5-2. 자주 쓰는 조합

```bash
# 일반 운용 (DEX3 + RealSense 멀티 + controller-mode, HMD 목에 걸고, hoist 현수)
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist

# Inspire + ZED stereo + waist 고정 (조각 작업 등 허리 안정 필요)
python main.py --hand inspire --camera zed --vr-input controller \
               --waist fixed --head dxl --lower-body hoist

# loco 모드 (내장 LocoClient + thumbstick 보행) — Debug Mode 아닌 motion control 진입 필요
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body loco --gait thumbstick

# Quest3 검증 only (G1/hand 없이)
python main.py --no-robot --vr-input controller --camera none \
               --hand dex3 --waist fixed --head off
```

## 6. 실 운용 시나리오

### 6-1. 운용 흐름

1. 로봇 ON, G1 / hand / DXL Ethernet/USB 연결. **hoist 운용 시 호이스트 현수 + 발 가벼운 접지**.
2. 카메라 USB 연결 (autodetect — `utils/cameras.yaml` 에 serial 매핑).
3. Quest 3 PC 에 USB 유선 연결 — `adb devices` 가 `device` 상태인지 확인
   ([docs/QUEST3_SETUP.md](docs/QUEST3_SETUP.md))
4. G1 리모컨 **L2+R2 → L2+A** 로 **Debug Mode** 진입 (hoist 모드 필수 — 내장 컨트롤러 OFF).
5. `python main.py --hand dex3 --camera realsense --vr-input controller --waist fixed --head off --lower-body hoist`
6. GUI 띄워지면 **G1 → Hand → VR** 순서로 connector 버튼 클릭:
   - G1 클릭 후 리모컨 **start → A** (zero_torque → default_pos 통과)
   - Hand 는 DEX3 양손 init (펴진 자세)
   - VR 클릭 시 `adb reverse tcp:8012` 자동 실행
7. Quest 3 헤드셋 내 브라우저에서 `https://127.0.0.1:8012` 접속 (자기서명 cert 우회) → "Enter VR".
   - patch_vuer_xr.py 적용된 상태면 controller 모드로 자동 진입 (hand-tracking 차단됨)
   - HMD 안에 ego 카메라 영상 + 컨트롤러 가상 모델 보이면 정상
8. **HMD 를 목에 걸기** (Quest 의 proximity sensor 자동 sleep 옵션은 꺼두기).
9. GUI **START** 버튼 → worker FSM RUN 진입.
10. 컨트롤러 **Grip 누른 채** 자연스러운 자세 잡고 조작 시작 — head-yaw 정렬 + delta clutch 적용.

### 6-2. 데이터 수집

1. GUI 에 task_name + num_episodes + episode_len 입력 → SET 버튼
2. 컨트롤러로 운용 시작
3. **Left X** rising-edge → record 시작 / 다시 누르면 early-stop+save
4. **Left Y** → 현 에피소드 폐기 (reset)
5. **Right B** → SET (task_name 은 GUI 에 이미 들어가 있는 값 사용)
6. 저장 위치: `record/<task_name>/data/chunk-XXX/episode_XXXXXX.parquet` + `videos/`

자세한 align 절차는 §8.

### 6-3. Ready Pose 복귀

운용 중 자세가 꼬였거나 안전 위치로 가고 싶을 때:
- **Right A** 1회 누름 → 3초 cosine ease 로 양손 EE + waist 가 ready pose 로 부드럽게 수렴
- 복귀 동안 모든 컨트롤러 입력 무시 (grip/trigger/다른 버튼)
- 복귀 끝나면 anchor 모두 리셋 → grip 다시 잡으면 새로 anchor 캡처되어 자연스럽게 이어 조작

## 7. 컨트롤러 매핑

| 버튼 / 입력 | 동작 |
|---|---|
| **좌/우 Squeeze (grip)** | 누른 동안만 EE clutch 추종 (떼면 freeze). 다시 잡으면 새 anchor 캡처 |
| **좌/우 Trigger** | rising-edge 마다 해당 손 grasp toggle (close↔open) |
| **HMD orientation** | grip 누른 동안 (R_delta) waist 매핑 (`--waist hmd`) |
| **Right A** | Ready pose 복귀 (3초 ease + 입력 lockout) |
| **Right B** | SET (task_name 은 GUI 에서 미리 입력) |
| **Left X** | Record start toggle (RECORDING 중이면 early-stop+save) |
| **Left Y** | Drop (현 에피소드 폐기) |
| **좌/우 Thumbstick** | (`--gait thumbstick`) loco 모드 보행 (vx/vy/wz → LocoClient.Move) |

### 7-1. SE(3) clutch 알고리즘 (controller-mode)

PART6 에 따라 grip rising-edge 에서 **HMD head yaw** 를 한 번 캡처해
**`R_yaw_align`** 으로 고정한 뒤, 컨트롤러 변위를 회전/병진 따로 분리해서 EE 에 적용
(`workers/worker_g1_ik.py` § `_yaw_align_from_head` + grip-engage 블록):

```
R_yaw_align  = Rz(head_yaw_at_grip_engage)
R_rel        = R_yaw_align · (R_ctrl_now · R_ctrl_anchorᵀ) · R_yaw_alignᵀ
Δp           = R_yaw_align · (p_ctrl_now − p_ctrl_anchor)
R_ee_target  = R_rel · R_ee_anchor
p_ee_target  = p_ee_anchor + Δp
```

효과: HMD 를 목에 걸어 약간 기울어진 상태에서 컨트롤러를 들어도, 사용자의 “앞/옆”
감각이 G1 base frame 의 앞/옆과 정합. recovery (Right A) 후엔 `R_yaw_align` 이
None 으로 리셋되어 다음 grip 시 재캡처.

### 7-2. DEX3 안전성

`hand_control/robot_hand_dex3.py` 가 명령 publish 단계에서:

- **rate-limit** `_STEP_MAX = 0.18 rad/cycle` — trigger toggle 시 양손 갑작스러운 close 충격 방지
- **position error cap** `_MAX_POS_ERR = 0.30 rad` — 상태값 정체 / 비정상 점프 시 폭주 차단
- **좌우 거울대칭 정정** — 동일 grasp 의도가 좌/우 손에서 실제로 같은 모양이 되도록 Thumb2/Index/Middle 의 close 방향 부호 반전 (DEX3-1 공식 spec)
- **가동범위**: Index0/Mid0 = ±90° (1.571 rad), `_LIMIT_MARGIN = 0.97` 로 hard limit 직전 clamp

### 7-3. 안전 종료 (damp_to_release)

`g1_control/g1_whole_control.py.damp_to_release(ramp_sec=2.5, kd_hold=5.0)` 가
`q` 종료 시 / SIGTERM 시 kp 를 1→0 으로 cosine ramp 하면서 kd 만 유지 →
사용자가 hoist 풀기 전까지 G1 이 갑자기 떨어지지 않음.

> `--vr-input hand` (hand-tracking) 모드는 5 finger tip 만 SHM 으로 노출하는 현재
> 구조상 Inspire 만 정상 retargeting. DEX3 hand-tracking 은 25 landmark 필요해 
> 안전 release 자세만 publish 됨 → DEX3 + hand-tracking 시 main.py 에 경고.

## 8. 데이터 수집 파이프라인

### 8-1. SHM 시각 동기화

8 개 streaming SHM 마다 `*_ts` 필드 (np.int64, `time.perf_counter_ns()`):

| SHM | TS field(s) | writer | 주기 |
|---|---|---|---|
| ROBOT_OBS | `obs_body_ts`, `obs_hand_ts` | worker_g1_ctrl, worker_hand_ctrl | 300Hz / 100Hz |
| ROBOT_ACTION | `action_body_ts`, `action_hand_ts` | worker_g1_ik, worker_hand_ctrl, worker_deploy_policy/dp | 60Hz / 100Hz |
| CAMERA_VIEW (ego/wrist_l/wrist_r) | `frame_ts` | worker_zed or worker_camera | 60fps (RealSense) / ~30Hz (ZED) |
| TELEVISION | `television_ts` | worker_vr | 60Hz |
| QUEST_CONTROLLER | `controller_ts` | worker_vr | 60Hz |
| LEFT/RIGHT_TOUCH | `l_touch_ts`, `r_touch_ts` | worker_hand_l/r_dds | ~100Hz |
| DEPTH_MAP | `depth_map_ts` | worker_zed | ~30Hz |

`SharedMemoryManager` 의 partial-write 덕에 같은 SHM 안에서도 body / hand writer 가
독립적으로 자기 ts 만 갱신.

### 8-2. RecordCollectors (백그라운드 polling)

`worker_record.RECORDING` 진입 시 `utils/record_collectors.RecordCollectors`
인스턴스가 3 개 thread 시작:

- `_poll_robot` (1 kHz): obs_body, obs_hand, action_body, action_hand
- `_poll_television` (200 Hz): television, controller
- `_poll_camera` (100 Hz): camera_zed left+right, camera_realsense

각 poll 에서 SHM 의 `*_ts` 확인 → 이전 ts 와 같으면 skip (dedup),
새 ts 면 `(ts, payload_copy)` 를 `RawStreamBuffer` 에 push.

### 8-3. 에피소드 close → align_and_save_episode

1. 모든 collector thread stop → `dump()` 로 raw (ts, payload) 시퀀스 획득
2. `utils/align.common_time_axis` → 채워진 모든 stream 의 intersection 에 60 Hz uniform 축
3. 각 stream 을 `interp_to_axis` 로 정렬:
   - obs/action (continuous): `linear` (numpy.interp per-column)
   - camera frames: `zoh` (직전 sample frame 그대로 — 이미지 보간 X)
4. `ParquetSink.append` loop → state/action/timestamp 컬럼
5. `ParquetSink.add_extra_column` 으로 `axis_ts_ns` + 각 stream 의 `raw_ts_<stream>`
   추가 → 후처리/디버그 시 phase offset 분석 가능
6. `VideoSink` 가 ZOH-pick 된 frame 리스트로 mp4 저장 (양안 stereo 또는 RealSense single)

LeRobot v2.1 호환 파일 구조 (parquet + mp4) 유지.

### 8-4. 후처리 (선택)

`data_refinement/` 에 변환기:
- `convert_to_dp.py` — LeRobot → Diffusion Policy zarr replay buffer
- `convert_to_act.py` — LeRobot → ACT per-episode HDF5
- `merge_parquet_data.py`, `sequential_merge.py` — 에피소드 합치기
- `inspect_parquet.py`, `plot_parquet.py` — 분석/시각화
- `apply_mask_to_videos.py` — ZED workspace mask post-apply

## 9. 정책 평가 (eval)

두 가지 정책 경로 — **GR00T N1.7** 또는 **Diffusion Policy** — 별도 conda env 에서.

### 9-1. GR00T N1.7 (`evaluate.py`)

전체 파이프라인 (수집 60Hz → GR00T 변환 20fps → 학습 → 추론) 은
[`GR00T_PIPELINE_GUIDE.md`](GR00T_PIPELINE_GUIDE.md) 참고. 배포 측 N1.7 정합 분석은
[`GR00T_N17_deploy_analysis.md`](GR00T_N17_deploy_analysis.md).

```bash
# Terminal 1 (teleop env)
conda activate teleop
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist

# Terminal 2 (gr00t env)
conda activate groot
python evaluate.py --mode gr00t_rs_multi --model-path /path/to/checkpoint-XXXXX \
    --embodiment-tag new_embodiment --device cuda \
    --action-method tem --slow-hz 20 --fast-hz 60 \
    --lag-compensate --lag-log-every 50
```

N1.7 은 `Gr00tPolicy(embodiment_tag, model_path, *, device)` 시그니처. 학습 시
지정한 `modality_config` 가 체크포인트(processor)에 저장되어 inference 시 자동
로딩되므로 deploy 측에 별도 `data_config_key` 불필요. `--data-config-key` /
`--denoising-steps` 는 하위호환 stub.

**Video temporal history (Phase B, 학습 정합)**: `g1_dex3_config.video.delta_indices=[-20, 0]`
(20fps 다운샘플 후 -20 = 정확히 1초 전 frame) 이므로, deploy 는 카메라 60fps SHM 을
별도 polling thread 가 ring buffer (per-camera 120 슬롯) 에 모아두고, 추론 시점마다
`target_past_ts = now - 1.0s` 로 ts 기반 pick → `frames[role] = [past, current]` 두
frame 을 학습과 동일 순서로 stack. warmup(<1초) 시 현재 frame 복제로 `allow_padding`
clamp 와 일치하는 거동. (`workers/worker_deploy_policy.py:_CameraFrameRing`).

### 9-2. Diffusion Policy (`evaluate_dp.py`)

```bash
# Terminal 1 (동일)
# Terminal 2 (DP 학습 환경, 예: umi)
conda activate umi
python evaluate_dp.py --mode gr00t_rs_multi \
    --model-path /path/to/checkpoints/latest.ckpt \
    --slow-hz 10 --fast-hz 60
```

DP 는 단일 task / 평탄 obs dict / `n_obs_steps=2` 누적 / 60→10fps 학습이라
slow-hz=10 기본. language/embodiment 인자 없음.

### 9-3. 공통 — GUI Deploy

GUI 의 Deploy 버튼 → `record_mode.deploy=True` + `set_start` 자동 set →
evaluate{,_dp}.py 가 ROBOT_OBS 읽어 정책 호출 → ROBOT_ACTION publish →
worker_g1_ctrl, worker_hand_ctrl 가 motor 명령.

**Inference-lag 보상**: `lag_ns = t_after_policy - obs_ts` 만큼 chunk 앞부분 trim →
cross-fade. `--no-lag-compensate` 로 끄고 비교 가능.

## 10. 검증 절차

### 10-1. 오프라인 검증 (하드웨어 0개)

```bash
conda activate teleop
python scripts/verify_offline.py
```

검증 항목 (9 단계):
1. 핵심 의존성 import (numpy/scipy/cv2/pandas)
2. SHM schema mapping + ts field 완정성
3. `mat_tool.cosine_ease` / `se3_interp` 수치 검증
4. `RawStreamBuffer` dedup + `interp_to_axis` linear/zoh
5. `camera_discovery` lazy import + device enumerate
6. `record_collectors` + `parquet_sink.add_extra_column`
7. `G1_29_ArmIK` build (pinocchio + casadi)
8. 모든 worker import + argparse 보존 (vuer hijack 회피 확인)
9. `main.py --help` 가 신규 CLI flags 노출

### 10-2. Quest3 검증 (로봇 0개, Quest3 만)

```bash
# Terminal 1
python main.py --no-robot --vr-input controller --camera none \
               --hand dex3 --waist fixed --head off

# Terminal 2 (GUI: VR 버튼 → START 버튼 클릭)

# Terminal 3
python scripts/verify_quest3.py --rate 2.0 --watch
```

10 가지 체크리스트는 [`docs/DEPLOY_PRO4000.md`](docs/DEPLOY_PRO4000.md) 의
"Quest3-only 동작 검증" 섹션 참고. 절차는 pro4000 / 로컬 동일.

### 10-3. 실 운용 검증

로봇 연결 후 (hoist 현수 + Debug Mode 진입 후):
```bash
python main.py --hand dex3 --camera realsense --vr-input controller --waist fixed --head off --lower-body hoist
```
GUI 의 freq_shm 값 (`g1_freq`, `hand_freq`, `vr_freq`, `camera_freq`) 가 안정적인지
확인. 데이터 수집 시 parquet 의 `raw_ts_*` 컬럼이 stream 간 phase offset 을 보여줘
정렬 품질 사후 검증 가능.

## 11. 트러블슈팅

### Quest 3
| 증상 | 해법 |
|---|---|
| `adb devices` 빈 리스트 | USB 디버깅 OFF — 헤드셋 안 Settings → System → Developer → USB Debugging |
| `unauthorized` | 케이블 재연결 → 헤드셋 dialog 수락 |
| Vuer 페이지 안 열림 | `adb reverse tcp:8012 tcp:8012`, cert.pem 존재 확인 |
| "Horizon Link 가 실행 중인가요?" | Linux 무관 메시지 — 무시 |
| controller 입력 안 잡힘 (hand-tracking 모드만 열림) | vuer 0.0.60 client JS 가 hand-tracking 을 hardcode 요청. `python scripts/patch_vuer_xr.py disable` 적용 후 `status` 확인. Quest 가 controller-only 모드로 진입함 |
| Vuer 브라우저 “Enter VR” 후 WebSocket 연결 실패 | vuer 0.0.60 의 client URL 에 port 누락 (Quest 가 wss://...:443 시도). 동일 patch script 가 함께 수정 |

자세히는 [`docs/QUEST3_SETUP.md`](docs/QUEST3_SETUP.md)

### G1 / 손
| 증상 | 해법 |
|---|---|
| DDS 못 잡음 | `utils/lan_config.yaml` 의 `network_interface` 확인, G1 와 같은 LAN |
| G1 init 시 joint index 어긋남 | `G1_29_JointIndex` enum 이 arm_sdk weight 용으로 35 entry 를 가짐. 본체 init 은 `list(...)[:29]` 만 사용 (worker_g1_ctrl 에서 처리됨) |
| DEX3 grasp 가 약함 | `hand_control/robot_hand_dex3.py` 의 kp/kd 또는 `hand_control/DEX3-1_spec.md` 의 가동범위/한계 참고 |
| DEX3 state subscriber 가 가끔 0 msg | DEX3 보드 publish 자체가 fragile. `worker_hand_ctrl` 초기화 시 timeout 후 release-pose 로 가더라도 다음 메시지가 오면 자연 복귀. 케이블/전원 우선 점검 |
| DEX3 양손 grasp 가 거울대칭 안 됨 | 구 코드 잔재일 수 있음 — `robot_hand_dex3.py` 가 좌/우 부호 반전 (Thumb2/Index/Middle) 적용된 버전인지 확인 |
| 종료 시 G1 가 갑자기 떨어짐 | `g1_whole_control.damp_to_release` 가 호출됐는지 (main.py SIGTERM/exit 경로) 확인. hoist 풀기 전엔 항상 ramp 종료 |
| Inspire 가 안 닫힘 | `robot_hand_inspire.py` 의 normalize range (0~1.7 등) 확인 |
| 손 모양이 어색 | controller-mode 면 `--thumb-bend / --thumb-yaw` 조정 |

### 카메라
| 증상 | 해법 |
|---|---|
| `--camera auto` 가 'none' fallback | USB / 권한 / SDK 설치 확인. `python -c "from utils.camera_discovery import discover_realsense, discover_zed; print(discover_realsense(), discover_zed())"` |
| 멀티-카메라 serial 매핑 어긋남 | `utils/cameras.yaml` 의 `role: ego / wrist_l / wrist_r ↔ serial` 확인. D405 가 lsusb 안 잡히면 USB-C 포트 또는 케이블 우선 교체 |
| pyzed import 실패 | ZED SDK 설치 + `python /usr/local/zed/get_python_api.py` |
| RealSense permission denied | udev rule (`/etc/udev/rules.d/99-realsense-libusb.rules`) |
| GUI ego view 가 검은 화면 | workspace_mask 가 all-zero 상태로 적용되면 까맣게 나옴. `gui/ui_launcher.py` 의 `APPLY_MASK_IN_GUI` + `mask_left_flat.any()` guard 동작 확인 |
| `import cv2` 직후 Qt 플러그인 충돌 (`Could not load the Qt platform plugin "xcb"`) | `opencv-python` 의 번들 Qt 가 PyQt5 와 충돌. `pip uninstall opencv-python && pip install 'opencv-contrib-python-headless<4.11'` 로 교체 (INSTALL.md §4 참고) |

### 데이터 수집
| 증상 | 해법 |
|---|---|
| parquet `raw_ts_*` 가 0 | 해당 SHM writer 가 안 돌고 있음 — worker 로그 확인 |
| 메모리 부족 (긴 에피소드) | `record_collectors.py` 의 maxlen 조정 또는 episode_len 짧게 |
| video frame 적게 저장됨 | camera fps 가 낮음 — 카메라 자체 fps 또는 ZOH-pick 결과 확인 |

### 평가 (eval)
| 증상 | 해법 |
|---|---|
| Deploy 눌러도 모터 안 움직임 | main.py 가 따로 켜져 있는지, GUI Deploy 버튼이 set_start 같이 set 했는지 (Phase A 수정 후 자동) |
| 정책 chunk 너무 stale 보임 | `evaluate.py --lag-log-every 50` 으로 avg/max 측정. 모델 too slow 면 `--slow-hz` 줄이기 |

## 12. 관련 문서 / 파일 인덱스

### 12-1. 문서
- [`USAGE.md`](USAGE.md) — 전체 entry point 명령어/인자/시나리오별 사용법 (수집 → 검증 → 변환 → 학습 → 배포)
- [`Project_tree.md`](Project_tree.md) — 디렉토리 트리 + 파일별 한 줄 설명
- [`GR00T_PIPELINE_GUIDE.md`](GR00T_PIPELINE_GUIDE.md) — 수집 → GR00T 변환(60→20fps) → stats → 학습 → 추론 end-to-end. video.delta_indices=[-20,0] / ACTION_HORIZON=40 / allow_padding 운영 가이드 포함
- [`GR00T_N17_deploy_analysis.md`](GR00T_N17_deploy_analysis.md) — N1.7 Gr00tPolicy 정합 분석 (배포 측 변경 근거)
- [`DP_PIPELINE_CHECKLIST.md`](DP_PIPELINE_CHECKLIST.md) — Diffusion Policy 수집/변환/학습/배포 운영 체크리스트
- [`docs/HARDWARE.md`](docs/HARDWARE.md) — Quest3 / G1 / DEX3 / Inspire / DXL / ZED / RealSense IDL·msg·Hz·latency, SDK 사실 검증 (444줄)
- [`docs/QUEST3_SETUP.md`](docs/QUEST3_SETUP.md) — Linux USB 연결 가이드 (adb, udev, dev mode)
- [`docs/DEPLOY_PRO4000.md`](docs/DEPLOY_PRO4000.md) — pro4000 배포 + Quest3-only 검증 절차
- [`data_refinement/README.md`](data_refinement/README.md) — 데이터셋 변환기 사용법

### 12-2. 주요 entry
- `main.py` — Teleoperation + 데이터 수집
- `evaluate.py` — GR00T N1.7 정책 평가 (별도 conda env)
- `evaluate_dp.py` — Diffusion Policy 평가 (별도 conda env)
- `scripts/verify_offline.py` — 하드웨어 0개 검증 (코드/SHM/IK 빌드)
- `scripts/verify_quest3.py` — Quest3 + IK 검증
- `scripts/patch_vuer_xr.py` — vuer client JS 패치 (hand-tracking OFF + WS port)

### 12-3. 진단 / 검증 스크립트 (루트)
- `check_dex3_recv.py` — DEX3 state DDS 수신 단독 진단 (main.py 끄고 실행)
- `check_dex3_state.py` / `check_dex3_grasp.py` — DEX3 추가 단독 진단
- `check_pipeline_live.py` — main.py RUN 중 모든 SHM 의 실 hz / 신선도 / 지터 측정
- `verify_episode.py` — 저장된 에피소드 (parquet+mp4) 종합 검증 (60Hz 정렬, raw_ts_*, 영상-상태 정합)
- `verify_trajectory.py` — 에피소드 시계열 궤적 정밀 진단 (점프/추종오차/관절범위, PNG 저장)

### 12-4. 데이터 변환
- `data_refinement/convert_to_gr00t.py` — 60→20fps GR00T 학습 형식 변환 (단일 task)
- `data_refinement/convert_to_gr00t_multitask.py` — 다중 task 묶음 변환
- `data_refinement/verify_gr00t_dataset.py` — GR00T 변환 결과 검증
- `data_refinement/convert_to_dp.py` — 60→10fps DP zarr 변환
- `data_refinement/verify_dp_dataset.py` — DP zarr 검증
- 그 외 ACT/merge/inspect/plot/mask 변환기 — `data_refinement/README.md`

### 12-3. 라이선스 & 감사

이 워크스페이스는 Unitree 의 `xr_teleoperate` (DEX3, Inspire FTP 패턴) 와
Vuer (WebXR 서버) 의 영향을 받았습니다. 자세한 SDK 인용은 `docs/HARDWARE.md` 의
"검증 출처" 절 참고.
