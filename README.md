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
  - 에피소드 close 시 50 Hz 공통 시간축 → linear (continuous) / ZOH (image, discrete) 보간
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
                            (50Hz)         QUEST_CONTROLLER SHM
                                              │
                                              ▼
                                  worker_g1_ik (50Hz)
                                  - SE(3) clutch on grip-hold
                                  - HMD R_delta → waist (옵션)
                                  - Right-A 3s cosine recovery
                                  - DEX3/Inspire common interface
                                              │
                          ROBOT_ACTION SHM ◄──┘
                                  │
                                  ▼
                          worker_g1_ctrl (250Hz cmd / 500Hz obs)
                          - LowCmd_ / LowState_ DDS
                          - Dynamixel head (--head dxl)
                          - ROBOT_OBS SHM (300Hz)

                  worker_hand_ctrl (50Hz)
                  ├─ Inspire DDS rt/inspire_hand/ctrl/{l,r}
                  └─ DEX3 DDS    rt/dex3/{left,right}/{cmd,state}

                  worker_zed (~30Hz)  or  worker_camera (RealSense ~30Hz)
                          │
                          ▼
                      CAMERA SHM ─► ego_left/right or realsense + ts

                  worker_record (FSM 외부 20Hz)
                  ├─ start: RecordCollectors 시작 (3 poller thread)
                  │   - robot poller 1kHz : obs_body / obs_hand / action_body/hand
                  │   - tv poller 200Hz   : television + controller
                  │   - camera poller 100Hz : zed left/right or realsense
                  ├─ stop: stop_and_dump → align_and_save_episode
                  │   - common_time_axis 50Hz (intersection)
                  │   - linear-interp continuous, ZOH images
                  │   - ParquetSink + extra_columns(raw_ts_*) + VideoSink
                  └─ LeRobot v2.1 dataset layout 유지
```

### 2-2. 모듈 책임 한 줄 요약

| 영역 | 모듈 | 역할 |
|---|---|---|
| Entry | `main.py` | CLI parse → SHM owner-create → worker spawn |
| Eval entry | `evaluate.py` | 별도 conda env 에서 GR00T 정책 평가 (lag-compensate) |
| VR 입력 | `open_television/television.py`, `tv_wrapper.py`, `workers/worker_vr.py` | Vuer 이벤트 → SHM, OpenXR→Robot 기저 변환 |
| IK | `g1_control/g1_ik.py`, `workers/worker_g1_ik.py` | Pinocchio+CasADi IPOPT, clutch + recovery + waist clutch |
| G1 본체 | `g1_control/g1_whole_control.py`, `workers/worker_g1_ctrl.py` | LowCmd_/LowState_ DualRate 50/300Hz |
| 머리 | `g1_control/g1_head_dynamixel.py` | XL 시리즈 2모터 syncwrite (`--head dxl`) |
| 손 | `hand_control/robot_hand_inspire.py`, `robot_hand_dex3.py`, `workers/worker_hand_ctrl.py` | DDS, controller-mode trigger toggle, hand-mode retargeting |
| 손 터치 | `workers/worker_hand_dds.py` | Inspire Modbus (DEX3 미해당) |
| 카메라 | `workers/worker_zed.py`, `worker_camera.py`, `utils/camera_discovery.py` | direct/stream, RealSense first auto-detect |
| 정렬 | `utils/raw_stream.py`, `utils/align.py`, `utils/record_collectors.py` | RawStreamBuffer dedup + interp_to_axis + common_time_axis |
| 저장 | `utils/parquet_sink.py`, `utils/video_sink.py` | LeRobot v2.1 + raw_ts_* extra |
| 정책 평가 | `workers/worker_deploy_policy.py` | slow(20Hz)/fast(50Hz) + cross-fade + lag trim |
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

```bash
# 1) conda env (Python 3.8 권장)
conda create -n teleop python=3.8 -y
conda activate teleop
python -m pip install --upgrade pip wheel setuptools

# 2) PyTorch 등 학습 의존성 (옵션, eval 시 필요)
# 3) 본 워크스페이스 install
cd /path/to/G1_Teleoperation
pip install -e .

# 4) 외부 SDK 별도 설치
#   - ZED SDK     : https://www.stereolabs.com/developers/release
#   - pyrealsense2: pip install pyrealsense2
#   - vuer        : pip install 'vuer[all]>=0.0.60,<0.1.0'
#   - dynamixel_sdk: pip install dynamixel_sdk
#   - unitree_sdk2py / inspire_sdkpy 는 각각 git clone 후 pip install -e .
```

> 평가 (`evaluate.py`) 는 별도 conda env (예: `gr00t`) 에서 실행 권장. main.py 의
> teleop env 와 충돌 회피.

자세한 단계는 본 워크스페이스의 기존 환경 설치 자료 또는
[`docs/HARDWARE.md`](docs/HARDWARE.md) 의 SDK 절을 참고.

## 5. 실행 모드 + CLI

### 5-1. `main.py` 핵심 옵션

```text
--hand     {inspire, dex3}            손 하드웨어 (default: inspire)
--camera   {auto, zed, realsense,     'auto' 면 RealSense 우선 자동감지 → ZED fallback.
            none, <serial>}            'none' 카메라 없이. serial 직접 지정 가능.
--camera-role  ego (default)          멀티-cam 확장용 stub
--zed-mode {direct, stream}           direct=USB, stream=set_from_stream(legacy)
--vr-input {hand, controller}         (default: hand)
--waist    {hmd, fixed}               HMD→waist 매핑 활성 / 고정
--head     {dxl, off}                 Dynamixel head 사용 / 미사용
--thumb-bend, --thumb-yaw  (0.0~1.0)  controller-mode 엄지 사전 자세
--no-robot                            G1/hand worker 생략 (set_g1/set_hand 자동 set)
```

### 5-2. 자주 쓰는 조합

```bash
# 일반 운용 (DEX3 + RealSense + controller-mode, HMD 목에 걸고)
python main.py --hand dex3 --camera auto --vr-input controller \
               --waist hmd --head dxl --thumb-bend 0.5 --thumb-yaw 0.5

# Inspire + ZED stereo + waist 고정 (조각 작업 등 허리 안정 필요)
python main.py --hand inspire --camera zed --vr-input controller \
               --waist fixed --head dxl

# 데이터 수집 만 (머리 흔들림 없이)
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off

# Quest3 검증 only (G1/hand 없이)
python main.py --no-robot --vr-input controller --camera none \
               --hand dex3 --waist fixed --head off
```

## 6. 실 운용 시나리오

### 6-1. 운용 흐름

1. 로봇 ON, G1 / hand / DXL Ethernet/USB 연결
2. 카메라 USB 연결 (자동감지 됨)
3. Quest 3 PC 에 USB 유선 연결 — `adb devices` 가 `device` 상태인지 확인
   ([docs/QUEST3_SETUP.md](docs/QUEST3_SETUP.md))
4. `python main.py --hand dex3 --camera auto --vr-input controller --waist hmd`
5. GUI 띄워지면 G1 → Hand → VR 순서로 버튼 클릭 → adb reverse 자동 실행
6. Quest 3 헤드셋 내 브라우저에서 `https://127.0.0.1:8012` 접속 → "Enter VR"
7. GUI START 버튼 → worker FSM RUN 진입
8. 컨트롤러 **Grip 누른 채** 자연스러운 자세 잡고 조작 시작

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
| **좌/우 Squeeze (grip)** | 누른 동안만 EE clutch 추종 (떼면 freeze) |
| **좌/우 Trigger** | rising-edge 마다 해당 손 grasp toggle (close↔open) |
| **HMD orientation** | grip 누른 동안 (R_delta) waist 매핑 (`--waist hmd`) |
| **Right A** | Ready pose 복귀 (3초 ease + 입력 lockout) |
| **Right B** | SET (task_name 은 GUI 에서 미리 입력) |
| **Left X** | Record start toggle (RECORDING 중이면 early-stop+save) |
| **Left Y** | Drop (현 에피소드 폐기) |
| **좌/우 Thumbstick** | 미사용 |

> `--vr-input hand` (hand-tracking) 모드는 5 finger tip 만 SHM 으로 노출하는 현재
> 구조상 Inspire 만 정상 retargeting. DEX3 hand-tracking 은 25 landmark 필요해 
> 안전 release 자세만 publish 됨 → DEX3 + hand-tracking 시 main.py 에 경고.

## 8. 데이터 수집 파이프라인

### 8-1. SHM 시각 동기화

8 개 streaming SHM 마다 `*_ts` 필드 (np.int64, `time.perf_counter_ns()`):

| SHM | TS field(s) | writer | 주기 |
|---|---|---|---|
| ROBOT_OBS | `obs_body_ts`, `obs_hand_ts` | worker_g1_ctrl, worker_hand_ctrl | 300Hz / 50Hz |
| ROBOT_ACTION | `action_body_ts`, `action_hand_ts` | worker_g1_ik, worker_hand_ctrl, worker_deploy_policy | 50Hz |
| CAMERA | `camera_zed_ts`, `camera_realsense_ts` | worker_zed, worker_camera | ~30Hz |
| TELEVISION | `television_ts` | worker_vr | 50Hz |
| QUEST_CONTROLLER | `controller_ts` | worker_vr | 50Hz |
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
2. `utils/align.common_time_axis` → 채워진 모든 stream 의 intersection 에 50 Hz uniform 축
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

별도 conda env (`gr00t` 등) 에서 GR00T 정책 실행:

```bash
# Terminal 1 (teleop env)
conda activate teleop
python main.py --hand dex3 --camera auto --vr-input controller --waist fixed --head dxl

# Terminal 2 (gr00t env)
conda activate gr00t
python evaluate.py --mode gr00t_zed --model-path /path/to/checkpoint-XXXXX \
    --data-config-key unitree_g1_inspire --action-method tem \
    --lag-compensate --lag-log-every 50
```

GUI 의 Deploy 버튼 → `record_mode.deploy=True` + `set_start` 자동 set →
evaluate.py 가 ROBOT_OBS 읽어 정책 호출 → ROBOT_ACTION publish → worker_g1_ctrl,
worker_hand_ctrl 가 motor 명령.

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

로봇 연결 후:
```bash
python main.py --hand dex3 --camera auto --vr-input controller --waist hmd
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
| controller 입력 안 잡힘 | `--vr-input controller`, 헤드셋 안 controller 활성 |

자세히는 [`docs/QUEST3_SETUP.md`](docs/QUEST3_SETUP.md)

### G1 / 손
| 증상 | 해법 |
|---|---|
| DDS 못 잡음 | `utils/lan_config.yaml` 의 `network_interface` 확인, G1 와 같은 LAN |
| DEX3 grasp 가 약함 | `hand_control/robot_hand_dex3.py` 의 kp/kd 또는 URDF limit constant 조정 |
| Inspire 가 안 닫힘 | `robot_hand_inspire.py` 의 normalize range (0~1.7 등) 확인 |
| 손 모양이 어색 | controller-mode 면 `--thumb-bend / --thumb-yaw` 조정 |

### 카메라
| 증상 | 해법 |
|---|---|
| `--camera auto` 가 'none' fallback | USB / 권한 / SDK 설치 확인. `python -c "from utils.camera_discovery import discover_realsense, discover_zed; print(discover_realsense(), discover_zed())"` |
| pyzed import 실패 | ZED SDK 설치 + `python /usr/local/zed/get_python_api.py` |
| RealSense permission denied | udev rule (`/etc/udev/rules.d/99-realsense-libusb.rules`) |

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
- [`Project_tree.md`](Project_tree.md) — 디렉토리 트리 + 파일별 한 줄 설명
- [`docs/HARDWARE.md`](docs/HARDWARE.md) — Quest3 / G1 / DEX3 / Inspire / DXL / ZED / RealSense IDL·msg·Hz·latency, SDK 사실 검증 (444줄)
- [`docs/QUEST3_SETUP.md`](docs/QUEST3_SETUP.md) — Linux USB 연결 가이드 (adb, udev, dev mode)
- [`docs/DEPLOY_PRO4000.md`](docs/DEPLOY_PRO4000.md) — pro4000 배포 + Quest3-only 검증 절차
- [`data_refinement/README.md`](data_refinement/README.md) — 데이터셋 변환기 사용법

### 12-2. 주요 entry
- `main.py` — Teleoperation + 데이터 수집
- `evaluate.py` — 외부 정책 평가 (별도 conda env)
- `scripts/verify_offline.py` — 하드웨어 0개 검증
- `scripts/verify_quest3.py` — Quest3 + IK 검증

### 12-3. 라이선스 & 감사

이 워크스페이스는 Unitree 의 `xr_teleoperate` (DEX3, Inspire FTP 패턴) 와
Vuer (WebXR 서버) 의 영향을 받았습니다. 자세한 SDK 인용은 `docs/HARDWARE.md` 의
"검증 출처" 절 참고.
