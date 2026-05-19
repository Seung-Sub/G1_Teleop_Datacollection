# Hardware Pipeline 사실 정리

이 문서는 G1_Teleoperation 워크스페이스에서 사용하는 각 하드웨어의 통신
방식, 메시지 layout, 권장 주기(Hz), 지연 추정치를 **소스 코드 사실 기반**으로
정리한다. Phase C/D/E 작업 시 추측 없이 인용할 단일 출처.

검증 출처:
- `unitree_sdk2_python/unitree_sdk2py/idl/unitree_hg/msg/dds_/` — IDL 정의
- `xr_teleoperate/teleop/televuer/src/televuer/televuer.py` — Vuer 사용 패턴
- `xr_teleoperate/teleop/robot_control/robot_hand_unitree.py` — DEX3 검증
- `xr_teleoperate/teleop/robot_control/robot_hand_inspire.py` — Inspire FTP 검증
- 자체 코드 `g1_control/`, `hand_control/`, `workers/`

---

## 1. Meta Quest 3 (VR 입력)

### 1.1 연결
- **유선**: USB → ADB. `adb reverse tcp:8012 tcp:8012` (gui/ui_launcher.py:VR 버튼 핸들러)
- **무선**: 같은 LAN 의 HTTPS:8012 (Vuer 기본 포트). cert/key 필요.
- 지연: 유선 ~5-30ms, 무선 ~30-100ms (RTT). 본 워크스페이스는 유선 권장.

### 1.2 Vuer 이벤트 payload
설치 버전: `vuer[all]>=0.0.60,<0.1.0` (setup.py).

#### `CAMERA_MOVE`
```python
event.value["camera"]["matrix"]  # 16 floats column-major SE(3), HMD pose
event.value["camera"]["aspect"]  # float
```

#### `HAND_MOVE`
```python
event.value["left"]   # 25 * 16 = 400 floats (25 landmark joints, 각 16-float SE(3))
event.value["right"]  # 동일
event.value["leftState"]  # dict: pinch, pinchValue, squeeze, squeezeValue
event.value["rightState"]
```
landmark 25개: wrist(0) + 5개 손가락 × 4 joint + finger tip 등.
우리 코드는 tip indices `[4, 9, 14, 19, 24]`, distal `[3, 8, 13, 18, 23]`,
proximal `[2, 7, 12, 17, 22]` 만 SHM 으로 추출 (open_television/television.py:131-184).

#### `CONTROLLER_MOVE`  (xr_teleoperate televuer.py:228-265 기준)
```python
event.value["left"]   # 16 floats column-major SE(3) 좌 컨트롤러 pose
event.value["right"]  # 동일
event.value["leftState"]  # dict
event.value["rightState"]
```

`*State` dict 키:
| 키 | 타입 | 의미 |
|---|---|---|
| `trigger` | bool | press 여부 |
| `triggerValue` | float | 0.0~1.0 analog |
| `squeeze` | bool | grip press 여부 |
| `squeezeValue` | float | 0.0~1.0 grip analog |
| `thumbstick` | bool | touched/clicked (모호 — 두 의미를 모두 포함) |
| `thumbstickValue` | [float, float] | x, y axis ±1 |
| `aButton` | bool | 좌측 컨트롤러 = X 버튼 / 우측 = A 버튼 |
| `bButton` | bool | 좌측 = Y / 우측 = B |

**Oculus 버튼 매핑**: 좌 컨트롤러는 X/Y, 우는 A/B. Vuer 는 좌우 모두 `aButton/bButton` 이름 사용 — 매핑은 우리 코드에서 명시 (Phase C 의 record 트리거).

### 1.3 좌표 변환
- Vuer raw = **OpenXR Convention** (y up, z back, x right)
- 우리는 **Robot Convention** (z up, y left, x front) 사용
- 변환: `T_robot = T_robot_openxr @ T_vuer @ inv(T_robot_openxr)`  (similarity transform)
- 정의는 `open_television/constants.py` 와 `tv_wrapper.py` doc
- hand-mode 는 추가로 `T_to_unitree_left/right_wrist` 회전 곱셈 (URDF 손목 축 보정).
- **controller-mode 는 `T_to_unitree_*_wrist` 곱셈 없음** — Vuer 가 컨트롤러 pose 를
  Unitree URDF 축 규약으로 이미 전달 (xr_teleoperate 권장 패턴).

### 1.4 우리 SHM 매핑
- `TELEVISION.head_rmat / left_wrist_mat / right_wrist_mat`: 4×4 SE(3)
- `TELEVISION.left_hand / right_hand`: (5, 3) — finger tip (hand-mode 만 사용)
- `TELEVISION.right_distal / right_proximal`: (5, 3) — IK 보조용
- `QUEST_CONTROLLER` 전체 — controller-mode 입력 (Phase C 버튼 매핑 input source)

---

## 2. Unitree G1 본체 (DDS via Ethernet)

### 2.1 네트워크
- `utils/lan_config.yaml` 의 `network_interface` (보통 `enp...` 또는 `eth0`)
- 로봇 IP: `192.168.123.x` 기본. DDS multicast.
- DDS Topic:
  - publisher: `rt/lowcmd` (LowCmd_)
  - subscriber: `rt/lowstate` (LowState_)

### 2.2 LowCmd_ / LowState_ IDL (검증된 사실)
```
LowCmd_ {
    mode_pr     : uint8
    mode_machine: uint8                  # G1_29_ArmController.get_mode_machine() 로 동적 획득
    motor_cmd[35] : MotorCmd_           # 35 motor slot
    reserve[4]    : uint32
    crc           : uint32              # CRC().Crc(msg) 계산
}

MotorCmd_ {
    mode  : uint8                       # RIS bitfield (id 4b + status 3b + timeout 1b)
    q     : float32 (rad)
    dq    : float32
    tau   : float32
    kp    : float32
    kd    : float32
    reserve : uint32
}

LowState_ {
    motor_state[35] : MotorState_
    imu_state       : IMUState_  (quaternion, gyroscope, accel)
    ... (battery, sensors)
}

MotorState_ {
    mode  : uint8
    q     : float32 (rad)
    dq    : float32
    ddq   : float32
    tau_est : float32
    temperature[2] : int16
    vol   : float32
    sensor[2] : uint32
    motorstate : uint32
    reserve[4] : uint32
}
```

### 2.3 RIS Mode bit field
```
[7]    timeout flag    (1 bit)
[6:4]  status         (3 bits, 0x01 = position-control enable)
[3:0]  motor id       (4 bits, 모터 인덱스 echo)
```
Encode: `mode = (id & 0xF) | ((status & 0x7) << 4) | ((timeout & 0x1) << 7)`
G1 본체에서는 `worker_g1_ctrl` 가 mode 0 으로 publish (`self.msg.mode_pr = 0`).
DEX3 HandCmd_ 에서는 `status=0x01` 사용 (xr_teleoperate L153).

### 2.4 G1 Joint enum (g1_control/g1_whole_control.py)
```
G1_29_Num_Motors = 35
Leg L0..5 / R6..11  (12)
Waist 12, 13, 14 (yaw, roll, pitch)
Arm L 15..21 / R 22..28 (14)
NotUsed 29..34
```

### 2.5 주기 / 지연
| 채널 | 코드 위치 | 주기 |
|---|---|---|
| lowcmd publish | `_ctrl_motor_state` rate=250Hz | 4ms |
| lowstate subscribe | `_subscribe_motor_state` rate=500Hz | 2ms |
| `worker_g1_ctrl.do_slow` action apply | ACT_HZ=50.0 | 20ms |
| `worker_g1_ctrl.do_fast` obs read | OBS_HZ=300.0 | 3.3ms |

DDS Ethernet 지연 추정 < 5ms.

### 2.6 Mode 프로파일 (`g1_control/joint_setting.yaml`)
- `teleop` (기본): kp/kd/default_dof_pos 29-element vector
- `gr00t`, `gr00t_zed`: 동일 (현재는 teleop 동일 값)

---

## 3. Unitree DEX3-1 (DDS, unitree_hg)

### 3.1 DDS Topic
```
rt/dex3/left/cmd    (HandCmd_)
rt/dex3/right/cmd   (HandCmd_)
rt/dex3/left/state  (HandState_)
rt/dex3/right/state (HandState_)
```

### 3.2 HandCmd_ / HandState_ IDL
```
HandCmd_ {
    motor_cmd : sequence<MotorCmd_>   # length 7
    reserve[4] : uint32
}

HandState_ {
    motor_state : sequence<MotorState_>          # length 7
    press_sensor_state : sequence<PressSensorState_>   # ← DEX3 내장 접촉 센서 (현재 미활용)
    imu_state : IMUState_                        # 손 IMU
    power_v, power_a, system_v, device_v : float32
    error[2] : uint32
    reserve[2] : uint32
}
```
**미활용 자산**: `press_sensor_state` 는 DEX3 자체 손가락 압력 센서. Phase D 데이터
수집에서 옵션으로 활용 가능.

### 3.3 모터 IntEnum 순서 (Unitree 하드웨어 고정)
```
Dex3_1_Left_JointIndex  = [Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1]  # idx 0..6
Dex3_1_Right_JointIndex = [Thumb0, Thumb1, Thumb2, Index0,  Index1,  Middle0, Middle1] # idx 0..6
```
좌·우가 다름에 주의 — `motor_cmd[id]` 순서가 좌우 hardware 별로 다른 enum 을 따른다.

### 3.4 q 단위 / kp / kd
- `q` 단위: **raw radian** (Inspire 와 다름 — Inspire 는 0..1 normalized)
- 권장 게인: `kp=1.5, kd=0.2` (xr_teleoperate L147-148 검증)
- mode: `_RIS_Mode(id=id, status=0x01).to_uint8()` (position control enable)

### 3.5 URDF joint limit (rad)
좌·우 동일 limit:
```
thumb_0 (yaw)  : [-1.04719755,  1.04719755]
thumb_1 (bend) : [-0.72431163,  0.92]
thumb_2        : [ 0.0,         1.74532925]   (0=open, 1.745=closed)
middle_0       : [-1.57079632,  0.0]          (-1.57=closed)
middle_1       : [-1.74532925,  0.0]
index_0        : [-1.57079632,  0.0]
index_1        : [-1.74532925,  0.0]
```
fingers 의 굽힘 부호는 좌우 동일 — 음수 방향이 닫힘.

### 3.6 retargeting yml 호환성
- xr_teleoperate `unitree_dex3.yml` 은 신버전 dex_retargeting (DexPilot + Vector
  split) 가정 → `target_link_human_indices_dexpilot` 등.
- 우리 `hand_control/dex_retargeting/` fork 는 구버전 API → 단일
  `target_link_human_indices` 만 지원.
- 우리 `hand_control/unitree_dex3_hand/unitree_dex3.yml` 은 dexpilot 값을 단일
  키로 옮긴 축약형 (commit 5a0053a).

### 3.7 hand-tracking 입력 shape
xr_teleoperate `robot_hand_unitree.py:187` 패턴:
```python
ref = hand_data[indices[1,:]] - hand_data[indices[0,:]]   # 25 landmark 중 indexing
left_q = retarget(ref)[left_dex_retargeting_to_hardware]
```
**우리 television.py 는 5 tip 만 노출** — 25 landmark 차분 벡터 미가능 →
controller-mode 만 hand-tracking 미지원. `hand_control/robot_hand_dex3.py:290-305` 에
명시.

### 3.8 주기
| 채널 | 주기 |
|---|---|
| HandCmd_ publish | `Dex3_Controller.control_process` fps=100Hz |
| HandState_ subscribe | `_subscribe_hand_state` sleep(0.002) ≈ 500Hz |

---

## 4. Inspire RH56 (DDS via unitree_sdk2py + Modbus touch)

### 4.1 DDS Topic
```
rt/inspire_hand/ctrl/l   (inspire_dds.inspire_hand_ctrl)
rt/inspire_hand/ctrl/r   (inspire_dds.inspire_hand_ctrl)
rt/inspire_hand/state/l  (inspire_dds.inspire_hand_state)
rt/inspire_hand/state/r  (inspire_dds.inspire_hand_state)
```

### 4.2 Msg payload (`inspire_sdkpy`)
- `inspire_hand_ctrl.angle_set` = `list[int][6]`, 값 0..1000 (mille). 0=닫힘, 1000=열림.
- `inspire_hand_ctrl.mode` = `0b0001` (Angle control, 검증됨 — xr_teleoperate L247)
- `inspire_hand_state.angle_act` = `list[int][6]`, 0..1000 (mille)
- 단일 hand 당 6 motor (Inspire FTP 변종)

### 4.3 IntEnum (state msg 시퀀스 순서)
```
Right hand : [pinky=0, ring=1, middle=2, index=3, thumb-bend=4, thumb-rotation=5]
Left  hand : [pinky=6, ring=7, middle=8, index=9, thumb-bend=10, thumb-rotation=11]
```
좌우 topic 이 분리되어 있으므로 실제 access 는 각 메시지 angle_act[0..5] 6개씩.

### 4.4 q normalize 룰
URDF rad → Inspire normalized [0,1] 변환:
```python
def normalize(val, min_val, max_val):
    return clip((max_val - val) / (max_val - min_val), 0, 1)
# idx 0..3 : range [0.0, 1.7]
# idx 4    : range [0.0, 0.5]
# idx 5    : range [-0.1, 1.3]
```
normalized 1.0 = 열림, 0.0 = 닫힘. inspire_hand_ctrl.angle_set 에 `int(v * 1000)` 으로 scale.

### 4.5 Modbus touch sensor
- IP: 좌 `192.168.123.211`, 우 `192.168.123.210` (main.py 의 worker_hand_dds args)
- `inspire_sdk.ModbusDataHandler(network, ip, LR='l'|'r', device_id=1).read()` →
  dict 반환. 키는 `fingerone~five_tip/top/palm/middle_touch` + `palm_touch`.
  `LEFT/RIGHT_TOUCH_SENSOR_LAYOUT` SHM 스키마와 매핑 (worker_hand_dds.py:23).
- 주기: 현재 `time.sleep(0.001)` (≈1kHz polling, Modbus TCP 가 받쳐주지 못할
  수 있음). **권장 측정값으로 조정 필요** — Phase B 후속 작업.
- Touch dict 키 정확한 명세는 pro4000 환경의 `inspire_sdkpy` 설치본 확인 필요.

### 4.6 주기
| 채널 | 주기 |
|---|---|
| inspire DDS publish (Inspire_Controller.control_process) | fps=100Hz |
| inspire DDS state subscribe (_subscribe_hand_state) | sleep(0.002) ≈ 500Hz |
| Modbus touch read (worker_hand_dds) | sleep(0.001) ≈ 1kHz (과도, 조정 필요) |

---

## 5. Dynamixel XL (Head camera 2 motor)

### 5.1 통신
- 시리얼 `/dev/ttyUSB0`, baud 1000000, Protocol 2.0
- `DXL1_ID=1, DXL2_ID=2` (camera_yaw, camera_pitch)
- 라이브러리: `dynamixel_sdk.PortHandler/PacketHandler/GroupSyncWrite/GroupSyncRead`

### 5.2 주소
```
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116  (LEN=4)
ADDR_PRESENT_POSITION = 132  (LEN=4)
OPERATING_MODE        = 11  → POSITION_CONTROL_MODE = 3
```

### 5.3 tick ↔ rad 변환
```
TICKS_PER_REV    = 4096
TICKS_PER_RAD    = 4096 / (2π) ≈ 651.898
ZERO_OFFSET_TICK = 2048   # rad 0 → tick 2048
tick = ZERO_OFFSET_TICK + rad * TICKS_PER_RAD
```
허용 범위: 0..4095 (clamp).

### 5.4 Ready-pose
`__init__` 의 `self.q_target = np.array([2048, 1934])`.
- yaw=2048 (center)
- pitch=1934 → rad = (1934-2048)/TICKS_PER_RAD = **-0.17486 rad** (≈-10°, 카메라 mount 의 자연스러운 down-tilt)

**Phase A 수정 후 (commit 미정)**: `_ctrl_motor_state` 의 hardcode override
제거. `ctrl_dynamixel(q_target)` 호출이 실제로 반영되며, controller-mode 에서는
`worker_g1_ik.HEAD_READY_RAD = [0.0, -0.17486]` 가 published → DXL 이
[2048, 1934] 로 수렴.

### 5.5 주기
| 채널 | 주기 |
|---|---|
| publish (`_ctrl_motor_state`) | control_dt = 1/100 → 100Hz |
| subscribe (`_subscribe_motor_state`) | sleep(1/100) → 100Hz |

---

## 6. ZED 스테레오 카메라

### 6.1 연결
- 송신측: 별도 PC 의 `192.168.5.11:30000` 으로 stream out (`zed_sender`)
- 수신측: 본 워크스페이스 `worker_zed.py` 가 `init_params.set_from_stream(IP, PORT)` 로 attach
- `DEPTH_MODE.PERFORMANCE`, `UNIT.METER`

### 6.2 retrieve
```python
zed.retrieve_image(left_mat,  sl.VIEW.LEFT)
zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
```
이미지: BGRA → cv2 BGR → 480×640 resize. depth: float32 480×640 (resized).

### 6.3 부가 처리
- ArUco 4-marker (ID 0..3) 감지 → workspace mask polygon
- mask_control_shm 토글로 mask 갱신 ON/OFF
- 모든 SHM (CAMERA, ARUCO_MARKERS, WORKSPACE_MASK, DEPTH_MAP) 매 frame write

### 6.4 주기
ZED grab loop 는 명시 rate 없음 — `grab()` blocking 으로 SDK 가 native 30fps 또는 60fps 결정. 보통 30Hz.

---

## 7. RealSense D435

### 7.1 연결 / 스트림
```python
rs.pipeline().start(rs.config().enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30))
```
- color 640×480 @ 30Hz (BGR8)
- depth 비활성 (주석 처리됨)

### 7.2 주기
`Rate(30.0)`. wait_for_frames timeout 1초.

---

## 8. SHM 토폴로지 (요약 — Phase D 정렬 pipeline input)

| SHM | Owner | Reader | 주기 | 핵심 필드 |
|---|---|---|---|---|
| `camera_shm` (CAMERA) | worker_zed / worker_camera | UI, worker_record, worker_deploy_policy | 30Hz | `camera_left/right/realsense` |
| `television_shm` (TELEVISION) | worker_vr | worker_g1_ik, worker_hand_ctrl | 50Hz | `head_rmat`, `left/right_wrist_mat`, `left/right_hand` |
| `quest_controller_shm` (QUEST_CONTROLLER) | worker_vr | worker_g1_ik, hand_controller | 50Hz | button/trigger/squeeze + ctrl_mat |
| `aruco_shm` / `workspace_mask_shm` / `depth_map_shm` | worker_zed | UI, deploy | 30Hz | mask, depth |
| `record_task/episode/mode_shm` | GUI | worker_record, worker_g1_ik, worker_hand_ctrl, deploy | event-driven | task_name, num/episode_len, start/done/reset/replay/home/deploy |
| `teleop_config_shm` | main.py 시작 시 1회 | record/UI | one-shot | hand/camera/vr_input int code |
| `left_touch_shm` / `right_touch_shm` | worker_hand_l/r_dds | UI, record | 100Hz+ | finger touch matrices |
| `freq_shm` | 모든 worker | UI | 매 cycle | per-worker actual Hz |
| `gr00t_shm` | GUI (Deploy 버튼) | evaluate.py | event-driven | task_name (language instruction) |
| `robot_obs_shm` (ROBOT_OBS) | worker_g1_ctrl + worker_hand_ctrl | worker_record, worker_deploy_policy, UI | 300Hz / 50Hz | leg/waist/head/arm/hand observation |
| `robot_action_shm` (ROBOT_ACTION) | worker_g1_ik + worker_hand_ctrl + worker_deploy_policy | worker_g1_ctrl, worker_record | 50Hz | leg/waist/head/arm/hand action |
| `mask_control_shm` | UI | worker_zed | event-driven | mask_enabled / generate_new |

### Hz mismatch
- camera 30Hz vs robot_obs 300Hz vs television 50Hz vs record 20Hz → Phase D 의 *_ts 기반 보간 정렬이 학습 데이터 품질에 필수.

---

## 9. 지연 (latency) 예산 — 측정 가이드

각 worker 의 `freq_shm.write_data(*_freq=...)` 가 실측 Hz 를 SHM 에 publish.
실 운영 중 GUI 의 `read_from_shm` 가 표시 — Phase B 후속에서 그래프 위젯
부활시켜 정량화 권장.

| 경로 | 추정 latency |
|---|---|
| Quest3 wired → Vuer event → worker_vr SHM write | 10~30ms |
| robot_obs (DXL/G1) → SHM | 2~5ms |
| robot_obs → IK → action → DDS → motor | 30~50ms (50Hz 한 cycle + DDS round-trip) |
| Camera grab → SHM | 30~50ms |
| Policy inference (slow_hz 20Hz, GR00T) | 30~80ms (모델/GPU 의존) |

**측정 권장 항목** (Phase B 후속, hardware 연결 후):
1. 각 SHM 의 `*_ts` (Phase D 도입 예정) 값으로 worker 간 latency 분포
2. Vuer event → SHM write 까지의 timestamp gap
3. policy.get_action() 의 입출력 시각
4. action publish → motor angle 도달까지의 closed-loop latency

---

## 10. 위 분석에서 확인된 정합/불일치 요약

### 정합 확인
- DEX3 IntEnum 좌·우 순서 (Middle/Index swap on right) ✓
- DEX3 kp=1.5/kd=0.2 ✓
- Inspire mode=0b0001 + angle 0..1000 mille ✓
- Inspire normalize range per idx ✓
- Vuer controller payload 키 (trigger/squeeze/thumbstick/aButton/bButton) ✓

### 우리 측 의도적 분기
- DEX3 retargeting `left_q_target ← left_dex_retargeting_to_hardware` (xr_teleoperate L190 오타와 다름)
- DEX3 yml 단일 `target_link_human_indices` (구버전 dex_retargeting fork 호환)
- DEX3 hand-tracking 미배선 (television.py 가 5 tip 만 노출)
- Vuer monocular branch dead path (worker_vr 가 binocular=True 강제)

### 미해결 / 후속 확인 필요
- `inspire_sdkpy.inspire_sdk.ModbusDataHandler.read()` 응답 dict 키 (pro4000 머신 확인)
- Modbus polling 주기 1kHz 적정성
- Quest3 wired vs wireless 실측 RTT
- ZED stream native fps (PERFORMANCE depth 모드)
- DEX3 press_sensor_state 활용 여부
- Head IK 시 URDF↔DXL 좌표 offset 일관화 (현재 controller-mode 만 ready_pose 강제)
