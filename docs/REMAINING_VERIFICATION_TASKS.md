# 실 하드웨어 연결 후 검증 / 후속 작업 목록

본 문서는 **코드 변경으로 미리 처리할 수 없고 실제 하드웨어 (G1, DEX3, Inspire,
RealSense D435i + D405 × 2, ZED 2i / Mini, Meta Quest 3, Dynamixel) 연결 후에만
확인 가능한 항목들** 의 단일 추적 파일이다.

작성 시점: 2026-05-20 (Phase K1~K8 + Phase L1~L6 commit 후).
직전 commit: Phase L 시리즈 (4f53e4d 이후).

각 항목은 다음 형식:
- **트리거**: 어떤 사건 후 이 항목을 확인할지
- **확인 방법**: 실제 측정/실험 절차
- **결과 → 후속 코드 변경**: 측정값에 따라 어떤 코드를 추가/수정할지

---

## L0. 환경 / 코드 무결성 (선행 점검)

각 신규 머신에서 hardware 연결 *전* 한 번 통과해야 함.

```bash
conda activate teleop
QT_QPA_PLATFORM=offscreen python scripts/verify_offline.py
# → SUMMARY  PASS=9  FAIL=0
```

이게 안 되면 docs/INSTALL.md §3 (vuer/params_proto monkey-patch), §4-5 (logging_mp
alias), §8 (cert.pem) 재확인. README §11 트러블슈팅 참고.

---

## L1. 카메라 — RealSense 3대 동시 운용 검증

### L1-A. USB 대역폭 확인 (트리거: D435i + D405 × 2 USB 연결 직후)

**확인 방법**:
```bash
conda activate teleop
cd /path/to/G1_Teleoperation

# 1) 디바이스 인식 + USB 등급
python -c "
from utils.camera_discovery import discover_realsense
for d in discover_realsense():
    print(d)
"
# 출력에 D435i serial + D405 serial 2개 (총 3대) 가 나와야 함

# 2) cameras.yaml 채우기 — 해당 serial 들을 ego/wrist_l/wrist_r 에 매핑.

# 3) main.py 실행 (foreground) — 각 worker 의 USB descriptor 로그 관찰
python main.py --hand dex3 --cameras-config utils/cameras.yaml \
               --vr-input controller --waist fixed --head off --no-robot
```

**기대 로그**:
```
[Realsense:ego]     USB type descriptor = 3.2
[Realsense:wrist_l] USB type descriptor = 3.2
[Realsense:wrist_r] USB type descriptor = 3.2
[Realsense:ego]     pipeline started.
[Realsense:wrist_l] pipeline started.
[Realsense:wrist_r] pipeline started.
[Realsense:ego]     timestamp_domain = GLOBAL_TIME
```

USB2.x 가 잡히면 worker 가 `USB2.x detected — 3대 동시 grab 시 대역폭 부족 위험`
경고 출력. 그 경우:
- 카메라 3대를 PC 의 서로 다른 USB3 root hub 에 분산 연결 (lsusb -t 로 트리 확인)
- 또는 D405 의 fps 를 30 → 15 로 낮춤 (utils/cameras.yaml 에 fps:15 추가 + worker_camera 에서 사용 로직 추가 — 현재 코드는 30 고정, 후속 작업)

### L1-D. AFFINITY 키 패턴 매칭 (SUPPLEMENT 보강)

> Phase M5 (commit 후) 적용 — main.py 의 AFFINITY 키가 K7 이전 워커 이름 (`'WORKER_ZED'`)
> 으로 남아있어 K7 이후 카메라 워커 (`WORKER_RS_EGO`, `WORKER_RS_WRIST_L`, `WORKER_ZED_EGO`)
> 와 매칭 안 됨. 동작은 하지만 CPU 코어 격리 상실 → 코어 경합 가능성.

**확인**: 실 운용 중 `htop` 로 카메라 워커가 의도한 코어에 묶이는지. 안 묶이면:
- 코드: M5 (PART3 §L1 보강) 가 prefix 매칭으로 갱신함.
- 머신 별 코어 수에 맞춰 AFFINITY 의 CPU 번호 (현재 18~23 하드코딩) 조정.

**트리거**: 멀티카메라 jitter (§L1-B) 가 큰데 USB 는 정상일 때 — 코어 경합이 원인 가능성.

### L1-B. 프레임 드롭 + jitter 측정

**확인 방법**: 한 에피소드 (10~30s) 녹화 후 parquet 의 `raw_ts_camera_<role>` 컬럼
분석.

```python
import pandas as pd, numpy as np
df = pd.read_parquet('record/<task>/data/chunk-000/episode_000000.parquet')
for role in ['ego', 'wrist_l', 'wrist_r']:
    col = f'raw_ts_camera_{role}'
    if col not in df.columns: continue
    ts = df[col].to_numpy(dtype=np.int64)
    diffs_ms = np.diff(ts) / 1e6
    print(f'{role}: mean_dt={diffs_ms.mean():.2f}ms  std={diffs_ms.std():.2f}ms '
          f'max_gap={diffs_ms.max():.2f}ms')
```

**기대**: 30fps → mean_dt ≈ 33.3ms, std < 5ms, max_gap < 100ms. max_gap 이 큰 경우
USB 경합 또는 main.py 의 다른 worker process 의 CPU 점유 영향.

### L1-C. Global Time 동작 확인

worker_camera 의 `timestamp_domain` 로그가 `GLOBAL_TIME` 이 아니라 `SYSTEM_TIME`
또는 `HARDWARE_CLOCK` 으로 떨어지면, `perf_counter_ns()` fallback 으로 동작. 그
경우 다른 모달리티 (G1/hand DDS recv_ts) 와의 align 정확도가 낮아진다. 원인:
- D405 일부 펌웨어가 Global Time 미지원 → librealsense 업그레이드 + 펌웨어 업데이트
  (`rs-fw-update --recover -f <fw>`)
- 또는 metadata 헤더 (uvcvideo 모듈) 미활성 — `/etc/modprobe.d/realsense-libuvc.conf`
  에 `options uvcvideo nodrop=1` 같은 설정 필요할 수 있음 (Intel 공식 가이드 참고)

---

## L2. ZED 카메라 (사용 시)

### L2-A. ZED SDK + pyzed 설치 확인

`docs/INSTALL.md §5-2` 절차. 설치 후:
```bash
python -c "from utils.camera_discovery import discover_zed; print(discover_zed())"
```

ZED 2i / Mini 가 보여야 함.

### L2-B. CAMERA_VIEW schema 호환 검증

Phase K7-A 에서 worker_zed.py 를 새 `CAMERA_VIEW` schema (frame_left/right/ts/
is_stereo) 로 마이그레이션. 실제 ZED 연결 후 첫 grab 에서 frame_left/right 가
정상 채워지는지, is_stereo=1 인지 확인:

```python
import multiprocessing as mp
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA_VIEW
s = SharedMemoryManager(CAMERA_VIEW, mp.Lock(), 'camera_shm')
d = s.read_data()
print('frame_ts:', int(d['frame_ts']))
print('is_stereo:', int(d['is_stereo']))
print('left shape:', d['frame_left'].shape, 'mean:', d['frame_left'].mean())
print('right shape:', d['frame_right'].shape, 'mean:', d['frame_right'].mean())
```

---

## L3. Quest 3 정밀 검증 (Phase H 보강)

### L3-A. 컨트롤러 입력 → SHM ts 흐름

main.py `--no-robot` 으로 실행 후 `scripts/verify_quest3.py --watch` 로 다음 시나리오:

1. HMD 머리 움직임 → `HMD head trans` 변화 + `television_ts` 증가
2. Left grip 누름 → `L: grip` 0.0→1.0, `controller_ts` 증가
3. Left grip 누른 채 컨트롤러 이동 → `action_arm` (worker_g1_ik 가 IK 계산) 변화
4. Right A 누름 → 3초 동안 `action_arm` 이 ready pose 로 cosine ease (`action_body_ts`
   매 50Hz 증가). 이 동안 다른 input (grip/trigger/Left X) 무시 확인
5. Left X → `RecMode.start` 토글, Left Y → `RecMode.reset=True`, Right B → SET

### L3-B. Vuer 페이지 latency

Quest 3 안 브라우저에서 `https://127.0.0.1:8012` 접속 후 "Enter VR". HMD 움직임이
verify_quest3.py 출력에 반영되는 지연을 체감 (목표 < 50ms). 더 크면:
- USB 케이블 데이터 라인 품질 (충전 전용 케이블 가능성)
- Quest 3 의 다른 백그라운드 앱 종료
- Vuer FFmpeg pipe 의 quality=80 인 jpeg 인코딩 부하

### L3-C. Recovery 동안 lockout 동작

Right A 를 누른 직후 즉시 Left grip + 컨트롤러 흔들기. 3초 동안 worker_g1_ik 의
log `[G1_IK] Recovery COMPLETE` 가 뜨기 전까지 `action_arm` 이 cosine 궤적만 따르고
grip 입력에 반응하지 않아야 함.

---

## L4. G1 + DEX3 본체 연결 후

### L4-A. DDS network_interface 확인

`utils/lan_config.yaml` 의 `network_interface` 가 G1 와 같은 네트워크 인터페이스
이름인지. `ip a` 로 확인 → 다르면 yaml 수정.

### L4-B. obs_body_ts 의 do_fast 중복 dedup 검증 (Phase K2 적용 후)

```python
# 한 에피소드 녹화 후 raw_ts_obs_body 의 연속 차분 분포 확인
ts = df['raw_ts_obs_body'].to_numpy(dtype=np.int64)
diffs_ms = np.diff(ts) / 1e6
# LowState 실제 갱신율 추정 (Unitree G1 = ~500Hz 가정)
print(f'obs_body dt(ms): median={np.median(diffs_ms):.3f}, p95={np.percentile(diffs_ms, 95):.3f}')
# 기대: ~2ms (500Hz) 근처에 모임. do_fast 300Hz 폴링이지만 같은 LowState 중복은
# K2 의 ts<=last_ts dedup 으로 제거됨.
```

300Hz 폴링이 의미 있는지 (LowState 갱신율과 비교) 확인. 갱신율이 200Hz 이하면
do_fast 주기를 낮추는 게 효율적 (Part 2 P2-7).

### L4-C. IK solve 시간 분포 (Phase L4)

freq_shm 에 `ik_solve_ms_avg`, `_p95`, `_max` 가 publish 됨. 다음 helper 스크립트로
관찰 (한 30s 운용 동안):

```python
import multiprocessing as mp, time
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import WORKER_FREQ
s = SharedMemoryManager(WORKER_FREQ, mp.Lock(), 'freq_shm')
for _ in range(60):
    d = s.read_data()
    print(f'IK ms avg={float(d["ik_solve_ms_avg"]):.2f} '
          f'p95={float(d["ik_solve_ms_p95"]):.2f} '
          f'max={float(d["ik_solve_ms_max"]):.2f} | '
          f'g1_freq={float(d["g1_freq"]):.1f}Hz')
    time.sleep(0.5)
```

**기대**: avg < 10ms, p95 < 20ms (50Hz 예산). max 가 20ms 초과 빈도 큰 경우:
- g1_control/g1_ik.py 의 `max_iter=50` → 30 으로 낮춤
- warm-start 강화 (현재 `self.init_data = sol_q` 적용 중)
- 또는 IK 루프를 별도 프로세스 분리

### L4-D. DEX3 grasp 검증

`--hand dex3 --vr-input controller --thumb-bend 0.5 --thumb-yaw 0.5` 로 실행 후
Right trigger 누름 → 우손 5 손가락이 close 자세 (rad). 다시 누름 → release.
손가락 위치가 자연스러운지 확인. thumb-bend / thumb-yaw 값 조정으로 잡기 좋은
자세 튜닝.

---

## L5. Inspire 손 (선택 — 사용 시)

### L5-A. Modbus touch sensor read length

`workers/worker_hand_dds.py` 가 `inspire_sdk.ModbusDataHandler.read()` 의 응답
dict 의 키들을 `LEFT/RIGHT_TOUCH_SENSOR_LAYOUT` 의 field 이름과 매핑. **실 device
연결 시 처음 read 의 dict.keys() 를 출력해 schema 와 일치하는지 확인**:

```python
# 임시 디버그 라인 — worker_hand_dds 의 while 진입 직전에 추가
data_dict = handler.read()
import json
logger_mp.info(f"[hand_dds:{LR}] read dict keys: {list(data_dict.get('touch', {}).keys())}")
```

field 명 mismatch 면 `LEFT/RIGHT_TOUCH_SENSOR_LAYOUT` 수정 필요.

---

## L6. 데이터 수집 → 학습 → 배포 일관성

### L6-A. 모달리티 토글 일관성 (PART3 §1 + §2 로 일반화됨 — SUPPLEMENT 보강)

> 초판 (이 섹션) 은 "head 단건" 문제로 적었으나 분석자의 SUPPLEMENT 보강과 PART3 개정판이
> 더 일반적인 **waist / head / tactile 공통 토글 일관성** 으로 재정의. 이 항목은 Phase M
> (commit 후) 의 §M1~M4 로 해소된다. 본 섹션은 회귀 검증 매트릭스만 남김.

**Phase M 이전 상태 (검증된 사실)**:
- `modality_dex3.json` (33D), `modality_inspire.json` (31D) 모두 **state.head 포함**.
- `align_and_save_episode` 의 state_vec 는 항상 `[obs_waist, obs_head, obs_arm, obs_hand]`
  concat — **head 항상 포함** ✓ (modality 와 일치).
- `worker_deploy_policy._split_qpos` 는 `qpos[3:5]` (head) **의도적으로 버림** → modality 와
  **차원 불일치** (정책 입력이 학습 데이터와 다른 차원).
- waist_mode / head_mode / tactile_mode 는 TELEOP_CONFIG 에 기록만 되고 저장/배포 레이아웃
  미반영.

**Phase M (PART3 적용) 후**:
- `utils/modality_layout.py` 가 단일 진실 출처. modality 빌드 + state_vec concat + deploy
  분할 모두 동일 함수에서 layout 결정.
- `--waist fixed` → 데이터에서 waist 제외 / `--waist hmd` → 포함.
- `--head off` → 데이터에서 head 제외 / `--head dxl` → 포함.
- `--tactile off` → observation.sensor 제외 / `--tactile on` → 포함.
- deploy 가 record/<task>/meta/modality.json 을 로드해 obs_dict 키/차원 자동 결정.

**회귀 검증 매트릭스** (각 조합 1 에피소드 수집 후):
```python
df = pd.read_parquet('record/<task>/data/chunk-000/episode_000000.parquet')
print(f'state_vec dim: {len(df["observation.state"].iloc[0])}')
import json
m = json.load(open('record/<task>/meta/modality.json'))
max_end = max(v['end'] for v in m['state'].values())
print(f'modality max end: {max_end}')
assert len(df["observation.state"].iloc[0]) == max_end, 'state_vec / modality 불일치'
```

조합:
| waist | head | hand    | tactile | 기대 state dim |
|-------|------|---------|---------|----------------|
| fixed | off  | inspire | off     | 14 (arm14 + hand12 → 26)... 정확한 값은 modality 결과에서 확인 |
| hmd   | dxl  | dex3    | off     | 33 (waist3 + head2 + arm14 + hand14) |
| ...   | ...  | ...     | ...     | ... |

(테이블 값은 modality_layout.py 의 build_state_layout 결과로 정확히 산출.)

### L6-B. GR00T DataConfig 등록

`evaluate.py --data-config-key` 의 기본은 `unitree_g1` (Phase L1 변경). 사용자 학습
때 `gr00t/experiment/data_config.py` 에 신규 DataConfig 등록 필요:

```python
# DEX3 + RealSense 3뷰 예시
class UnitreeG1Dex3Rs3DataConfig(UnitreeG1DataConfig):
    video_keys = ["video.ego", "video.wrist_l", "video.wrist_r"]
    state_keys = ["state.waist", "state.left_arm", "state.right_arm",
                  "state.left_hand", "state.right_hand"]
    action_keys = ["action.waist", "action.left_arm", "action.right_arm",
                   "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

DATA_CONFIG_MAP["unitree_g1_dex3_rs3"] = UnitreeG1Dex3Rs3DataConfig()
```

modality_dex3.json 의 video.{ego,wrist_l,wrist_r} 와 정확히 매칭.

학습 + 배포 시:
```bash
python evaluate.py --mode gr00t_rs_multi \
    --data-config-key unitree_g1_dex3_rs3 \
    --model-path /path/to/checkpoint-XXXXX
```

### L6-C. action_horizon × slow_hz × execution_horizon 튜닝 (Part 2 P1-4)

GR00T 의 action_horizon = 16. slow_hz = 20Hz 면 chunk 가 0.8초 분량. fast_hz =
50Hz 면 upsample 후 40 sample. trim 으로 앞부분 짧아짐.

**관찰 항목** (deploy 중 lag stats 로그):
```
[Deploy] lag stats: avg=XXms max=YYms trim_last=N chunk_remain=M
```

- `trim_last` (last trim_samples) 가 평균 10 미만이면 OK
- `chunk_remain` 이 < 10 으로 자주 떨어지면 cross-fade overlap 부족 → slow_hz 낮추기
  (예: 10Hz, action_horizon 0.8초 분량 동일하지만 새 chunk 도착 간격 ↑) 또는
  execution_horizon 명시화 (H/2 만큼 실행 후 강제 재추론).

### L6-D. hand action overshoot 확인 (Phase L3 효과 검증)

Phase L3 (P1-3) 적용 후 hand 는 linear upsample. spline ringing 사라졌는지 확인:

```python
# eval 중 ROBOT_ACTION SHM 의 action_hand 를 1초간 sampling
import multiprocessing as mp, time, numpy as np
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import ROBOT_ACTION
s = SharedMemoryManager(ROBOT_ACTION, mp.Lock(), 'robot_action_shm')
arr = []
for _ in range(50):
    arr.append(s.read_data()['action_hand'].copy())
    time.sleep(0.02)
hand = np.stack(arr)
print(f'hand min={hand.min():.3f} max={hand.max():.3f}')
# Inspire (0..1): max 1.0 초과면 overshoot
# DEX3 (rad): 각 joint URDF limit 초과면 overshoot
```

---

## L7. 채널 일치 검증 (Part 2 P2-8)

수집 mp4 의 색공간과 deploy 입력의 색공간이 동일 (RGB) 인지 검증.

```python
# 1) 수집 mp4 한 프레임 추출
import imageio
v = imageio.get_reader('record/<task>/videos/chunk-000/observation.images.ego/episode_000000.mp4')
mp4_frame = v.get_data(0)  # RGB

# 2) 같은 시점 SHM 의 frame 을 process_frame 으로 변환
from workers.worker_deploy_policy import process_frame
shm_frame_bgr = ...  # SHM read 결과 (BGR)
deploy_rgb = process_frame(shm_frame_bgr)

# 두 frame 의 평균 RGB 값이 비슷한지 (정확히 같지는 않아도 색이 안 뒤집혔는지)
print('mp4 frame mean BGR/RGB:', mp4_frame.mean(axis=(0,1)))
print('deploy frame mean:', deploy_rgb.mean(axis=(0,1)))
```

빨강↔파랑이 뒤바뀐 듯한 큰 차이면 어딘가 BGR/RGB 변환 누락. 학습 데이터와 deploy
입력의 채널 순서 불일치는 정책 성능 큰 저하 원인.

---

## L8. 촉각 (Inspire 먼저, DEX3 는 N 확정 후 — SUPPLEMENT 보강)

### L8-pre. Inspire FTP tactile 토글 데이터 경로 (PART3 §3)

> 분석자 SUPPLEMENT 보강 채택: **Inspire 의 LEFT/RIGHT_TOUCH_SENSOR_LAYOUT 은 이미 차원
> 확정** → Phase M 토글 메커니즘에 먼저 연동. Inspire 가 토글 일관성 레퍼런스 구현이 됨.

Phase M6 (commit 후) 적용 — `--hand inspire --tactile on` 시 record_collectors 가 touch
buffer 수집 + parquet `observation.sensor` 컬럼에 저장. `--tactile off` 일 때 100% 불변
(zero placeholder 유지).

검증:
```python
df = pd.read_parquet('record/<task>/data/chunk-000/episode_000000.parquet')
# --tactile on 일 때
sensor = df['observation.sensor'].iloc[0]
print(f'sensor dim: {len(sensor)}, all-zero? {all(v==0 for v in sensor)}')
# off → 12-D zero, on → 실제 touch 값
```

### L8-A. DEX3 press_sensor_state sequence length 실측

`--tactile on` 으로 main.py 실행 후 첫 second 안에 다음 로그 1회 출력:
```
[Dex3:l] tactile press_sensor_state: N objects, each pressure[12]. Total tactile values per hand = N*12.
```

**기대**: 외부 자료 (사용자 검색) 는 N=9. 실측이 다르면 N 값을 기록.

### L8-B. SHM schema 확정 + 후속 PR

L8-A 결과로 N 확정 후 후속 작업:

1. `sharedmemory/shm_schema.py` 에 신규 schema:
   ```python
   DEX3_TACTILE = [
       ("left_pressure",     (N*12,), np.float32),
       ("left_temperature",  (N*12,), np.float32),
       ("right_pressure",    (N*12,), np.float32),
       ("right_temperature", (N*12,), np.float32),
       ("tactile_ts",        (),      np.int64),
   ]
   ```
2. main.py SHM_CONFIG 에 `dex3_tactile_shm` 추가 (TACTILE_MAPPING.on 시에만).
3. robot_hand_dex3._subscribe_hand_state 가 collect_tactile=True 시 위 SHM 에 write.
4. record_collectors 가 tactile buffer + parquet `observation.sensor` 컬럼에
   pressure 데이터 저장 (현재 zero placeholder 대체).

**즉시 진행 가능 부분**: 위 1~4 의 코드 자체는 차원 결정 후 30분 작업.

---

## L9. 데이터 정제 (data_refinement) 호환

### L9-A. convert_to_dp / convert_to_act 차원 가정 검증

`data_refinement/convert_to_dp.py`, `convert_to_act.py` 가 hand DOF 를 12 로
가정하고 있다면 (현재 코드 확인 필요) DEX3 14D 데이터 변환 시 마지막 2 컬럼이
사라진다. parquet 의 `observation.state` 길이로 확인:

```python
# DEX3 데이터로 변환 후
import zarr
z = zarr.open('path/to/dp_zarr')
print('state shape:', z['data/state'].shape)  # (T, 33) 이면 OK, (T, 31) 이면 hand truncate 됨
```

---

## L11. Phase N — `--lower-body loco` (motion mode / 보행) 검증

### L11-A. arm_sdk weight ramp 안전 동작

`--lower-body loco` 로 main.py 띄우면 worker_g1_ctrl 가
`G1_29_ArmController.engage_arm_sdk(ramp_sec=2.0)` 호출 → 약 2초 동안 motor_cmd[29].q
가 0→1 ramp. 종료 시 stop() 가 disengage_arm_sdk(2.0) 호출 → 1→0 ramp.

**확인**:
- G1 hoist 가 *내려져* 있고 사용자가 손/허리로 잡고 있는 상태에서 시작 (loco 첫 검증).
- engage 중 (2s) 팔/허리에 급변 없는지 — `motor_cmd[29].q` 가 0.5 부근에서 weak/wrist 만
  partial 제어, 정상 동작이면 *부드러운 전이*.
- main.py 에 SIGINT(Ctrl+C) → disengage ramp 후 종료. 종료 직후 motion controller
  (`elmo` / `g1_locomotion` 등 firmware 측) 가 다시 hold 자세 잡는지.

**비정상 시**:
- engage 중 팔이 급가속 → init_motion_mode_lock 단계에서 `arm_q_target = 현재 q` 가
  제대로 적용 안 됐을 가능성. g1_whole_control.py 의 `_init_motion_mode_lock()` 안에서
  `self.q_target` 초기화 코드 확인.
- engage 후 motion controller 가 갑자기 자세 무너지면 firmware 측 motion mode 가
  arm_sdk 와 충돌. 그 경우 `--gait off` 만 사용하고 `loco.Start()` 호출 안 함.

### L11-B. thumbstick → LocoClient.Move 보행

`--lower-body loco --gait thumbstick --gait-stick split` 권장 (병진 = 왼쪽,
회전 = 오른쪽 X).

**확인 절차 (보수 → 점진)**:
1. 처음에는 `loco` 모드 + `--gait off` 로 motion mode 안정성만 확인 (30s).
2. 다음 `--gait thumbstick` 추가, 사용자는 stick 을 *살짝* 만 (deadzone 0.15 넘어서
   |v|≈0.05 m/s 정도). worker_loco 의 첫 non-zero Move 에서 `loco.Start()` 호출 →
   robot 이 보행 모드 진입. 1초 후 stick 중립 → Move(0,0,0,duration=1s) 가 자연 정지.
3. 점진적으로 stick 폭 늘려 vx=0.15 m/s 까지 (현재 _SCALE_VX). 안정적이면
   workers/worker_loco.py 의 `_SCALE_VX/_VY = 0.15` → 0.3 까지 상향 (xr_teleoperate
   공식 상한).
4. 양쪽 thumbstick click → worker_loco 가 `loco.Damp()` 호출 (soft e-stop, cooldown
   1s).

**비정상 시**:
- robot 이 stick 입력 안 받음 → `loco.SetFsmId(...)` 또는 `loco.Start()` 호출 직전 단계
  실패. logger_mp 의 `[Loco] Start() called` 메시지 유무 확인.
- continous_move=True 가 실수로 사용되면 명령 한번 보낸 후 stick 놓아도 ~10일 (864000s)
  동안 그 속도 유지. worker_loco.py 어디에도 `continous_move=True` 없는지 grep 으로
  재확인.

### L11-C. hoist ↔ loco 데이터 일관성 (deploy 영향)

학습 데이터가 hoist 로 수집되었을 때 loco 모드로 deploy:
- ROBOT_OBS 의 `obs_leg / obs_waist` 는 motion controller 가 보행 중에도 LowState 로
  publish — 즉 deploy 시 실제 leg/waist q 가 정책에 들어감 (hoist 데이터의 *고정*
  leg/waist 와 분포 차이 발생). 학습/배포 모드 정합이 깨지면 정책 출력 quality 저하
  가능.

**권장**:
- 첫 운용 단계에서는 hoist 데이터로만 학습 → hoist 모드로 deploy. loco 는 일단
  텔레오퍼레이션 보행 동작만 검증.
- 향후 loco 보행 데이터 수집 시 modality.json 에 `lower_body` 메타 추가 권장 (학습
  시 모드 분리). 현재 TELEOP_CONFIG.lower_body 는 SHM 에만 있고 parquet meta 에는
  미반영 — Phase N 후 추가 PR 필요 시 `record_collectors.py` 의 episode_meta_json 에
  `lower_body: 'hoist'|'loco'` 필드 1줄 추가.

### L11-D. evaluate.py 측 영향 없음 (확인 사실)

`evaluate.py` 는 worker_g1_ctrl 을 spawn 하지 않고 main.py 가 owner 인 SHM 에
attach 만 한다. 따라서 lower_body 는 *main.py 측 단일 결정*. deploy 도 자동 일관:
- main.py 가 `--lower-body loco` 면 worker_g1_ctrl 의 RUN 분기가 leg/waist 명령을
  버림 → 정책이 action_leg/action_waist 를 추론해도 robot 에 가지 않음.
- main.py 가 `--lower-body hoist` (기본) 면 100% 기존 동작.

evaluate.py CLI 에 별도 lower_body flag 추가 *불필요*.

---

## L10. 잡다 후속 정리

| 항목 | 메모 |
|---|---|
| **VideoSink fps 정합** | 현재 `DEFAULT_OUTPUT_HZ=50Hz` 로 mp4 저장. 카메라 native 30fps + ZOH-resample 결과를 50fps 로 태깅. 학습 도구가 fps 메타로 시간 해석하면 ×1.67 빠르게 인식할 수 있음. modality.json 또는 video meta 에 명시 권장. |
| **freq_shm.camera_freq 멀티-cam 충돌** | 3개 카메라 worker 가 같은 `camera_freq` 필드에 write → 마지막 writer 만 반영. WORKER_FREQ schema 를 `camera_ego_freq / camera_wrist_l_freq / camera_wrist_r_freq` 등으로 확장 또는 worker_camera 가 role 별 freq field 동적 추가. |
| **wrist pose 학습 입력 저장 시 SLERP** | tv_wrapper.py 의 TODO 주석 참조. utils.mat_tool.se3_interp 호출만 추가 (이미 구현). |
| **pro4000 ↔ 로컬 머신 sync** | rsync 또는 git pull (Seung-Sub/G1_Teleop_Datacollection main 기준). pro4000 의 `~/G1_Teleoperation_clean` 의 old history 와 충돌 가능 — git reset --hard origin/main 필요. |
| **KISTAR zip 54MB** | GitHub push 시 warning. `git filter-branch` 로 history 정리 가능. 사용자 결정 필요. |
| **modality 정적 파일 → template (SUPPLEMENT 보강)** | Phase M 동적 빌드 도입 후, 기존 `utils/parquet/modality_{inspire,dex3}.json` 은 (a) 기본 템플릿 reference 로만 (확장자 `.template.json`) 또는 (b) 제거. 동적 빌드 결과가 record/<task>/meta/modality.json 의 정본. 두 파일 공존 시 어느 게 진짜인지 혼란. Phase M2 에서 `.template.json` 으로 개명. |
| **data_refinement convert_to_{dp,act} (§L9 와 연계)** | hand DOF 12 하드코딩 가능성. 동적 modality (가변 state 차원) 도입 후, modality.json 의 max end 를 읽어 state 차원 결정하도록 변경. |

---

## 참고 commit

| 단계 | 커밋 | 관련 후속 (이 문서) |
|---|---|---|
| Phase K1 (P0-1.1) | `7199019` | L1-C (Global Time 확인) |
| Phase K2 (P0-1.2) | `7199019` | L4-B (obs_body_ts dedup) |
| Phase K3 (P0-1.3) | `7199019` | (DEX3/Inspire recv_ts 동작 확인) |
| Phase K4 (P0-1.4) | `7199019` | L6-D (hand action 검증) |
| Phase K5 (P0-2)   | `7199019` | L6-D (action 보간 확인) |
| Phase K6 (P1-8)   | `f9c3190` | L6-A (head 일관성) |
| Phase K7 (P0-3+P1-4) | `5535bc1` | L1-A/B/C (멀티 카메라) |
| Phase K8 (P1-5)   | `4f53e4d` | L8-A/B (촉각 sequence length) |
| Phase K9 (P2-7)   | `08d10e1` | — |
| Phase L1 (Part2 P0-1) | (this) | L6-B (DataConfig 등록) |
| Phase L2 (Part2 P0-2) | (this) | L4-B (obs_ts_policy) |
| Phase L3 (Part2 P1-3) | (this) | L6-D (overshoot 검증) |
| Phase L4 (Part2 P1-5) | (this) | L4-C (IK 예산) |
| Phase L5 (Part2 P1-6) | (this) | — (Rate 단위 테스트 완료) |
| Phase L6 (Part2 P2)   | (this) | (이 문서) |
| Phase N (lower-body)  | (this) | L11-A/B/C/D (motion mode + gait) |

각 commit 의 상세는 `git log --oneline` + 본 워크스페이스의 docs/HARDWARE.md 참고.
