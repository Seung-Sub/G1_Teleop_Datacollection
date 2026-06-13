# Workspace 사용법 (USAGE.md)

본 워크스페이스의 **모든 entry point 명령어 + 인자 + 사용 시점** 정리. 
수집 → 검증 → 변환 → 학습 → 배포 end-to-end 순서.

> 인자/기본값은 실제 코드(`argparse`) 기반. 자세한 컨트롤러 매핑/운용 흐름은
> [`README.md`](README.md), N1.7 학습은 [`GR00T_PIPELINE_GUIDE.md`](GR00T_PIPELINE_GUIDE.md),
> DP 학습은 [`DP_PIPELINE_CHECKLIST.md`](DP_PIPELINE_CHECKLIST.md).

---

## 0. 사전 — conda 환경 3개

| Env | 용도 | 위치 |
|---|---|---|
| `teleop` | 수집/teleop 본체 (SHM owner, main.py, evaluate.py, evaluate_dp.py, 데이터 변환) | 본 워크스페이스 |
| `groot` | GR00T N1.7 학습/추론 | `~/Isaac-GR00T` |
| `umi` | Diffusion Policy 학습/추론 | `~/diffusion_policy` |

설치는 [`docs/INSTALL.md`](docs/INSTALL.md).

---

## 1. Teleop + 데이터 수집 (`main.py`)

```bash
conda activate teleop
python main.py [옵션]
```

### 1-1. CLI 옵션 전체

| 옵션 | 기본값 | choices / type | 의미 |
|---|---|---|---|
| `--hand` | `inspire` | `{inspire, dex3}` | 손 하드웨어 |
| `--camera` | `auto` | `{auto, zed, realsense, none, <serial>}` | 단일 카메라 모드 (cameras-config 미사용 시) |
| `--camera-role` | `ego` | str | cameras-config 없이 운용 시 role 라벨 |
| `--zed-mode` | `direct` | `{direct, stream}` | ZED 연결 방식 |
| `--cameras-config` | `utils/cameras.yaml` | path | 멀티 카메라 yaml. 비어있으면 `--camera` 단일 모드 |
| `--vr-input` | `hand` | `{hand, controller}` | Quest3 입력 모드. controller 권장 |
| `--waist` | `hmd` | `{hmd, fixed}` | waist: HMD 변위 / 고정 |
| `--head` | `dxl` | `{dxl, off}` | Dynamixel head 사용 / 비활성 |
| `--tactile` | `off` | `{off, on}` | 손 촉각 로깅. on=DEX3 press_sensor_state / Inspire 17점 촉각. **Inspire는 off일 때 손 상태 Hz↑**(브리지가 촉각 미read) |
| `--lower-body` | `hoist` | `{hoist, loco}` | hoist=호이스트 현수 rt/lowcmd / loco=motion mode rt/arm_sdk |
| `--gait` | `off` | `{off, thumbstick}` | (loco) 보행 |
| `--gait-stick` | `split` | `{split, left, right}` | (loco+gait) stick 매핑 |
| `--grip-profile` | `full_oppose` | profile name | (Inspire) 상황별 그립 메뉴 선택. `full_oppose\|tripod\|pinch\|lateral\|hook` (→ `hand_control/inspire_grip_profiles.yaml`) |
| `--thumb-bend` | (profile) | float 0..1 | (Inspire) 엄지 굽힘 override |
| `--thumb-yaw` | (profile) | float 0..1 | (Inspire) 엄지 회전(대향 각도) override |
| `--grasp-fingers` | (profile) | comma subset | (Inspire) 파지 시 닫히는 손가락. `pinky,ring,middle,index`(+`thumb` 시 엄지도 굽힘) override |
| `--close-depth` | (profile) | float 0..1 | (Inspire) 파지 깊이 (1.0=완전 폐쇄) override |
| `--grip-force` | (profile) | int 0..1000 | (Inspire) force_set 파지력 상한(g). 도달 시 펌웨어 정지=과부하 차단. **deploy 에서도 적용** |
| `--grip-speed` | (profile) | int 0..1000 | (Inspire) speed_set 속도 (1000=full≈800ms) |
| `--no-robot` | False | flag | G1/hand 워커 생략 (Quest3+IK만 검증) |

> Inspire 그립 프로파일: `hand_control/inspire_grip_profiles.yaml` 를 열어 상황별 세팅(손가락 수·엄지 각도·force/speed)을 보고 `--grip-profile <name>` 로 선택. 값은 파일에서 직접 튜닝/추가 가능. 개별 플래그를 주면 그 항목만 override. 엄지 회전(yaw) 0..1 의 실제 방향(정면 대향 vs 측면)은 실하드웨어에서 1회 확인 후 필요 시 값 반전.

### 1-2. 자주 쓰는 조합

```bash
# 표준 운용 — DEX3 양손 + RealSense 3대 + 호이스트 현수 + HMD 목 걸기 + controller
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist

# Inspire 양손 — 범용 파워 그립 (5지 전부 + 엄지 정면 대향)
python main.py --hand inspire --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist --grip-profile full_oppose

# Inspire — 엄지+검지+중지 3점(tripod) 프로파일
python main.py --hand inspire --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist --grip-profile tripod

# Inspire — 프로파일 + 일부 override (tripod 인데 엄지 회전만 더 옆으로)
python main.py --hand inspire --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist \
               --grip-profile tripod --thumb-yaw 0.6

# (deploy) 수집 때 쓴 프로파일로 실행 → force/speed 안전 envelope 동일
#   python evaluate_dp.py ...  와 함께 main.py 를 같은 --grip-profile 로 띄움

# loco 모드 (모션 모드 진입 후 — 리모컨 L2+B → L2+UP → R1+X)
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body loco --gait thumbstick

# 로봇 없이 Quest3 + IK 만 검증 (USB로 Quest3 만 연결)
python main.py --no-robot --vr-input controller --camera none \
               --hand dex3 --waist fixed --head off

# ZED stereo + waist HMD 추종 + DXL head
python main.py --hand dex3 --camera zed --vr-input controller \
               --waist hmd --head dxl --lower-body hoist
```

### 1-3. GUI 흐름 (실행 후)

1. **G1** 버튼 → 리모컨 `start → A` (zero_torque → default_pos)
2. **Hand** → DEX3/Inspire init
3. **VR** → `adb reverse tcp:8012` 자동 실행
4. Quest3 헤드셋 안 브라우저 `https://127.0.0.1:8012` → "Enter VR" → 목에 걸기
5. GUI 우측 `task_name` 입력 + `num_episodes` + `episode_len` → **SET**
6. **START** → worker FSM RUN. 컨트롤러 Grip 누르고 teleop 시작
7. 데이터 수집: **Left X** = record start/early-stop+save / **Left Y** = drop / **Right B** = SET
8. 자세 복귀: **Right A** = ready-pose cosine 복귀
9. 저장: `record/<task_name>/data/chunk-XXX/episode_XXXXXX.parquet` + `videos/...`

자세한 매핑은 [`README.md`](README.md) §7.

---

## 2. 수집 직후 검증

### 2-1. 에피소드 종합 검증 (저장 즉시)
```bash
conda activate teleop

# 최신 task / 최신 ep 자동 탐색
python verify_episode.py

# 명시 지정
python verify_episode.py --task <task_name> --ep 0
python verify_episode.py record/<task>/data/chunk-000/episode_000000.parquet

# 옵션:
#   --target-hz 60   (저장 정렬 기대 hz)
#   --base record    (record root)
```
→ 60Hz 정렬 / raw_ts_* / 영상-상태 정합 / NaN 체크.

### 2-2. 시계열 궤적 정밀 진단
```bash
python verify_trajectory.py [parquet]
# 옵션: --base record / --target-hz 60
```
→ action 점프 / 추종오차 / 관절범위 + PNG 자동 저장.

### 2-3. 라이브 SHM 진단 (main.py RUN 중 동시 실행)
```bash
# 다른 터미널 (main.py 가 SHM owner 로 살아있어야 함)
conda activate teleop
python check_pipeline_live.py [duration_sec]   # 기본 10초
```
→ 카메라 3대 별 실 fps, hand_freq, vr_freq, action freq, 신선도/지터.

### 2-4. DEX3 단독 DDS 진단 (main.py 끄고)
```bash
python check_dex3_state.py     # rt/dex3/.../state 4개 토픽 비교
python check_dex3_recv.py      # state 수신 + Hz + 관절값
python check_dex3_grasp.py     # main.py 켠 상태에서 grasp 거동 (q/tau/temp/mode)
```

---

## 3. 데이터 변환

### 3-1. GR00T N1.7 형식 (60→20fps, 단일 task)
```bash
conda activate teleop
python data_refinement/convert_to_gr00t.py \
    --src record/<task_name> \
    --out record_gr00t/<task_name> \
    --task "Pick the apple and place it on the plate." \
    --src-fps 60 --tgt-fps 20

# 검증
python data_refinement/verify_gr00t_dataset.py --dataset record_gr00t/<task_name>
```
옵션:
- `--robot-type Unitree_G1` (기본)
- `--codebase-version v2.1` (기본 — LeRobot 호환)
- `--slim-cols` (수집용 부가 컬럼 제거 경량화)

### 3-2. 다중 task 묶음 GR00T
```bash
python data_refinement/convert_to_gr00t_multitask.py \
    --task-spec "record/pick_apple:Pick the apple and place it on the plate" \
    --task-spec "record/pour_water:Pour water from the bottle into the cup" \
    --out record_gr00t/multitask \
    --src-fps 60 --tgt-fps 20

# 또는 JSON spec 파일
python data_refinement/convert_to_gr00t_multitask.py \
    --spec multitask.json --out record_gr00t/multitask
# multitask.json: [{"src":"record/pick_apple","task":"Pick the apple..."}, ...]
```

### 3-3. Diffusion Policy zarr (60→10fps)
```bash
conda activate teleop   # (또는 umi)
python data_refinement/convert_to_dp.py \
    --src record/<task_name> \
    --out record/<task_name>_dp.zarr \
    --src-fps 60 --tgt-fps 10

# 검증
python data_refinement/verify_dp_dataset.py --zarr record/<task_name>_dp.zarr --tgt-fps 10
```
옵션:
- `--views ego wrist_l wrist_r` (기본)
- `--include-sensor` (observation.sensor 컬럼도 zarr 에 포함)

### 3-4. ACT HDF5 (legacy)
```bash
python data_refinement/convert_to_act.py \
    --src record/<task_name> \
    --out act_data/<task_name> \
    --fps 20 [--sim]
```

### 3-5. 데이터셋 유틸 (인자 X — 하드코딩 스크립트 안의 경로 수정 후 실행)
```bash
python data_refinement/merge_parquet_data.py    # 다중 task / chunk 병합 + img_state_delta
python data_refinement/sequential_merge.py      # 에피소드 0부터 재번호 병합
python data_refinement/inspect_parquet.py <parquet>   # schema / 한 행 출력
python data_refinement/plot_parquet.py <parquet>      # action 컬럼 시계열 PNG
python data_refinement/apply_mask_to_videos.py        # ZED workspace mask post-apply
```

---

## 4. 학습 (외부 env)

### 4-1. GR00T N1.7 — 자세히 [`GR00T_PIPELINE_GUIDE.md`](GR00T_PIPELINE_GUIDE.md)
```bash
# stats 사전 생성 (검증용 — 학습 시 launch_finetune 가 자동 생성도 함)
conda activate groot
cd ~/Isaac-GR00T
python -m gr00t.data.stats \
    --dataset-path $HOME/G1_Teleop_Datacollection/record_gr00t/<task> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1_DEX3/g1_dex3_config.py

# 학습
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path $HOME/G1_Teleop_Datacollection/record_gr00t/<task> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1_DEX3/g1_dex3_config.py \
    --num-gpus 1 --output-dir ./outputs/g1_<task> \
    --max-steps 10000 --global-batch-size 32 \
    --dataloader-num-workers 4 --episode-sampling-rate 1.0 \
    --allow-padding
```
- 권장 episode 길이 ≥ 10초 (200 frame @20fps). 자세한 sample 가능량 표는
  `GR00T_PIPELINE_GUIDE.md §1-4`.

### 4-2. Diffusion Policy — 자세히 [`DP_PIPELINE_CHECKLIST.md`](DP_PIPELINE_CHECKLIST.md)
```bash
conda activate umi
cd ~/diffusion_policy
python train.py --config-name=train_diffusion_unet_g1_dex3_workspace \
    task.dataset_path=$HOME/G1_Teleop_Datacollection/record/<task>_dp.zarr \
    logging.mode=offline    # wandb 안 쓸 때
```
산출물: `data/outputs/<날짜>/<시각>_train_diffusion_unet_image_g1_dex3_image/checkpoints/{latest,epoch=N}.ckpt`.

---

## 5. Deploy (정책 추론)

> 공통: **Terminal 1** = `main.py` (teleop env, SHM owner), **Terminal 2** = 정책
> 추론 (외부 env, SHM attach). 추론 시작은 GUI 의 **Deploy=True** 토글로.

### 5-1. GR00T N1.7 (`evaluate.py`)
```bash
# Terminal 1
conda activate teleop
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist

# Terminal 2
conda activate groot
python evaluate.py \
    --mode gr00t_rs_multi \
    --model-path /path/to/checkpoint-XXXXX \
    --embodiment-tag new_embodiment \
    --device cuda \
    --action-method tem --decay 0.3 --window-size 5 \
    --slow-hz 20 --fast-hz 60 \
    --lag-compensate --lag-log-every 50
```
주요 옵션:
- `--mode` ∈ `{gr00t_rs_multi, gr00t_zed, gr00t}` — 카메라 모달리티 분기
- `--action-method` ∈ `{base, maf, tem}` — chunk 평활화. tem 권장
- `--obs-ts-policy` ∈ `{min, max}` — lag 계산에 쓸 obs ts 선택
- `--no-lag-compensate` — lag trim 끔 (비교용)
- `--modality-json <path>` — modality 명시 override (보통 자동)
- `--no-binocular` / `--masking` — ZED 모드 옵션
- `--data-config-key`, `--denoising-steps` — N1.7 미사용 (하위호환 stub)

내부: `_CameraFrameRing` 가 60fps 카메라 SHM 을 polling → 추론 시 `[t-1초, t]` frame
두 장 stack (학습 `video.delta_indices=[-20, 0]` 정합). warmup(<1초) 시 현재 frame 복제.

### 5-2. Diffusion Policy (`evaluate_dp.py`)
```bash
# Terminal 1 (동일)

# Terminal 2
conda activate umi
python evaluate_dp.py \
    --mode gr00t_rs_multi \
    --model-path /home/kist/diffusion_policy/data/outputs/.../checkpoints/latest.ckpt \
    --action-method tem --decay 0.3 --window-size 5 \
    --slow-hz 10 --fast-hz 60 \
    --device cuda \
    --lag-compensate --lag-log-every 50
```
GR00T 와 동일한 옵션 + DP 특화:
- `--slow-hz 10` 기본 (DP 학습 60→10 다운샘플)
- `--fast-hz 60` (arm 제어 60Hz)
- 학습 측 단일 task → `embodiment-tag` / `data-config-key` 인자 없음
- 내부: `n_obs_steps=2` deque 누적 obs

### 5-3. GUI Deploy 흐름
1. main.py 띄우고 G1 → Hand → VR → SET → START 까지 (위 §1-3)
2. Terminal 2 정책 띄우기 (위)
3. GUI 우측 **Deploy** 버튼 클릭 → `record_mode.deploy=True` + `set_start` 자동 set
4. 정책이 ROBOT_OBS 읽어 추론 → ROBOT_ACTION publish → 로봇 동작
5. **Right A** 로 ready-pose 복귀 가능 (deploy 도중 안전 복귀 안전망)

---

## 6. 사전 점검 / 디버깅

### 6-1. 오프라인 검증 (하드웨어 0개)
```bash
conda activate teleop
QT_QPA_PLATFORM=offscreen python scripts/verify_offline.py
# → SUMMARY  PASS=9  FAIL=0
```
검증: imports, SHM schema, IK build, mat_tool, align utils, record_collectors,
camera_discovery, worker imports, main.py CLI 노출.

### 6-2. Quest3 검증 (로봇 없이)
```bash
# Terminal 1
python main.py --no-robot --vr-input controller --camera none \
               --hand dex3 --waist fixed --head off

# Terminal 2 (GUI 의 VR → START 까지 누른 뒤)
python scripts/verify_quest3.py --rate 2.0 --watch
# 옵션: --full (4x4 행렬 모두 출력)
```
ctrl_connected / HMD/L/R trans / trig·grip·btn / arm action / freq / TS 실측.

### 6-3. vuer client JS 패치 (설치 후 1회 + 재설치 후 매번)
```bash
python scripts/patch_vuer_xr.py status     # PATCHED 인지 확인
python scripts/patch_vuer_xr.py disable    # hand-tracking OFF + WS port 누락 fix
python scripts/patch_vuer_xr.py restore    # 원상 복구
```
미적용 시 Quest3 가 controller demote → main.py `--vr-input controller` 작동 X.

---

## 7. 시나리오별 명령 모음

### 시나리오 A — 처음 데이터 수집 시작
```bash
# (사전 — INSTALL.md 1회만)
conda activate teleop
python scripts/patch_vuer_xr.py status                  # PATCHED 확인
QT_QPA_PLATFORM=offscreen python scripts/verify_offline.py

# 매번
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist
# → GUI: G1→Hand→VR→SET(task_name)→START → 컨트롤러 teleop + Left X 로 ep start
```

### 시나리오 B — 수집 1 에피소드 직후 품질 확인
```bash
# 다른 터미널 (main 그대로 두고)
python verify_episode.py                                # 최신 ep 자동 탐색
python verify_trajectory.py                             # PNG 저장
```

### 시나리오 C — GR00T 학습 준비 (수집 충분히 끝난 뒤)
```bash
# 변환
python data_refinement/convert_to_gr00t.py \
    --src record/<task> --out record_gr00t/<task> \
    --task "<자연어 instruction>" --src-fps 60 --tgt-fps 20
python data_refinement/verify_gr00t_dataset.py --dataset record_gr00t/<task>

# 외부 서버로 record_gr00t/<task> + Isaac-GR00T/ 이동 후 §4-1 학습
```

### 시나리오 D — DP 학습 준비
```bash
python data_refinement/convert_to_dp.py \
    --src record/<task> --out record/<task>_dp.zarr --tgt-fps 10
python data_refinement/verify_dp_dataset.py --zarr record/<task>_dp.zarr --tgt-fps 10

# §4-2 학습 (umi env)
```

### 시나리오 E — 학습된 정책 실로봇 배포
```bash
# Terminal 1 — 정상 수집 명령 그대로 (필수 옵션 동일)
python main.py --hand dex3 --camera realsense --vr-input controller \
               --waist fixed --head off --lower-body hoist

# Terminal 2 — GR00T
conda activate groot
python evaluate.py --mode gr00t_rs_multi --model-path <ckpt> \
                   --device cuda --lag-compensate

# 또는 DP
conda activate umi
python evaluate_dp.py --mode gr00t_rs_multi --model-path <ckpt.ckpt> \
                      --slow-hz 10 --fast-hz 60

# GUI: Deploy 토글 → 정책 추론 시작.
# 비상 시 Right A 로 ready-pose 복귀.
```

### 시나리오 F — 라이브에서 카메라/SHM 이상 의심
```bash
# main.py 그대로 두고 다른 터미널
python check_pipeline_live.py 20     # 20초 측정 — 각 카메라 fps / hand_freq / vr_freq
```

### 시나리오 G — DEX3 가 init 에서 멈춤
```bash
# main.py 종료 후
python check_dex3_state.py           # state 수신 토픽 비교
python check_dex3_recv.py            # 단독 진단
# 통신 OK 면 main.py 내부 순서/도메인 문제 — 케이블·전원·펌웨어 점검은 README §11
```

---

## 8. 데이터 저장 경로 / 산출물 구조

수집:
```
record/<task_name>/
├── data/chunk-XXX/episode_XXXXXX.parquet   # state/action/ts + raw_ts_* 컬럼
├── videos/chunk-XXX/observation.images.ego/episode_XXXXXX.mp4
├── videos/chunk-XXX/observation.images.wrist_l/...
├── videos/chunk-XXX/observation.images.wrist_r/...
└── meta/modality.json                       # GR00T 형식 자동 생성
```

GR00T 변환:
```
record_gr00t/<task_name>/
├── data/chunk-XXX/episode_XXXXXX.parquet    # 20fps 다운샘플
├── videos/chunk-XXX/observation.images.*/episode_XXXXXX.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, stats.json, relative_stats.json}
```

DP 변환:
```
record/<task_name>_dp.zarr/   # zarr 디렉토리
└── data/{camera_0..N, state, action, sensor*}, meta/{episode_ends}
```

> 데이터 자체는 git 에 안 올라감 (`.gitignore` 가 `record/**/data,videos,meta/`,
> `record/**/*.zarr/`, `record_gr00t/` 모두 제외).
