# G1_Teleop_Datacollection → GR00T N1.7 학습 파이프라인 (전체 절차)

> 본 문서는 실제 GR00T N1.7 코드(lerobot_episode_loader / sharded_single_step_dataset /
> factory / stats / launch_finetune / embodiment_configs) + HuggingFace base 모델
> (`nvidia/GR00T-N1.7-3B`) 의 processor_config.json 직접 정독 + G1_Teleop_Datacollection
> 수집 코드(worker_record / parquet_sink / build_dataset_meta) 를 모두 확인해 확정한 절차입니다.
>
> **현재 상태 (Phase A~D 적용 후)**:
> - `examples/G1_DEX3/g1_dex3_config.py`: `video.delta_indices=[-20, 0]` (base 학습 일치),
>   `ACTION_HORIZON=40` (base capacity 일치).
> - `gr00t/configs/finetune_config.py` + `examples/finetune.sh`: `--allow-padding` 옵션 노출
>   (음수 video delta 안전 처리 — Phase A).
> - `workers/worker_deploy_policy.py`: video 2-frame ring buffer + 60Hz polling thread 추가
>   (학습-배포 정합 — Phase B).
> - 학습 가이드: `~/Isaac-GR00T/G1_TRAINING_GUIDE.md` (Phase D, 외부 서버용).
> - pick_test / multitask_test 실데이터로 데이터 파이프라인 end-to-end 검증 완료.

---

## 0. 파일 배치 (최초 1회)

데이터 수집/변환 워크스페이스 (`~/G1_Teleop_Datacollection`):
```
data_refinement/convert_to_gr00t.py       # GR00T 변환 (60→20fps)
data_refinement/convert_to_dp.py          # DP 변환 (60→10fps) — DP 쓸 때만
data_refinement/verify_gr00t_dataset.py   # 변환 결과 검증
data_refinement/verify_dp_dataset.py      # DP 검증 — DP 쓸 때만
```

GR00T 워크스페이스 (`~/Isaac-GR00T`):
```
examples/G1_DEX3/g1_dex3_config.py        # NEW_EMBODIMENT modality config
verify_gr00t_loading.py                   # 데이터 파이프라인 검증 (모델 없이)
```

---

## 1. 데이터 수집 (G1_Teleop_Datacollection)

### 1-1. 수집 실행
```bash
cd ~/G1_Teleop_Datacollection
conda activate teleop
python main.py --hand dex3 --camera realsense --vr-input controller \
    --waist fixed --head off --lower-body hoist
```
→ 28D state/action (left_arm 7 + right_arm 7 + left_hand 7 + right_hand 7),
   3 카메라(ego/wrist_l/wrist_r) 640x360, 60Hz, LeRobot v2 형식으로 저장.

### 1-2. task_name (instruction) 관련 — 중요
- 수집 GUI 에서 입력하는 **task_name 은 "폴더 식별자"** 입니다 (`record/<task_name>/`).
  parquet 에는 자연어 instruction 이 저장되지 않고, tasks.jsonl 도 이 시점엔 생성 안 됨.
- **자연어 instruction(학습용 task description)은 다음 단계(변환)에서 `--task` 로 지정**합니다.
- 즉 task_name 은 짧게(예: `pick_apple`) 두고, 실제 instruction 은 변환 시 자연어로 넣으면 됩니다.
  (task_name 을 자연어로 길게 입력해도 동작하지만 폴더명이 길어짐.)

### 1-3. 수집 결과 구조
```
record/<task_name>/
  data/chunk-000/episode_000000.parquet ...   # 28D state/action, timestamp, 60Hz
  videos/chunk-000/observation.images.ego/episode_000000.mp4 ...
  videos/chunk-000/observation.images.wrist_l/...
  videos/chunk-000/observation.images.wrist_r/...
  meta/modality.json                          # 수집 시 자동 생성 (GR00T 호환 형식)
```
- modality.json 은 수집 시 utils/modality_layout.py 가 자동 생성 (state/action 28D 분할,
  video original_key, annotation original_key=task_index). GR00T 형식과 일치 — 수정 불필요.

### 1-4. 학습용 데이터 수집 가이드 (권장, Phase A 변경 반영)

#### 최소 episode 길이 (★ 중요 — 새 config 기반)

학습 시 `g1_dex3_config.py` 가 `video.delta_indices=[-20, 0]` (1초 history) + `ACTION_HORIZON=40`
(2초 lookahead) 을 요구하므로, 다음 두 제약을 동시 만족해야 sample 학습 가치가 있음:
- effective_length = `original − ACTION_HORIZON + 1 = original − 39`
- 정상 video pair 가 잡히는 sample = `effective_length − |min video delta| = effective_length − 20`

원본 frame (= 변환 후 20fps parquet 행 수) 별 학습 가용 sample:

| 원본 length (frame) | duration | effective_length | 정상 video pair sample | clamp 비율 | 추천도 |
|---|---|---|---|---|---|
| 40  | 2.0s  | 1   | 0   | 100% | ❌ 학습 불가 |
| 60  | 3.0s  | 21  | 1   | 95.2% | ⚠️ 매우 부족 |
| 100 | 5.0s  | 61  | 41  | 32.8% | △ 단기 task |
| 200 | 10.0s | 161 | 141 | 12.4% | ✓ 권장 |
| 400 | 20.0s | 361 | 341 | 5.5%  | ✓✓ 권장 |
| 600 | 30.0s | 561 | 541 | 3.6%  | ✓✓ 권장 |

→ **에피소드당 최소 10초 (200 frame @20fps 변환) 이상 권장**.
→ 5초 이하의 짧은 task 는 학습 sample 부족. (allow_padding 으로 학습은 가능하나 처음 1초 video clamp 비율 ↑)

#### 에피소드 수 권장 (NVIDIA 공식 + 우리 계산)

| 에피소드 수 | 평균 episode | 전체 effective sample | 추천 task 난이도 |
|---|---|---|---|
| 10 ep  | 10초 (200 frame) | ~1,600   | smoke test 만 |
| 50 ep  | 10초 (200 frame) | ~8,000   | NVIDIA 가이드 최소 |
| 100 ep | 10초 (200 frame) | ~16,000  | 단일 task 권장 |
| 50 ep × N task | 10초 | ~8,000×N | 다중 task |

#### 데이터 다양성

- **DEX3 controller-mode 의 hand action**: thumb_0/thumb_1 dim 은 CLI 상수, thumb_2/index/middle 는 이산 toggle (open=0 / closed=±한계). state-action corr 0.73~0.82 검증됨 → 학습 가능. 단 **grasp/release timing 다양성** 이 중요 (한 task 당 grasp 1회만 있으면 모델이 timing 학습 불가).
- **arm 자세 다양성**: 같은 target 위치라도 접근 경로/속도 다양화 권장.
- **state-action 정합성**: 우리 데이터는 controller toggle → motor 추적 → state 의 자연스러운 인과관계. 학습에 적합.

---

## 2. GR00T 학습용 변환 (60→20fps)

```bash
cd ~/G1_Teleop_Datacollection
conda activate teleop          # pandas/pyarrow/numpy + ffmpeg 필요

python data_refinement/convert_to_gr00t.py \
    --src record/<task_name> \
    --out record_gr00t/<task_name> \
    --task "Pick the apple and place it on the plate." \   # ← 자연어 instruction
    --src-fps 60 --tgt-fps 20
```
- 60→20fps 다운샘플(3:1, 값 보존), 비디오 동기 다운샘플(ffmpeg).
- info.json/episodes.jsonl/tasks.jsonl/modality.json 생성. state/action float64 유지(GR00T G1 일치).
- `--slim-cols` 옵션: 수집용 부가 컬럼 제거 경량화(선택). 기본은 원본 컬럼 유지.

### 2-1. 변환 결과 검증
```bash
python data_refinement/verify_gr00t_dataset.py --dataset record_gr00t/<task_name>
```
→ meta 형식 / 차원 / 비디오-상태 정합 확인. (stats.json 은 아직 없음 — 다음 단계에서 생성, 경고 정상)

---

## 3. GR00T 데이터 검증 (학습 전, GR00T 환경)

> stats 생성 + 데이터 로딩 검증. **모델을 안 올리므로 저사양 GPU(로컬 eval 머신)에서도 가능.**

### 3-1. stats.json / relative_stats.json 생성
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate groot
cd ~/Isaac-GR00T

python -m gr00t.data.stats \
    --dataset-path $HOME/G1_Teleop_Datacollection/record_gr00t/<task_name> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1_DEX3/g1_dex3_config.py
```
→ meta/stats.json (28D mean/std/min/max/q01/q99),
  meta/relative_stats.json (left_arm/right_arm 의 16×7 horizon별 델타 — RELATIVE arm 만).
- 참고: 이 stats 는 학습(launch_finetune) 시작 시 factory.py 가 자동 생성도 하므로,
  사전 생성은 "검증 목적". 자동 생성에 맡겨도 됨.

### 3-2. 데이터 파이프라인 검증 (모델 없이)
```bash
python verify_gr00t_loading.py \
    --dataset-path $HOME/G1_Teleop_Datacollection/record_gr00t/<task_name> \
    --modality-config-path examples/G1_DEX3/g1_dex3_config.py \
    --num-samples 3 \
    --allow-padding             # ★ Phase A: video.delta_indices=[-20, 0] 시 필수
```
→ config 등록 / 데이터셋 인스턴스화 / state(1×7×4) / action(**40**×7×4) /
  비디오 디코딩(**2프레임** 360×640×3 uint8, [과거, 현재] 순서) / 영상-상태 정합(GR00T assert) / text 확인.
- 비디오 디코딩 에러 시: `--video-backend torchvision_av` 로 재시도.
- `--allow-padding` 미설정 시: pandas iloc 의 negative wrap-around 로 step_index<20 의 sample 이 잘못된 frame pair 로 로드됨 (silent data corruption). **반드시 함께 사용**.

---

## 4. 학습 머신(서버)로 이전 + 학습

> ★ **상세 절차 별도 문서**: `~/Isaac-GR00T/G1_TRAINING_GUIDE.md` (Phase D, 외부 서버 설치/학습/검증 전체 절차). 본 섹션은 요약만.

### 4-1. 이전 항목 + 명령
- Isaac-GR00T 전체 코드 (`examples/G1_DEX3/g1_dex3_config.py` 포함, Phase A~D 변경 모두 포함)
- 변환 데이터 `record_gr00t/<task_name>` (parquet + 비디오 + meta)
- 환경은 외부 서버에서 재구축 (Ubuntu 22.04 + CUDA 12.8+ + dgpu install_deps.sh 권장)
- base 모델은 HF auto-download (anonymous OK, public repo)

### 4-2. 서버 환경 세팅 (공식 dgpu install — 권장)
```bash
# 외부 서버에서
cd ~/Isaac-GR00T
bash scripts/deployment/dgpu/install_deps.sh   # ffmpeg/libaio/cuda-toolkit + uv + 패키지 자동
source .venv/bin/activate
```

### 4-3. 학습 (공식 finetune.sh wrapper — 권장 표준 경로)
```bash
cd ~/Isaac-GR00T
source .venv/bin/activate

# 적은 데이터 (≤100 ep) 예시
EPISODE_SAMPLING_RATE=1.0 \
NUM_GPUS=1 \
MAX_STEPS=10000 \
bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ~/G1_Teleop_Datacollection/record_gr00t/<task_name> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1_DEX3/g1_dex3_config.py \
    --output-dir ~/outputs/g1_<task_name>_$(date +%Y%m%d_%H%M) \
    --allow-padding     # ★ Phase A: 음수 video delta 안전 처리 (필수)
```
- `--allow-padding` 필수 (또는 `ALLOW_PADDING=1` env). 미설정 시 silent data corruption.
- 단일 GPU 권장: A6000/A100/H100 48GB+. 16~24GB 는 `GLOBAL_BATCH_SIZE` ↓ + `gradient_accumulation_steps` ↑ 필요.
- 멀티 GPU: `NUM_GPUS=N` 환경변수 — finetune.sh 가 자동으로 torchrun 사용.
- 자세한 파라미터/트러블슈팅: `~/Isaac-GR00T/G1_TRAINING_GUIDE.md` §5-6 참조.

---

## 5. (선택) Diffusion Policy 변환 — DP 도 쓸 경우
```bash
cd ~/G1_Teleop_Datacollection
conda activate teleop
python data_refinement/convert_to_dp.py \
    --src record/<task_name> --out record/<task_name>_dp.zarr \
    --src-fps 60 --tgt-fps 10
python data_refinement/verify_dp_dataset.py --zarr record/<task_name>_dp.zarr --tgt-fps 10
```

---

## 6. 배포 (학습 후, 실로봇)

### 6-1. 배포 코드 변경 사항 (Phase B 완료)
- `workers/worker_deploy_policy.py`:
  - `_CameraFrameRing` 클래스 추가 (per-camera, 120 슬롯, ts 기반).
  - 별도 60Hz `_camera_poll_loop` thread 가 카메라 SHM 을 polling 해 ring 채움.
  - `get_real_obs` 가 ring 에서 현재 frame + 1초 전 frame 두 개를 ts 기반 pick.
  - Warmup (시작 후 1초 동안 ring 미충전 시): 현재 frame 복제 → 학습 시 step_index<20 의 allow_padding clamp 와 정확히 일치 거동.
  - `build_obs_dict` 가 video=(B=1, T=2, H, W, C) uint8 로 stack ([과거, 현재] 순서).
- `evaluate.py`: 변경 없음 (worker_deploy_policy 가 자동 처리).

### 6-2. 배포 명령
```bash
# Terminal 1: main.py (teleop env, SHM owner)
cd ~/G1_Teleop_Datacollection
conda activate teleop
python main.py --hand dex3 --camera realsense --vr-input controller \
    --waist fixed --head off --lower-body hoist

# Terminal 2: GR00T 정책 추론 (groot env, SHM attach)
conda activate groot
python evaluate.py \
    --mode gr00t_rs_multi \
    --model-path ~/checkpoints/g1_<task_name>/ \
    --embodiment-tag new_embodiment \
    --device cuda \
    --modality-json ~/G1_Teleop_Datacollection/record_gr00t/<task_name>/meta/modality.json
```

→ UI 에서 Deploy=True 버튼 누르면 정책 추론 시작.

---

## 검증 완료 상태 (pick_test / multitask_test 실데이터 기준, Phase A~D 적용 후)
- [x] 데이터 수집 (G1+DEX3+RealSense 3대, 60Hz, 변경 없음)
- [x] 변환 (60→20fps, 메타 4종)
- [x] embodiment config 등록 (NEW_EMBODIMENT, video=[-20,0], action horizon=40, arm RELATIVE / hand ABSOLUTE)
- [x] stats.json (28D) + relative_stats.json (16×7 — 음수 video delta 와 무관, arm 만)
- [x] 데이터 로딩 (state(1,7)×4 / action(40,7)×4 / video(2,360,640,3) / text) — `--allow-padding` 통과
- [x] `--allow-padding` CLI 노출 (FinetuneConfig + launch_finetune + finetune.sh)
- [x] Deploy ring buffer + 60Hz polling thread + warmup (Phase B)
- [x] action_horizon=40 deploy upsample/cross-fade/lag 호환성 (Phase C, 코드 변경 0)
- [x] 학습 가이드 (`G1_TRAINING_GUIDE.md`)
- [ ] **실제 학습 (외부 서버에서, 권장 ≥50 ep × 평균 ≥10s 수집 후)**
- [ ] **실로봇 정책 배포 검증 (학습 체크포인트로 evaluate.py 실행)**
- [ ] DP 변환 실데이터 검증 (DP 쓸 경우)
