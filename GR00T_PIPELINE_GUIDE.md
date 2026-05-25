# G1_Teleop_Datacollection → GR00T N1.7 학습 파이프라인 (전체 절차)

> 본 문서는 실제 GR00T N1.7 코드(lerobot_episode_loader / sharded_single_step_dataset /
> factory / stats / launch_finetune / embodiment_configs)와 G1_Teleop_Datacollection
> 수집 코드(worker_record / parquet_sink / build_dataset_meta)를 정독해 확정한 절차입니다.
> pick_test 1개 에피소드로 변환→stats→로딩까지 실데이터 end-to-end 검증 완료.

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

### 1-4. 학습용 데이터 수집 가이드 (권장)
- **에피소드 수**: 검증은 1개로 충분하나, 실제 학습은 최소 50개 이상 권장 (GR00T 가이드).
- **hand 동작 다양성**: pick_test 분석 시 엄지 일부 관절이 상수(미사용)였음. 다양한 grip 포함 권장.
  (상수 dim 은 GR00T 가 1e-8 클램프로 안전 처리하므로 학습은 가능하나, 다양성이 일반화에 유리.)

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
    --num-samples 3
```
→ config 등록 / 데이터셋 인스턴스화 / state(1×7×4) / action(16×7×4) /
  비디오 디코딩(360×640×3 uint8) / 영상-상태 정합(GR00T assert) / text 확인.
- 비디오 디코딩 에러 시: `--video-backend torchvision_av` 로 재시도.

---

## 4. 학습 머신(서버)로 이전 + 학습

### 4-1. 이전 항목
| 항목 | 방법 |
|---|---|
| GR00T 코드 | `~/Isaac-GR00T` 전체 복사 (examples/G1_DEX3/g1_dex3_config.py 포함) |
| 변환 데이터 | `record_gr00t/<task_name>` 복사 (parquet+비디오+meta, 비디오 용량 주의) |
| conda 환경 | **재구축** (복사 X) — 서버 GPU/CUDA 에 맞게 |
| 모델 가중치 | nvidia/GR00T-N1.7-3B (~6GB) — 학습 시 HF 자동 다운로드 or 미리 받기 |

### 4-2. 서버 환경 재구축
```bash
# 서버에서
cd ~/Isaac-GR00T
conda create -n groot python=3.10
conda activate groot
pip install -e .[base]
pip install --no-build-isolation flash-attn   # GPU 아키텍처별 컴파일
# torch 는 서버 CUDA 버전에 맞는 버전 설치 (예: cu121/cu124 등)
```

### 4-3. 학습 (A6000/RTX6000 ×10 — VRAM 48GB 충분)
```bash
cd ~/Isaac-GR00T
conda activate groot
# 단일 GPU 예시 (멀티는 --num-gpus 조정)
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path $HOME/record_gr00t/<task_name> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1_DEX3/g1_dex3_config.py \
    --num-gpus 1 \
    --output-dir ./outputs/g1_<task_name> \
    --max-steps 10000 --global-batch-size 32 \
    --dataloader-num-workers 4 \
    --episode-sampling-rate 1.0     # 에피소드 적을 때. 많으면 기본 0.1 검토
```
- A6000(48GB)/RTX6000 Ada(48GB): 3B full finetune(40GB 권장) 단일 GPU 가능.
  10 GPU 면 --num-gpus 로 분산학습.
- `--episode-sampling-rate`: 기본 0.1(에피소드 10%만 샘플). 에피소드 적으면 1.0.
- 멀티 GPU: README 의 `uv run torchrun` 또는 launch_finetune --num-gpus N.

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

## 검증 완료 상태 (pick_test 실데이터 기준)
- [x] 변환 (60→20fps, 메타 4종, float64 유지)
- [x] config 등록 (NEW_EMBODIMENT, arm RELATIVE / hand ABSOLUTE)
- [x] stats.json (28D) + relative_stats.json (16×7, arm 만)
- [x] 데이터 로딩 (state/action 차원, 비디오 디코딩, 영상-상태 정합, text)
- [ ] 실제 학습 (학습 서버에서, 다수 에피소드 수집 후)
- [ ] DP 변환 실데이터 검증 (DP 쓸 경우)
- [ ] 배포 경로 (worker_deploy_policy.py N1.7 import 경로 수정 — 학습 후)
