# G1 → Diffusion Policy 파이프라인 체크리스트

수집 → 변환 → 학습 → 배포 전 과정을 순서대로 정리한 운영 체크리스트.
2026-05-25 기준, 하드웨어 없이 검증 가능한 전 단계는 **실제 실행으로 검증 완료**
(아래 "사전 검증 완료" 참고). 남은 실질 작업은 ①충분한 실데이터 수집, ②실로봇 배포.

- 수집/배포 코드: `G1_Teleop_Datacollection` (teleop env)
- 학습 코드: `/home/kist/diffusion_policy` (umi env)
- 모델 형식: 28D joint = `[left_arm7, right_arm7, left_hand7, right_hand7]` (dex3, waist/head 제외)
- 카메라: RealSense 3대 `ego / wrist_l / wrist_r` @ **640×360** (`workers/worker_camera.py`) → camera_0/1/2
- 주기: 수집 60Hz → 학습 10Hz(6:1 다운샘플) → 배포 추론 10Hz / arm 실행 60Hz

---

## 0. 환경 1회 세팅 (하드웨어 불필요) — ✅ 이미 적용됨

DP 학습/배포 환경은 conda env **`umi`** (정식 `robodiff`는 미생성, umi가 대체).

- [x] **`diffusion_policy` editable 설치**: `conda run -n umi pip install -e /home/kist/diffusion_policy`
      → 이제 어느 cwd 에서도 `import diffusion_policy` 가능 (PYTHONPATH 불필요).
- [x] **`logging_mp` 호환 shim**: `workers/worker_deploy_dp.py` 상단에 `get_logger`/`basic_config`
      별칭 보강 추가. (현 PyPI `logging_mp 0.2.1` 은 `getLogger` 만 제공 → 미보강 시 import 크래시.)
- [ ] (선택) wandb 로그인: 안 하면 학습 시 `logging.mode=offline` 로 회피.

> 주의: G1 `setup.py` 의 `logging_mp` 는 버전 미핀이라 `pip install` 시 깨진 빌드가 올 수 있음.
> shim 으로 방어되지만, 근본적으로는 별칭 있는 빌드로 핀하는 것이 더 안전.

---

## 1. 데이터 수집 (하드웨어) — 실질 작업 ①

```bash
conda activate teleop
python main.py --hand dex3 --waist fixed --head off \
    --camera realsense --vr-input controller --lower-body hoist
```

- [ ] **토글이 28D 를 만든다**: `--hand dex3`(7+7), `--waist fixed`(waist 제외), `--head off`(head 제외).
      → state/action = 28D. **이 조합이 어긋나면 차원이 바뀌어 학습 config 와 불일치.**
- [ ] 카메라 3대(ego/wrist_l/wrist_r) 모두 인식되는지 (`utils/cameras.yaml` serial 확인).
- [ ] **충분한 에피소드 수집** — 정합 검증은 3 ep 로 끝났지만, 정책 품질엔 보통 수십~수백 ep 필요.
- [ ] 저장 위치: `record/<task>/` (LeRobot v2.1: data/chunk-*/*.parquet + videos/.../*.mp4 + meta/modality.json).
- [ ] 수집 직후 `meta/modality.json` 의 state 가 `left_arm/right_arm/left_hand/right_hand` (각 0:7/7:14/14:21/21:28) 인지 확인.

## 2. 변환 + 데이터 점검 (하드웨어 불필요) — ✅ 스크립트 검증됨

```bash
conda activate umi   # 또는 teleop(zarr 설치 시)

# LeRobot → DP zarr (60→10Hz)
python data_refinement/convert_to_dp.py \
    --src record/<task> --out record/<task>_dp.zarr --tgt-fps 10

# G1측 zarr invariant 검사
python data_refinement/verify_dp_dataset.py --zarr record/<task>_dp.zarr --tgt-fps 10

# DP측 로딩 검사 (DP repo 에서)
cd /home/kist/diffusion_policy
python verify_dp_loading.py --dataset-path /home/kist/G1_Teleop_Datacollection/record/<task>_dp.zarr
```

- [ ] 변환 로그에 `state_dim=28 action_dim=28`, 카메라 `(*,360,640,3)` 확인.
- [ ] `verify_dp_dataset.py` 전부 ✅ (길이/차원/NaN/fps=10/episode_ends).
- [ ] `verify_dp_loading.py` ✅ (obs `(To,3,360,640)`, action `(16,28)`, normalizer roundtrip).

## 3. 학습 (GPU, 하드웨어 불필요) — ✅ 종단 검증됨

```bash
cd /home/kist/diffusion_policy
conda activate umi
python train.py --config-name=train_diffusion_unet_g1_dex3_workspace \
    task.dataset_path=/home/kist/G1_Teleop_Datacollection/record/<task>_dp.zarr \
    logging.mode=offline      # wandb 로그인했으면 online
```

- [ ] **`task.dataset_path` 반드시 override** — config 기본값 `data/g1_dex3/multitask_test_dp.zarr` 는
      존재하지 않는 placeholder.
- [ ] 학습 지표 모니터링: `train_loss`, `val_loss`, **`train_action_mse_error`** 추세.
- [ ] 기본 600 epoch, 50 epoch 마다 checkpoint. 산출물:
      `data/outputs/<날짜>/<시각>_train_diffusion_unet_image_g1_dex3_image/checkpoints/{latest,epoch=...}.ckpt`
- 주요 하이퍼파라미터(검증된 기본값): horizon=16, n_obs=2, n_action=8, resize 180×320 → crop 162×288, resnet18×3(share 안 함), DDIM 100 step.

## 4. 배포 (하드웨어) — 실질 작업 ② / 실로봇 없이 미검증 구간

```bash
# 터미널 1 (teleop env): SHM 소유 + 로봇/카메라 워커 기동
conda activate teleop
python main.py --hand dex3 --waist fixed --head off --camera realsense \
    --vr-input controller --lower-body hoist

# 터미널 2 (umi env): SHM attach + 정책 추론
conda activate umi
python evaluate_dp.py --mode gr00t_rs_multi \
    --model-path /home/kist/diffusion_policy/data/outputs/.../checkpoints/latest.ckpt
```

- [ ] **UI 에서 Deploy=True** 토글해야 정책 로딩/추론 시작 (lazy load).
- [ ] hand_type 이 `dex3` 로 인식되는지 (teleop_config_shm).
- [ ] 3 카메라 SHM(`rs_ego/rs_wrist_l/rs_wrist_r`) 이 살아있고 640×360 인지 — 아니면 obs encoder
      의 `assert (3,360,640)` 에서 추론 크래시. (수집·배포 모두 같은 worker_camera.py 라 기본 일치.)
- [ ] **첫 배포는 안전하게**: 느린 속도 / 좁은 작업공간 / 비상정지 대기.
      (모델이 28D joint 절대 목표를 60Hz 로 ROBOT_ACTION SHM 에 기록 → 실제 arm/hand 구동.)
- [ ] action 평활화 옵션: `--action-method tem`(기본) / `maf` / `base`, `--decay`, `--window-size`.
- [ ] lag 통계 로그(`[Deploy] lag stats`) 로 추론 지연 모니터링.

---

## 사전 검증 완료 (2026-05-25, 로봇 없이)

다음은 실제 실행으로 통과 — "수집·변환·학습 후에야 알게 될 결함" 위험은 제거됨:

| 항목 | 검증 방식 |
|---|---|
| 원본 28D 레이아웃 / modality.json | parquet 직접 검사 (`pick_test2`) |
| `convert_to_dp.py` 60→10Hz, 3카메라, 28D | 원본 데이터로 실제 실행 |
| `verify_dp_dataset.py` invariant | 실행 통과 |
| DP `G1Dex3ImageDataset` 로딩/normalizer | 실행 통과 |
| `train.py` 전체 workspace (optimizer/ema/lr/val/sample/rollout/checkpoint) | GPU(RTX 5080) 종단 스모크 런, 체크포인트 생성 |
| 배포 `init_dp_policy` ckpt 로딩 + `predict_action` | 실제 ckpt 로딩 후 추론 (1,8,28) |
| 배포 obs/action 플러밍, 카메라 640×360 정합 | 합성 obs 종단 + worker_camera.py 확인 |

**미검증(실로봇 필수)**: SHM 실입출력, 라이브 카메라 실해상도, 다수 에피소드 학습 수렴 품질, 물리 동작·안전.

## 자주 막히는 지점 (Troubleshooting)

- `No module named 'diffusion_policy'` (배포/학습) → umi 에 editable 설치 확인 (`§0`).
- `module 'logging_mp' has no attribute 'get_logger'` → `worker_deploy_dp.py` shim 확인 (`§0`).
- `assert os.path.isdir` (학습 시작 직후) → `task.dataset_path` override 누락 (`§3`).
- 추론 시 shape assert 크래시 → 라이브 카메라 해상도 ≠ 360×640 (`§4`).
- wandb 로그인 프롬프트 → `logging.mode=offline`.
