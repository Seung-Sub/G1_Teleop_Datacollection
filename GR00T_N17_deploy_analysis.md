# GR00T N1.7 Deploy 정합 분석 (worker_deploy_policy.py)

실제 N1.7 코드(gr00t/policy/gr00t_policy.py, configs/data/data_config.py 등) 정독으로
확정한 사실. 현재 worker_deploy_policy.py 는 N1.5/1.6 방식이라 N1.7 에서 그대로는
동작하지 않음. 아래는 코드 근거가 있는 사실만 기재.

## 1. Gr00tPolicy 생성 방식이 완전히 다름 (확정)

**현재 (worker_deploy_policy.py L167-189, init_gr00t_policy):**
```python
from gr00t.experiment.data_config import DATA_CONFIG_MAP   # ← 경로 틀림
data_config = DATA_CONFIG_MAP[data_config_key]
Gr00tPolicy(model_path=, modality_config=, modality_transform=,
            embodiment_tag=, denoising_steps=, device=)   # ← 인자 다름
```

**N1.7 실제 (gr00t/policy/gr00t_policy.py L74-81):**
```python
Gr00tPolicy(
    embodiment_tag: EmbodimentTag | str,
    model_path: str,
    *,
    device: int | str,
    strict: bool = True,
)
```
- `modality_config`, `modality_transform`, `denoising_steps` **인자 없음**.
- modality config 는 **체크포인트의 processor 에서 자동 로딩** (L115 AutoProcessor.from_pretrained,
  L120 get_modality_configs). 공식 policy.md: "After finetuning, the config is saved in
  the checkpoint and loaded automatically during inference."
- `gr00t/configs/data/data_config.py` 에는 `DATA_CONFIG_MAP` 이 **없음** (DataConfig 클래스만).
  즉 deploy 에서 DATA_CONFIG_MAP 자체가 불필요.

**수정:** init_gr00t_policy 를 아래로 교체 (학습 시 쓴 g1_dex3_config.py 가 체크포인트에
저장되어 자동 로딩되므로, embodiment_tag + model_path + device 만 필요).
```python
def init_gr00t_policy(model_path, embodiment_tag="new_embodiment", device="cuda"):
    from gr00t.policy import Gr00tPolicy
    return Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
```
denoising_steps 는 N1.7 Gr00tPolicy 생성 인자가 아님(모델 설정에 내장). 필요 시 별도 확인.

## 2. obs / action 키 형식 — PolicyWrapper vs Gr00tPolicy

N1.7 은 두 인터페이스 제공:
- **Gr00tPolicy (직접)**: **중첩** dict. `obs["video"][key]`, `obs["state"][key]`,
  `obs["language"][lang_key]`. 출력 action 키 = modality_keys 그대로 (`left_arm` 등, 접두사 없음).
- **PolicyWrapper (sim 호환)**: **평탄** 키. `obs["video.cam"]`, `obs["state.joints"]`,
  language 는 `obs[lang_key]` (tuple/list[str]). 출력 `action.{key}`.

공식 권고 (gr00t_policy.py L493-495): "custom robots... should use **Gr00tPolicy directly**".

**현재 worker_deploy 의 평탄 형식**(L122-129 build_obs_dict, L132-160 action_to_array)은
PolicyWrapper 형식에 가까움. 두 가지 선택:

### 방식 A — Gr00tPolicy 직접 + 중첩 dict (공식 권고)
build_obs_dict 를 중첩 구조로 재작성:
```python
obs = {
    "video":    { role: rgb[None, None, ...] for role, rgb in frames.items() },   # (1,1,H,W,C) uint8
    "state":    { name: parts[name][None, None, :] for name, _ in layout },        # (1,1,D) float32
    "language": { policy.language_key: [[task_name]] },                            # list[list[str]] (1,1)
}
action = policy.get_action(obs)        # 반환 키: left_arm, right_arm, left_hand, right_hand (접두사 X)
# action[key] shape (1, T, D)
```
action_to_array 의 키도 `action.left_arm` → `left_arm` 으로 (접두사 제거).

### 방식 B — PolicyWrapper + 평탄 (현재 형식 거의 유지)
```python
from gr00t.policy import PolicyWrapper
policy = PolicyWrapper(Gr00tPolicy(embodiment_tag=, model_path=, device=))
# obs 평탄: "video.ego", "state.left_arm", language 는 obs[lang_key]=tuple[str]
# 출력 "action.left_arm" (현재 action_to_array 와 호환)
```
단 PolicyWrapper 의 language 입력은 `obs[lang_key]` = `tuple[str]`/`list[str]` (B,) 형식이며
key 는 `annotation.human.task_description` (접두사 없음). 현재 worker_deploy 의
`annotation.human.action.task_description` + `np.array([...], dtype=object)` 와 다름 → 수정 필요.

**권고: 방식 A (Gr00tPolicy 직접).** 공식 권고이고, sim wrapper 의존 없이 명확. 중첩 dict 로
재작성하는 비용은 있으나 인터페이스가 안정적.

## 3. language 키 (확정)

- 학습 config (g1_dex3_config.py L91): `annotation.human.task_description`
- N1.7 Gr00tPolicy 는 `self.language_key = modality_configs["language"].modality_keys[0]`
  (L162-166) — **체크포인트에서 자동**. 즉 deploy 에서 하드코딩 말고 `policy.language_key` 사용.
- 현재 worker_deploy 의 `annotation.human.action.task_description` (L123) 는 **틀림**.
  → 방식 A 에서 `obs["language"][policy.language_key] = [[task_name]]` 로 자동 정합.

## 4. obs 형식 상세 (N1.7 Gr00tPolicy get_action assert 기준)

- video: `(B, T, H, W, C)` **uint8**, T == len(video delta_indices)=1, C==3 (L267-283)
- state: `(B, T, D)` **float32**, T == len(state delta_indices)=1 (L311-322)
- language: `obs["language"][key]` = `list[list[str]]` (B, 1) (L342, PolicyWrapper L672)

현재 build_obs_dict 의 `[None, None, :]` (1,1,D) 와 `[None,None,...]` (1,1,H,W,C) 는
B=1,T=1 이라 맞음. dtype 만 보장하면 됨 (state float32, video uint8 — 이미 처리됨 L62,L100-105).

## 5. 변경 불필요 (그대로 정상)

- **upsample_actions** (L66-85): 20→50Hz, arm spline k=5 / hand linear k=1. GR00T action chunk
  를 로봇 제어 주파수로 보간. N1.7 무관하게 정상. action chunk shape (T,D) 처리.
- **robot_action_shm 출력** (L570-): action_np[19] + hand_action[14]. Teleop 과 동일 경로. 정상.
- **modality.json layout 기반 동적 obs** (L351-371): 학습 데이터셋 layout 사용. 정상.
- **lag compensation, cross-fade** 등: 추론 인터페이스와 무관. 정상.

## 요약 — 수정 범위

| 함수 | 수정 |
|---|---|
| init_gr00t_policy (L167) | N1.7 시그니처로 교체 (embodiment_tag, model_path, device). DATA_CONFIG_MAP 제거 |
| _init_gr00t_policy (L378) | init 호출 인자 변경 |
| build_obs_dict (L109) | 중첩 dict (video/state/language) + policy.language_key |
| action_to_array (L132) | action 키 접두사 제거 (action.left_arm → left_arm) |
| gr00t_inference (L473) | first_key, action[key] 인덱싱을 새 키로 |

upsample/robot_action/lag/layout 등 하드웨어 로직은 그대로 유지.
