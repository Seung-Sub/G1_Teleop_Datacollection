"""Single-source-of-truth for state/action vector layout (Phase M, PART3).

이 모듈이 modality.json 빌드 + 수집 측 state_vec/action_vec concat + deploy 측
obs/action 분할 3 곳 모두에서 사용된다.

핵심 원칙 (PART3 §8):
  1. 단일 진실 출처 = modality.json. 수집·배포가 모두 읽는다.
  2. 레이아웃 규칙은 한 함수 (build_state_layout). modality 빌드 = state_vec 빌드
     = deploy 구성.
  3. 토글 (waist_on, head_on, tactile_on, hand_type) 로 동적 결정. 새 플래그 최소화 —
     기존 CLI 의 --waist/--head/--tactile/--hand 그대로 재사용.
  4. off 경로 100% 불변.

토글-CLI 매핑 규칙 (수집 자동 적용):
  --waist fixed → waist_on = False (데이터에서 waist 제외)
  --waist hmd   → waist_on = True
  --head off    → head_on  = False
  --head dxl    → head_on  = True
  --tactile off → tactile_on = False
  --tactile on  → tactile_on = True (단 DEX3 sequence length N 확정 후만 의미)
  --hand inspire → hand_dim = 6 (left + right 6 씩)
  --hand dex3    → hand_dim = 7

video 키 (camera_roles) 는 main.py 의 cameras.yaml 활성 카메라 그대로 사용.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np


# state/action layout 항상 다음 순서 (build_state_layout 내부):
#   1. waist (3) — waist_on=True 일 때만
#   2. head  (2) — head_on=True 일 때만
#   3. left_arm  (7) — 항상
#   4. right_arm (7) — 항상
#   5. left_hand  (6 or 7) — 항상, hand_type 별로 dim
#   6. right_hand (6 or 7) — 항상, hand_type 별로 dim


def build_state_layout(
    hand_type: str = 'inspire',
    waist_on:  bool = True,
    head_on:   bool = True,
) -> List[Tuple[str, int]]:
    """(name, dim) 순서 리스트 반환. modality.json 의 state 키 순서/차원과 정확히 일치.

    tactile 은 별도 observation.sensor 컬럼 (학습 측 통상 관례) 이라 state layout 에서
    제외. tactile 처리는 build_modality_json 의 sensor 섹션 참고.
    """
    layout: List[Tuple[str, int]] = []
    if waist_on:
        layout.append(('waist', 3))
    if head_on:
        layout.append(('head', 2))
    layout.append(('left_arm', 7))
    layout.append(('right_arm', 7))
    hd = 7 if hand_type == 'dex3' else 6
    layout.append(('left_hand',  hd))
    layout.append(('right_hand', hd))
    return layout


def layout_to_index_map(layout: List[Tuple[str, int]]) -> Dict[str, Dict[str, int]]:
    """[(name, dim), ...] → {name: {'start': i, 'end': j}, ...} (modality.json 형식)."""
    idx = 0
    out: Dict[str, Dict[str, int]] = {}
    for name, dim in layout:
        out[name] = {'start': idx, 'end': idx + dim}
        idx += dim
    return out


def layout_max_end(layout: List[Tuple[str, int]]) -> int:
    return sum(d for _, d in layout)


def build_modality_json(
    hand_type: str = 'inspire',
    waist_on:  bool = True,
    head_on:   bool = True,
    tactile_on: bool = False,
    tactile_dim: int = 12,
    camera_roles: Optional[List[str]] = None,
) -> dict:
    """완전한 modality.json dict 반환.

    수집 측 _ensure_meta_modality 가 호출, 학습/배포 측이 동일 파일을 로드해 차원 정합.
    """
    layout = build_state_layout(hand_type, waist_on, head_on)
    state_idx = layout_to_index_map(layout)
    action_idx = dict(state_idx)  # state 와 action 동일 layout

    video: Dict[str, dict] = {}
    if camera_roles:
        for role in camera_roles:
            video[role] = {"original_key": f"observation.images.{role}"}

    m: dict = {
        "_comment": (
            "Phase M (PART3) 동적 빌드 결과. 토글: "
            f"hand={hand_type}, waist_on={waist_on}, head_on={head_on}, "
            f"tactile_on={tactile_on}, camera_roles={camera_roles}. "
            "utils/modality_layout.py 의 build_modality_json() 으로 생성."
        ),
        "state":  state_idx,
        "action": action_idx,
        "video":  video,
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }
    if tactile_on:
        m["sensor"] = {
            "tactile": {"start": 0, "end": int(tactile_dim)}
        }
    return m


# ============================================================================
# Concat / split helpers — 수집·배포 양측에서 사용
# ============================================================================

def concat_state_parts(
    parts: Dict[str, np.ndarray],
    layout: List[Tuple[str, int]],
) -> np.ndarray:
    """parts dict (name → 1-D np.array) 를 layout 순서로 concat."""
    arrays: List[np.ndarray] = []
    for name, dim in layout:
        if name not in parts:
            raise ValueError(f"concat_state_parts: missing '{name}' in parts")
        a = np.asarray(parts[name])
        if a.size != dim:
            raise ValueError(f"concat_state_parts: '{name}' expected dim {dim}, got {a.size}")
        arrays.append(a.reshape(-1))
    return np.concatenate(arrays) if arrays else np.zeros(0)


def split_state_vec(
    vec: np.ndarray,
    layout: List[Tuple[str, int]],
) -> Dict[str, np.ndarray]:
    """반대 방향. vec → {name: slice}. deploy 측에서 사용."""
    out: Dict[str, np.ndarray] = {}
    i = 0
    for name, dim in layout:
        out[name] = vec[i:i + dim]
        i += dim
    return out


# ============================================================================
# Loading existing modality.json (deploy 측 동적 obs_dict 구성용)
# ============================================================================

def layout_from_modality_json(m: dict) -> List[Tuple[str, int]]:
    """이미 저장된 modality.json (record/<task>/meta/) 을 받아 layout 재구성.

    deploy 측에서 학습에 쓴 데이터셋의 modality.json 을 로드 → 본 함수로 layout 추출
    → split_state_vec / obs_dict 구성에 사용.
    """
    state = m.get('state') or {}
    items = [(name, int(v['start']), int(v['end'])) for name, v in state.items()]
    items.sort(key=lambda x: x[1])
    return [(name, end - start) for name, start, end in items]
