import torch
import numpy as np
from einops import rearrange
import yaml

from policy import ACTPolicy, CNNMLPPolicy

def init_act_policy(yaml_path: str):
    """
    YAML 설정 파일에서 policy_class와 policy_config를 로드하여
    corresponding policy 객체를 반환합니다.
    Args:
        yaml_path: policy 설정이 담긴 YAML 파일 경로
    Returns:
        Initialized policy instance
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    policy_class = cfg.get('policy_class')
    policy_config = cfg.get('policy_config', {})
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy_class in YAML: {policy_class}")

    return policy


def get_act_obs(ts, camera_names, stats=None, device='cuda'):
    """
    시뮬/로봇 환경의 timestep 객체(ts)로부터 입력 관측(observation)을 구성합니다.
    Args:
        ts: env.reset() 또는 env.step() 반환 객체
        camera_names: 이미지 키 리스트
        stats: (선택) qpos 정규화 및 action 후처리에 필요한 통계 {qpos_mean, qpos_std, action_mean, action_std}
        device: torch 연산 디바이스
    Returns:
        qpos_tensor: 정규화된 qpos (1, state_dim) 형태 torch.Tensor
        image_tensor: (1, num_cams, c, h, w) 형태 torch.Tensor
    """
    obs = ts.observation
    qpos = np.array(obs['qpos'], dtype=np.float32)
    if stats and 'qpos_mean' in stats and 'qpos_std' in stats:
        qpos = (qpos - stats['qpos_mean']) / stats['qpos_std']
    qpos_tensor = torch.from_numpy(qpos).unsqueeze(0).to(device)

    images = []
    for cam in camera_names:
        img = obs['images'][cam]
        img = rearrange(img, 'h w c -> c h w')
        images.append(img)
    image_np = np.stack(images, axis=0).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)

    return qpos_tensor, image_tensor

def get_act_obs(ts, camera_names, stats=None, device='cuda'):
    """
    시뮬/로봇 환경의 timestep 객체(ts)로부터 입력 관측(observation)을 구성합니다.
    Args:
        ts: env.reset() 또는 env.step() 반환 객체
        camera_names: 이미지 키 리스트
        stats: (선택) qpos 정규화 및 action 후처리에 필요한 통계 {qpos_mean, qpos_std, action_mean, action_std}
        device: torch 연산 디바이스
    Returns:
        qpos_tensor: 정규화된 qpos (1, state_dim) 형태 torch.Tensor
        image_tensor: (1, num_cams, c, h, w) 형태 torch.Tensor
    """
    obs = ts.observation
    qpos = np.array(obs['qpos'], dtype=np.float32)
    if stats and 'qpos_mean' in stats and 'qpos_std' in stats:
        qpos = (qpos - stats['qpos_mean']) / stats['qpos_std']
    qpos_tensor = torch.from_numpy(qpos).unsqueeze(0).to(device)

    images = []
    for cam in camera_names:
        img = obs['images'][cam]
        img = rearrange(img, 'h w c -> c h w')
        images.append(img)
    image_np = np.stack(images, axis=0).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)

    return qpos_tensor, image_tensor


def get_action(policy, qpos_tensor, image_tensor, stats=None,
               past_actions_buffer=None, k=0.01):
    """
    항상 temporal ensemble을 적용해 행동을 계산합니다.
    Args:
        policy: make_policy_from_yaml()로 생성된 policy 인스턴스
        qpos_tensor: 전처리된 위치 상태 텐서 (1, state_dim)
        image_tensor: 전처리된 이미지 텐서 (1, num_cams, c, h, w)
        stats: (선택) action 후처리에 필요한 통계 {action_mean, action_std}
        past_actions_buffer: 이전 시점의 raw action 배열들을 저장하는 리스트
        k: 지수 감쇠 상수
    Returns:
        action: 후처리된 numpy 배열 형태 행동 벡터
        past_actions_buffer: 업데이트된 버퍼
    """
    # 정책 추론
    with torch.no_grad():
        all_actions = policy(qpos_tensor, image_tensor)  # [1, num_queries, action_dim]
    all_np = all_actions.squeeze(0).cpu().numpy()      # [num_queries, action_dim]

    # 버퍼 초기화
    if past_actions_buffer is None:
        past_actions_buffer = []
    # 최신 쿼리들 누적
    past_actions_buffer.append(all_np)
    # 최대 len 유지
    num_queries = all_np.shape[0]
    if len(past_actions_buffer) > num_queries:
        past_actions_buffer.pop(0)

    # 시간 축으로 연결
    concat = np.concatenate(past_actions_buffer, axis=0)  # [T, action_dim]
    # 지수 가중치
    weights = np.exp(-k * np.arange(len(concat)))
    weights = weights / weights.sum()
    # 가중 평균
    raw_np = (concat * weights[:, None]).sum(axis=0)

    # 후처리: 평균·표준편차 역변환
    if stats and 'action_mean' in stats and 'action_std' in stats:
        action = raw_np * stats['action_std'] + stats['action_mean']
    else:
        action = raw_np

    return action, past_actions_buffer
