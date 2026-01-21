#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Convert Parquet+MP4 → HDF5 for multiple episodes
Usage: 스크립트 상단의 디렉터리 경로를 지정 후, 실행만 하면 자동으로 모든 episode를 변환합니다.
"""
import os
import glob
import numpy as np
import pandas as pd
import cv2
import h5py

# === 사용자 정의 디렉터리 설정 ===
PARQUET_DIR = '/home/ansur/teleop_data_collector/record/g1-pick-apple/data/chunk-000'
VIDEO_DIR   = '/home/ansur/teleop_data_collector/record/g1-pick-apple/videos/chunk-000/observation.images.ego_realsense'
OUTPUT_DIR  = '/media/ansur/684845314844FEF6/Junho_IL_data/g1_hdf5_data/250802'  # 저장 디렉터리
CAMERA_NAME = 'ego_realsense'
SIM_FLAG    = False  # 실제 시뮬레이션 데이터인지 여부

# 파일 패턴
PARQUET_PATTERN = os.path.join(PARQUET_DIR, 'episode_*.parquet')
VIDEO_PATTERN   = os.path.join(VIDEO_DIR,   'episode_*.mp4')


def load_parquet(parquet_path):
    """Parquet 파일에서 각 타임스텝의 observation.state (qpos)와 action 리스트를 읽어 NumPy 배열로 반환합니다."""
    df = pd.read_parquet(parquet_path)
    qpos_list   = df['observation.state'].to_list()
    action_list = df['action'].to_list()
    qpos   = np.stack(qpos_list,   axis=0).astype(np.float32)
    action = np.stack(action_list, axis=0).astype(np.float32)
    return qpos, action


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def save_hdf5(out_path, qpos, action, images, cam_name, sim_flag=False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.attrs['sim'] = sim_flag
        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=qpos, dtype='float32')
        obs.create_dataset('qvel', data=np.zeros_like(qpos), dtype='float32')
        img_grp = obs.create_group('images')
        img_grp.create_dataset(
            cam_name,
            data=images,
            dtype='uint8',
            chunks=(1, images.shape[1], images.shape[2], 3)
        )
        f.create_dataset('action', data=action, dtype='float32')
    print(f"[INFO] Saved: {out_path}")


def main():
    parquet_files = sorted(glob.glob(PARQUET_PATTERN))
    video_files   = sorted(glob.glob(VIDEO_PATTERN))
    video_map = {os.path.splitext(os.path.basename(v))[0]: v for v in video_files}

    for p_path in parquet_files:
        base = os.path.splitext(os.path.basename(p_path))[0]  # 'episode_000012'
        v_path = video_map.get(base)
        if not v_path:
            print(f"[WARN] 비디오 없음: {base}")
            continue

        # 에피소드 번호를 추출하고, 숫자만 사용해 파일명 생성
        try:
            idx = int(base.split('_')[1])
        except Exception:
            idx = base  # 예외시 원본 사용
        filename = f"episode_{idx}.hdf5"
        out_path = os.path.join(OUTPUT_DIR, filename)

        # qpos 및 action 로드
        qpos, action = load_parquet(p_path)
        # 영상 프레임 로드
        images       = load_video_frames(v_path)

        # 길이 동기화
        T = min(len(qpos), len(action), len(images))
        if len(qpos) != T or len(action) != T or len(images) != T:
            print(f"[WARN] {base} 길이 불일치 → {T}으로 트리밍")
        qpos   = qpos[:T]
        action = action[:T]
        images = images[:T]

        # 저장
        save_hdf5(out_path, qpos, action, images, CAMERA_NAME, SIM_FLAG)

if __name__ == '__main__':
    main()
