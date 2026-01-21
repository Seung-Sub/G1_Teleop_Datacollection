#!/usr/bin/env python3
"""
순차적 데이터셋 병합 및 처리 스크립트 (V2 - 하드코딩 버전)

- dataset_names에 정의된 순서대로 데이터셋을 병합합니다.
- 모든 에피소드를 0부터 순차적으로 재번호화합니다.
- 각 .parquet 파일에 'task_index'와 'episode_index' 컬럼을 추가/수정합니다.
- 각 .parquet 파일에 'observation.img_state_delta' 컬럼을 추가합니다.

(이 스크립트는 모든 경로와 설정을 내부에 하드코딩하며, 별도 인자(arg)가 필요 없습니다.)
"""

import os
import shutil
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# parquet 파일 처리를 위해 pandas와 진행상황 표시를 위해 tqdm 추가
import pandas as pd
from tqdm import tqdm

# img_state_delta 계산을 위해 cv2와 numpy 추가
try:
    import cv2
except ImportError:
    print("경고: OpenCV(cv2)가 설치되지 않았습니다. 'observation.img_state_delta'는 0.0으로 채워집니다.", file=sys.stderr)
    print("      'pip install opencv-python-headless'로 설치할 수 있습니다.", file=sys.stderr)
    cv2 = None

import numpy as np

# --- 하드코딩 설정: 델타 계산 설정 ---
# 'observation.img_state_delta'를 계산할 때 사용할 비디오 뷰의 이름
# (요청 사항: .../videos/chunk-000/observation.images.ego_left_view/)
PRIMARY_VIEW_KEY = "observation.images.ego_left_view"

# --- 델타 계산 헬퍼 함수 ---

def calculate_img_state_delta(video_path: Path, expected_length: int) -> List[float]:
    """
    비디오 파일로부터 per-frame gray abs diff mean을 계산합니다.
    """
    if cv2 is None:
        if 'warned_cv2' not in globals():
            print(f"\n경고: cv2가 없어 img_state_delta를 계산할 수 없습니다. 0.0으로 채웁니다.", file=sys.stderr)
            globals()['warned_cv2'] = True
        return [0.0] * expected_length

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\n경고: 비디오 열기 실패 {video_path}. 0.0으로 채웁니다.", file=sys.stderr)
        return [0.0] * expected_length

    deltas = []
    ok, prev_frame = cap.read()
    if not ok:
        print(f"\n경고: 첫 프레임 읽기 실패 {video_path}. 0.0으로 채웁니다.", file=sys.stderr)
        cap.release()
        return [0.0] * expected_length

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        diff = np.abs(gray - prev_gray)
        deltas.append(float(diff.mean()))
        prev_gray = gray
    cap.release()

    # 길이 보정 (제공된 로직)
    if len(deltas) != expected_length:
        # print(f"  ~ 길이 불일치: {video_path.name} frames={len(deltas)} rows={expected_length} -> 보정")
        if len(deltas) < expected_length:
            deltas += [0.0] * (expected_length - len(deltas))
        else:
            deltas = deltas[:expected_length]
    
    return deltas

# --- 기존 헬퍼 함수 ---

def extract_episode_number(filename: str) -> int:
    """파일명에서 에피소드 번호를 추출합니다."""
    match = re.search(r'episode_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def get_episode_files(base_path: Path) -> Dict[int, List[Path]]:
    """주어진 경로에서 모든 에피소드 파일들을 수집합니다."""
    episode_files = {}
    
    # 하드코딩된 Parquet 경로 (요청 사항: .../data/chunk-000/)
    data_path = base_path / "data" / "chunk-000"
    if data_path.exists():
        for file in data_path.glob("episode_*.parquet"):
            episode_num = extract_episode_number(file.name)
            if episode_num >= 0:
                if episode_num not in episode_files:
                    episode_files[episode_num] = []
                episode_files[episode_num].append(file)
    
    # 하드코딩된 Video 기본 경로 (요청 사항: .../videos/chunk-000/)
    video_base = base_path / "videos" / "chunk-000"
    if video_base.exists():
        for view_dir in video_base.iterdir():
            if view_dir.is_dir():
                for file in view_dir.glob("episode_*.mp4"):
                    episode_num = extract_episode_number(file.name)
                    if episode_num >= 0:
                        if episode_num not in episode_files:
                            episode_files[episode_num] = []
                        episode_files[episode_num].append(file)
    
    return episode_files

def backup_original_data(source_path: Path):
    """원본 데이터를 백업합니다."""
    backup_path = source_path.parent / f"{source_path.name}_backup"
    if not backup_path.exists():
        print(f"원본 데이터를 백업 중: {backup_path}")
        shutil.copytree(source_path, backup_path)
        print("백업 완료!")
    else:
        print(f"백업이 이미 존재합니다: {backup_path}")

# --- 메인 로직 함수 ---

def sequential_merge():
    """모든 데이터를 보존하면서 순차적으로 병합하고 처리합니다."""
    
    # --- 하드코딩 설정: 기본 경로 ---
    # (요청 사항: /home/ansur/Ansur_unitree_teleop/record)
    base_dir = Path("/home/ansur/Ansur_unitree_teleop/record/")
    
    # --- 하드코딩 설정: 병합할 데이터셋 목록 ---
    # [task_index, "폴더명"] 형식으로 순서대로 나열합니다.
    # (요청 사항: "1030_pick_and_place_apple" 등)
    dataset_names = [
        [0, "0116_glue_bimanual_down"],
        [1, "0116_glue_bimanual_stand"],
    ]
    # -------------------------

    dataset_paths = [base_dir / name[1] for name in dataset_names]

    if not dataset_paths:
        print("병합할 데이터셋이 없습니다.")
        return None

    # --- 하드코딩 설정: 최종 병합 폴더 ---
    # (요청 사항: "1030_pick_and_place")
    merged_folder_name = "0116_glue_bimanual"
    primary_path = base_dir / merged_folder_name
    
    if primary_path.exists():
        backup_original_data(primary_path)
    
    primary_path.mkdir(exist_ok=True)
    
    for path in dataset_paths:
        if not path.exists():
            print(f"오류: '{path}' 경로를 찾을 수 없습니다. 목록을 확인해주세요.")
            return None

    print("=== 새로운 폴더에 병합 시작 ===")
    
    print("에피소드 파일들을 수집 중...")
    all_episodes: List[Tuple[str, int, List[Path], int]] = []
    
    print(f"\n📋 병합 계획:")
    for task_index, dataset_name in dataset_names:
        path = base_dir / dataset_name
        episodes = get_episode_files(path)
        
        if not episodes:
            print(f"경고: '{path.name}'에서 에피소드를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        sorted_episodes = sorted(episodes.keys())
        print(f"- {path.name} (Task {task_index}): {len(sorted_episodes)}개 에피소드")
        
        for episode_num in sorted_episodes:
            all_episodes.append((path.name, episode_num, episodes[episode_num], task_index))

    total_episodes = len(all_episodes)
    if total_episodes == 0:
        print("병합할 에피소드가 없습니다. 작업을 중단합니다.")
        return None

    print(f"총 에피소드: {total_episodes}개")
    print(f"최종 번호: 0 ~ {total_episodes - 1}")
    
    renumber_all_episodes(all_episodes, primary_path)
    
    print("=== 모든 데이터 보존 순차적 병합 및 처리 완료 ===")
    return primary_path, dataset_paths

def renumber_all_episodes(all_episodes: List, destination: Path):
    """모든 에피소드들을 순차적으로 재번호화하고 .parquet 파일을 처리합니다."""
    print("모든 에피소드 번호를 순차적으로 재정렬 및 처리 중...")
    
    temp_dir = destination.parent / "temp_preserve_all_merge"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    temp_data_dir = temp_dir / "data" / "chunk-000"
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    
    temp_video_base_dir = temp_dir / "videos" / "chunk-000"

    for new_episode_index, (source, old_episode_num, files, task_index) in enumerate(tqdm(all_episodes, desc="전체 에피소드 처리")):
        new_episode_name = f"episode_{new_episode_index:06d}"
        
        process_and_copy_files(
            files, new_episode_name, new_episode_index, task_index, 
            temp_data_dir, temp_video_base_dir
        )
    
    print("\n임시 데이터를 목적지로 이동 중...")
    if (destination / "data").exists():
        shutil.rmtree(destination / "data")
    if (destination / "videos").exists():
        shutil.rmtree(destination / "videos")
    
    shutil.move(str(temp_dir / "data"), str(destination / "data"))
    if temp_video_base_dir.exists():
        shutil.move(str(temp_video_base_dir.parent), str(destination / "videos"))
    
    shutil.rmtree(temp_dir)
    
    total_episodes = len(all_episodes)
    print(f"\n모든 데이터 보존 및 재정렬 완료! 총 {total_episodes}개의 에피소드가 0~{total_episodes-1}로 정렬되었습니다.")

def process_and_copy_files(
    files: List[Path], new_episode_name: str, new_episode_index: int, task_index: int,
    temp_data_dir: Path, temp_video_base: Path
):
    """
    에피소드 파일들을 분류하고, .parquet 처리 및 모든 파일을 새 이름으로 복사합니다.
    """
    parquet_file_path = None
    video_file_for_delta = None
    all_video_files = [] # 모든 비디오 파일 (복사용)

    # --- 1. 파일 분류 ---
    for file_path in files:
        if file_path.suffix == '.parquet':
            parquet_file_path = file_path
        elif file_path.suffix == '.mp4':
            all_video_files.append(file_path)
            # 델타 계산에 사용할 기본 비디오 뷰(PRIMARY_VIEW_KEY)인지 확인
            if PRIMARY_VIEW_KEY in file_path.parent.name:
                video_file_for_delta = file_path

    # --- 2. Parquet 파일 처리 ---
    if not parquet_file_path:
        print(f"\n경고: Parquet 파일 없음. {new_episode_name} 건너뜀.", file=sys.stderr)
        return

    try:
        df = pd.read_parquet(parquet_file_path)
        
        # 2-1. task_index와 episode_index 추가
        df['task_index'] = task_index
        df['episode_index'] = new_episode_index

        # 2-2. observation.img_state_delta 계산 및 추가
        deltas = []
        if video_file_for_delta:
            deltas = calculate_img_state_delta(video_file_for_delta, len(df))
        else:
            # 델타 계산용 비디오가 없는 경우
            if 'warned_no_video' not in globals():
                print(f"\n경고: {PRIMARY_VIEW_KEY} 비디오 없음. {new_episode_name}의 img_state_delta를 0.0으로 채웁니다.", file=sys.stderr)
                globals()['warned_no_video'] = True
            deltas = [0.0] * len(df)
        
        df["observation.img_state_delta"] = np.asarray(deltas, dtype=np.float32)

        # 2-3. 수정된 DataFrame을 새 경로에 저장
        new_file_path = temp_data_dir / f"{new_episode_name}.parquet"
        df.to_parquet(new_file_path, index=False)
        
    except Exception as e:
        print(f"\n오류: {parquet_file_path} 처리 중 예외 발생: {e}", file=sys.stderr)
        return # 이 에피소드 처리를 중단합니다.

    # --- 3. 모든 비디오 파일 복사 ---
    for file_path in all_video_files:
        view_name = file_path.parent.name
        target_dir = temp_video_base / view_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        new_file_path = target_dir / f"{new_episode_name}.mp4"
        shutil.copy2(file_path, new_file_path)

# --- 정리 함수 ---

def cleanup_secondary_dataset(secondary_paths: List[Path]):
    """병합이 완료된 후 secondary 데이터셋을 정리합니다."""
    if not secondary_paths:
        return

    print("\n병합이 완료되었습니다. 다음 소스 데이터셋은 수동으로 삭제할 수 있습니다:")
    for path in secondary_paths:
        print(f"- {path}")
    print("소스 데이터셋을 유지합니다.")

def main():
    """메인 함수"""
    try:
        result = sequential_merge()
        if result:
            primary_path, secondary_paths_to_clean = result
            if secondary_paths_to_clean:
                cleanup_secondary_dataset(secondary_paths_to_clean)
            print("\n모든 작업이 완료되었습니다!")
            print(f"결과: '{primary_path.name}'에 모든 에피소드가 0부터 순차적으로 정렬 및 처리되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("백업 데이터를 확인해주세요.")

if __name__ == "__main__":
    main()
