#!/usr/bin/env python3
"""
순차적 데이터셋 병합 스크립트
0901_pick_and_place_apple의 에피소드들을 0,1,2,3으로 재정렬하고
0901_pick_and_place_apple_2의 에피소드들을 4,5,6,7,8...로 추가합니다.
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Set

def extract_episode_number(filename: str) -> int:
    """파일명에서 에피소드 번호를 추출합니다."""
    match = re.search(r'episode_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def get_episode_files(base_path: Path) -> Dict[int, List[Path]]:
    """주어진 경로에서 모든 에피소드 파일들을 수집합니다."""
    episode_files = {}
    
    # data 파일들
    data_path = base_path / "data" / "chunk-000"
    if data_path.exists():
        for file in data_path.glob("episode_*.parquet"):
            episode_num = extract_episode_number(file.name)
            if episode_num >= 0:
                if episode_num not in episode_files:
                    episode_files[episode_num] = []
                episode_files[episode_num].append(file)
    
    # video 파일들
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

def sequential_merge():
    """모든 데이터를 보존하면서 순차적으로 병합합니다."""
    base_dir = Path("/home/ansur/Ansur_unitree_teleop/record")
    
    # --- 병합할 데이터셋 목록 ---
    # 여기서 원하는 데이터셋 폴더 이름을 순서대로 나열합니다.
    # 첫 번째 데이터셋이 병합의 기준(primary)이 됩니다.
    dataset_names = [
        # "downglue_J",
        # "downglue_J_1",
        # "downglue_p_1",
        "standglue_J",
        "standglue_J_2",
        "standglue_p_3",
        "standglue_p_4",
    ]
    # -------------------------

    dataset_paths = [base_dir / name for name in dataset_names]

    if not dataset_paths:
        print("병합할 데이터셋이 없습니다.")
        return None

    # 새로운 병합 폴더 생성
    merged_folder_name = "0116_glue_bimanual_stand"
    primary_path = base_dir / merged_folder_name
    
    # 기존 병합 폴더가 있으면 백업
    if primary_path.exists():
        backup_original_data(primary_path)
    
    # 새 폴더 생성
    primary_path.mkdir(exist_ok=True)
    
    secondary_paths = dataset_paths

    for path in dataset_paths:
        if not path.exists():
            print(f"오류: '{path}' 경로를 찾을 수 없습니다. 목록을 확인해주세요.")
            return None

    print("=== 새로운 폴더에 병합 시작 ===")
    
    # 모든 데이터셋의 에피소드 파일들 수집
    print("에피소드 파일들을 수집 중...")
    all_episodes = []
    
    print(f"\n📋 병합 계획:")
    for path in dataset_paths:
        episodes = get_episode_files(path)
        if not episodes:
            print(f"경고: '{path.name}'에서 에피소드를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        sorted_episodes = sorted(episodes.keys())
        print(f"- {path.name}: {len(sorted_episodes)}개 에피소드")
        
        for episode_num in sorted_episodes:
            all_episodes.append((path.name, episode_num, episodes[episode_num]))

    total_episodes = len(all_episodes)
    if total_episodes == 0:
        print("병합할 에피소드가 없습니다. 작업을 중단합니다.")
        return None

    print(f"총 에피소드: {total_episodes}개")
    print(f"최종 번호: 0 ~ {total_episodes - 1}")
    
    # 순차적 재번호화 실행
    renumber_all_episodes(all_episodes, primary_path)
    
    print("=== 모든 데이터 보존 순차적 병합 완료 ===")
    return primary_path, secondary_paths

def renumber_all_episodes(all_episodes: List, destination: Path):
    """모든 에피소드들을 순차적으로 재번호화합니다."""
    print("모든 에피소드 번호를 순차적으로 재정렬 중...")
    
    # 임시 디렉토리 생성
    temp_dir = destination.parent / "temp_preserve_all_merge"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # 임시 디렉토리 구조 생성
    temp_data_dir = temp_dir / "data" / "chunk-000"
    temp_video_dirs = {
        "ego_left_view": temp_dir / "videos" / "chunk-000" / "observation.images.ego_left_view",
        "ego_realsense": temp_dir / "videos" / "chunk-000" / "observation.images.ego_realsense", 
        "ego_right_view": temp_dir / "videos" / "chunk-000" / "observation.images.ego_right_view"
    }
    
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    for video_dir in temp_video_dirs.values():
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # 모든 에피소드를 순차적으로 처리
    print(f"\n모든 에피소드 처리:")
    for new_episode_index, (source, old_episode_num, files) in enumerate(all_episodes):
        new_episode_name = f"episode_{new_episode_index:06d}"
        
        print(f"  {source} {old_episode_num:06d} → {new_episode_index:06d}")
        
        copy_episode_files(files, new_episode_name, temp_data_dir, temp_video_dirs)
    
    # 임시 데이터를 목적지로 이동
    print("\n임시 데이터를 목적지로 이동 중...")
    if (destination / "data").exists():
        shutil.rmtree(destination / "data")
    if (destination / "videos").exists():
        shutil.rmtree(destination / "videos")
    
    shutil.move(str(temp_dir / "data"), str(destination / "data"))
    shutil.move(str(temp_dir / "videos"), str(destination / "videos"))
    
    # 임시 디렉토리 정리
    shutil.rmtree(temp_dir)
    
    total_episodes = len(all_episodes)
    print(f"\n모든 데이터 보존 재정렬 완료! 총 {total_episodes}개의 에피소드가 0~{total_episodes-1}로 정렬되었습니다.")

def copy_episode_files(files: List[Path], new_episode_name: str, 
                      temp_data_dir: Path, temp_video_dirs: Dict[str, Path]):
    """에피소드 파일들을 새로운 이름으로 복사합니다."""
    for file_path in files:
        if file_path.suffix == '.parquet':
            # data 파일
            new_file_path = temp_data_dir / f"{new_episode_name}.parquet"
            shutil.copy2(file_path, new_file_path)
        elif file_path.suffix == '.mp4':
            # video 파일
            parent_name = file_path.parent.name
            if "ego_left_view" in parent_name:
                new_file_path = temp_video_dirs["ego_left_view"] / f"{new_episode_name}.mp4"
            elif "ego_realsense" in parent_name:
                new_file_path = temp_video_dirs["ego_realsense"] / f"{new_episode_name}.mp4"
            elif "ego_right_view" in parent_name:
                new_file_path = temp_video_dirs["ego_right_view"] / f"{new_episode_name}.mp4"
            else:
                continue
            
            shutil.copy2(file_path, new_file_path)

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
            print(f"결과: '{primary_path.name}'에 모든 에피소드가 0부터 순차적으로 정렬되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("백업 데이터를 확인해주세요.")

if __name__ == "__main__":
    main()
