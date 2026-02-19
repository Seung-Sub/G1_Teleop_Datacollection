import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

mode = 'only_kistar'

def process_action_dimension(action_array):
    raw_length=41
    """
    raw_length차원 action 배열에서 특정 인덱스를 제거하고 일부를 평균내어 차원을 축소합니다.
    (기존 로직 유지)
    """
    if not isinstance(action_array, np.ndarray):
        action_array = np.array(action_array)

    if action_array.shape == (raw_length,):
        if mode == 'reduced':
            indices_to_delete = [3, 4]
            modified_action = action_array.copy()

            modified_action[19] = action_array[19]
            modified_action[20] = action_array[20]
            thumb_flexion = (action_array[21] + action_array[22]) / 2
            modified_action[21] = thumb_flexion
            index_flexion = (action_array[24] + action_array[25] + action_array[26]) / 3
            middle_flexion = (action_array[28] + action_array[29] + action_array[30]) / 3
            ring_flexion = (action_array[32] + action_array[33] + action_array[34]) / 3
            modified_action[22] = index_flexion
            modified_action[23] = middle_flexion
            modified_action[24] = ring_flexion

            modified_action = modified_action[:25]
            modified_action = np.delete(modified_action, indices_to_delete)
            #print('reduced mode action 수정 완료')
            # 참고: 이 로직은 최종적으로 (23,) shape을 반환합니다.
            return modified_action

        elif mode == 'reduced_v2':
            indices_to_delete = [3, 4]
            modified_action = action_array.copy()

            modified_action[19] = action_array[19]
            modified_action[20] = action_array[20]
            thumb_flexion = (action_array[21] + action_array[22]) / 2
            modified_action[21] = thumb_flexion
            index_flexion = (3*action_array[24] + action_array[25] + action_array[26]) / 5
            middle_flexion = (3*action_array[28] + action_array[29] + action_array[30]) / 5
            ring_flexion = (3*action_array[32] + action_array[33] + action_array[34]) / 5
            modified_action[22] = index_flexion
            modified_action[23] = middle_flexion
            modified_action[24] = ring_flexion

            modified_action = modified_action[:25]
            modified_action = np.delete(modified_action, indices_to_delete)

            return modified_action
            
        elif mode == 'full':
            indices_to_delete = [3, 4, 34]
            modified_action = action_array.copy()
            modified_action = np.delete(modified_action, indices_to_delete)
            #print('full mode action 수정 완료')
            return modified_action

        elif mode == 'reduced_v3':
            # 0~2: waist
            # 3~4: neck
            # 5~11: left arm
            # 12~18: right arm
            # 19~24: inspire hand
            # 25~40: kistar hand

            # 25~40 (16 DOF) -> 25~34 (10 DOF)

            indices_to_delete = [3, 4]
            modified_action = action_array.copy()

            modified_action[25] = action_array[25]
            modified_action[26] = action_array[26]
            modified_action[27] = action_array[27]
            modified_action[28] = action_array[28]
            # action_array[29] : Index Ab/Ad
            modified_action[29] = action_array[30]
            modified_action[30] = (action_array[31] + action_array[32]) / 2

            # action_array[33] : Middle Ab/Ad
            modified_action[31] = action_array[34]
            modified_action[32] = (action_array[35] + action_array[36]) / 2

            # action_array[37] : Ring Ab/Ad
            modified_action[33] = (3*action_array[38] + action_array[39] + action_array[40]) / 5

            modified_action = modified_action[:34]

            modified_action = np.delete(modified_action, indices_to_delete)
            #print('full mode action 수정 완료')
            return modified_action

        elif mode == 'only_kistar':
            # 0~2  : waist
            # 3~4  : neck (REMOVE)
            # 5~11 : left arm (REMOVE index 11) 
            # 12~18: right arm
            # 19~24: inspire hand (REMOVE ALL)
            # 25~40: kistar hand (KEEP ALL, 16 DOF)

            modified_action = action_array.copy()

            # 제거할 인덱스들
            # neck: 3,4
            # left arm wrist roll: 11
            # inspire hand: 19~24
            indices_to_delete = [3, 4, 11, 19, 20, 21, 22, 23, 24]

            # 실제로는 값 변형 없이 그대로 사용
            # (kistar hand는 전부 살림)

            modified_action = np.delete(modified_action, indices_to_delete)

            return modified_action



    else:
        print(f"Warning: 예상치 못한 action shape {action_array.shape}을 발견하여 수정하지 않았습니다.", file=sys.stderr)
        return action_array

def process_observation_dimension(observation_array):
    raw_length=41
    """
    observation 배열의 차원을 수정합니다.
    (!!! 중요: 이 함수 내부는 사용자의 데이터에 맞게 직접 수정해야 합니다 !!!)
    """
    if not isinstance(observation_array, np.ndarray):
        observation_array = np.array(observation_array)

    if observation_array.shape == (raw_length,):
        if mode == 'reduced':
            indices_to_delete = [3, 4]
            modified_observation = observation_array.copy()

            modified_observation[19] = observation_array[19]
            modified_observation[20] = observation_array[20]
            thumb_flexion = (observation_array[21] + observation_array[22]) / 2
            modified_observation[21] = thumb_flexion
            index_flexion = (observation_array[24] + observation_array[25] + observation_array[26]) / 3
            middle_flexion = (observation_array[28] + observation_array[29] + observation_array[30]) / 3
            ring_flexion = (observation_array[32] + observation_array[33] + observation_array[34]) / 3
            modified_observation[22] = index_flexion
            modified_observation[23] = middle_flexion
            modified_observation[24] = ring_flexion

            modified_observation = modified_observation[:25]
            modified_observation = np.delete(modified_observation, indices_to_delete)
            #print('reduced mode observation 수정 완료')
            # 참고: 이 로직은 최종적으로 (23,) shape을 반환합니다.
            return modified_observation

        elif mode == 'reduced_v2':
            indices_to_delete = [3, 4]
            modified_observation = observation_array.copy()

            modified_observation[19] = observation_array[19]
            modified_observation[20] = observation_array[20]
            thumb_flexion = (observation_array[21] + observation_array[22]) / 2
            modified_observation[21] = thumb_flexion
            index_flexion = (3*observation_array[24] + observation_array[25] + observation_array[26]) / 5
            middle_flexion = (3*observation_array[28] + observation_array[29] + observation_array[30]) / 5
            ring_flexion = (3*observation_array[32] + observation_array[33] + observation_array[34]) / 5
            modified_observation[22] = index_flexion
            modified_observation[23] = middle_flexion
            modified_observation[24] = ring_flexion

            modified_observation = modified_observation[:25]
            modified_observation = np.delete(modified_observation, indices_to_delete)
            #print('reduced mode observation 수정 완료')
            # 참고: 이 로직은 최종적으로 (23,) shape을 반환합니다.
            return modified_observation

        elif mode == 'full':
            indices_to_delete = [3, 4, 34]
            modified_observation = observation_array.copy()
            modified_observation = np.delete(modified_observation, indices_to_delete)
            #print('full mode observation 수정 완료')
            return modified_observation

        elif mode == 'reduced_v3':
            # 0~2: waist
            # 3~4: neck
            # 5~11: left arm
            # 12~18: right arm
            # 19~24: inspire hand
            # 25~40: kistar hand

            # 25~40 (16 DOF) -> 25~34 (10 DOF)

            indices_to_delete = [3, 4]
            modified_observation = observation_array.copy()

            modified_observation[25] = observation_array[25]
            modified_observation[26] = observation_array[26]
            modified_observation[27] = observation_array[27]
            modified_observation[28] = observation_array[28]
            # action_array[29] : Index Ab/Ad
            modified_observation[29] = observation_array[30]
            modified_observation[30] = (observation_array[31] + observation_array[32]) / 2

            # action_array[33] : Middle Ab/Ad
            modified_observation[31] = observation_array[34]
            modified_observation[32] = (observation_array[35] + observation_array[36]) / 2

            # action_array[31] : Ring Ab/Ad
            modified_observation[33] = (3*observation_array[38] + observation_array[39] + observation_array[40]) / 5

            modified_observation = modified_observation[:34]

            modified_observation = np.delete(modified_observation, indices_to_delete)
            #print('full mode action 수정 완료')
            return modified_observation


        elif mode == 'only_kistar':
            modified_observation = observation_array.copy()

            indices_to_delete = [3, 4, 11, 19, 20, 21, 22, 23, 24]

            modified_observation = np.delete(modified_observation, indices_to_delete)

            return modified_observation

            
    else:
        print(f"Warning: 예상치 못한 action shape {observation_array.shape}을 발견하여 수정하지 않았습니다.", file=sys.stderr)
        return observation_array
    
    # ---------------------------------------------------------
    
    return modified_observation

def main(directory_path):
    """
    지정된 디렉토리의 모든 .parquet 파일에 대해 action 및 observation 차원을 수정합니다.
    """
    p = Path(directory_path)
    if not p.is_dir():
        print(f"오류: '{directory_path}' 디렉토리를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    parquet_files = sorted(list(p.glob("*.parquet")))
    
    if not parquet_files:
        print(f"'{directory_path}'에서 .parquet 파일을 찾을 수 없습니다.", file=sys.stderr)
        return
        
    print(f"'{directory_path}'에서 {len(parquet_files)}개의 .parquet 파일을 처리합니다.")

    for file_path in tqdm(parquet_files, desc="Parquet 파일 처리 중"):
        try:
            df = pd.read_parquet(file_path)
            
            # 1. Action 컬럼 처리
            if 'action' in df.columns:
                df['action'] = df['action'].apply(process_action_dimension)
                # action 처리 후 shape 검증 (기존 코드에서는 32를 기대했지만 실제 로직은 23을 반환)
                # if not df.empty and df['action'].iloc[0].shape != (23,):
                #      print(f"경고: {file_path} 파일의 action 차원이 23이 아닙니다 (실제 shape: {df['action'].iloc[0].shape}). 덮어쓰기를 건너뜁니다.", file=sys.stderr)
                #      continue
            
            print('action 수정 완료')
            # 2. Observation 컬럼 처리 (새로 추가된 부분)
            if 'observation.state' in df.columns:
                df['observation.state'] = df['observation.state'].apply(process_observation_dimension)
                # 여기에 observation 처리 후 shape을 검증하는 로직을 추가할 수 있습니다.
                # 예: new_obs_shape = (80,)
                # if not df.empty and df['observation'].iloc[0].shape != new_obs_shape:
                #     print(f"경고: {file_path} 파일의 observation 차원이 {new_obs_shape}이 아닙니다. 덮어쓰기를 건너뜁니다.", file=sys.stderr)
                #     continue

            print('observation 수정 완료')
            # 수정된 데이터프레임으로 원본 파일 덮어쓰기
            df.to_parquet(file_path, index=False)
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생 {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        main(target_dir)
        print("\n모든 파일 처리가 완료되었습니다.")
    else:
        print("사용법: python modify_action_dimensions.py <parquet_files_directory_path>", file=sys.stderr)
