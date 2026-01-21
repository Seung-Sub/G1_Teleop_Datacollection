import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import numpy as np

# --- 플롯할 인덱스를 여기에 직접 입력하세요 ---
INDICES_TO_PLOT = [-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1] # 예시: action의 17번부터 22번 인덱스
# INDICES_TO_PLOT = [-22, -21, -20, -19, -18, -17] # 예시: action의 17번부터 22번 인덱스
# -----------------------------------------

def plot_action_from_parquet(file_path: Path):
    """
    지정된 Parquet 파일에서 'action' 컬럼의 특정 인덱스 값들을 플롯합니다.
    플롯할 인덱스는 코드 상단의 INDICES_TO_PLOT 리스트에 지정됩니다.

    Args:
        file_path (Path): Parquet 파일 경로.
    """
    if not file_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"오류: Parquet 파일을 읽는 중 문제가 발생했습니다 - {e}")
        return

    if 'action' not in df.columns:
        print(f"오류: 'action' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")
        return

    # 'frame_index'가 있으면 x축으로 사용, 없으면 단순 행 번호 사용
    if 'frame_index' in df.columns:
        x_axis = df['frame_index']
        x_label = "Frame Index"
    else:
        x_axis = range(len(df))
        x_label = "Row Index"

    plt.figure(figsize=(15, 8))

    print("action: ",df['action'][200])
    print("obs: ",df['observation.state'][200])

    for index in INDICES_TO_PLOT:
        try:            
            # 'action' 컬럼에서 지정된 인덱스의 데이터를 추출합니다.
            action_data = df['action'].apply(lambda x: x[index])
            line, = plt.plot(x_axis, action_data, label=f'Action[{index}]')
            current_color = line.get_color()

            # 3. 'observation' 데이터 추출 및 동일한 색상 적용
            observation_data = df['observation.state'].apply(lambda x: x[index])
            plt.plot(x_axis, observation_data, label=f'Observation[{index}]', 
                    linestyle='--', color=current_color) # 같은 색상 적용

        except IndexError:
            action_dim = len(df['action'].iloc[0]) if len(df) > 0 and df['action'].iloc[0] is not None else 'N/A'
            print(f"경고: 인덱스 {index}가 범위를 벗어났습니다. 'action'의 차원은 {action_dim} 입니다. 이 인덱스는 건너뜁니다.")
        except Exception as e:
            print(f"경고: 인덱스 {index}를 플롯하는 중 문제가 발생했습니다 - {e}")

    plt.title(f"Action Indices {INDICES_TO_PLOT} from\n{file_path.name}")
    plt.xlabel(x_label)
    plt.ylabel("Action Value")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot specific indices of the 'action' column from a Parquet file based on a hardcoded list.")
    parser.add_argument("file_path", type=str, help="Path to the Parquet file.")
    
    args = parser.parse_args()
    
    plot_action_from_parquet(Path(args.file_path))

# 사용 예시:
# python plot_parquet.py /path/to/your/episode_000000.parquet
