import pandas as pd
import numpy as np
import sys

def analyze_parquet_columns(file_path):
    """지정된 Parquet 파일의 모든 컬럼을 분석하고 각 컬럼의 정보를 출력합니다."""
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Parquet 파일을 읽는 중 오류가 발생했습니다: {e}", file=sys.stderr)
        return

    # 컬럼 필터링 로직을 제거하고 모든 컬럼을 대상으로 합니다.
    all_columns = df.columns

    if all_columns.empty:
        print("파일에 컬럼이 존재하지 않습니다.")
    else:
        print(f"'{file_path}' 파일에서 {len(all_columns)}개의 컬럼을 찾았습니다:\n")
        for col in all_columns:
            # 데이터가 비어있는 경우를 대비해 유효한 첫 번째 행을 찾습니다.
            try:
                # dropna()를 사용하여 null이 아닌 첫 번째 데이터를 가져옵니다.
                sample_data = df[col].dropna().iloc[5]
            except IndexError:
                # 컬럼의 모든 데이터가 null인 경우
                print(f"- 컬럼명: '{col}'")
                print("  - 유효한 데이터가 없습니다 (모두 null).")
                print("-" * 30)
                continue

            dimension = "N/A"
            dtype = "N/A"
            
            # 데이터가 리스트나 ndarray와 같은 배열 형태일 경우를 대비합니다.
            try:
                sample_data_np = np.array(sample_data)
                dimension = sample_data_np.shape
                dtype = sample_data_np.dtype
            except Exception:
                 # 배열로 변환할 수 없는 스칼라 값(int, float, str 등)인 경우
                 pass

            print(f"- 컬럼명: '{col}'")
            print(f"  - 첫 데이터의 타입: {type(sample_data)}")
            print(f"  - 첫 데이터: {sample_data}")
            
            # 차원 정보가 유효할 경우에만 출력합니다.
            if dimension != "N/A" and dimension != (): # 스칼라 값의 shape는 () 이므로 제외
                print(f"  - 데이터의 차원 (shape): {dimension}")
                print(f"  - 데이터 타입 (dtype): {dtype}")
            print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_to_analyze = sys.argv[1]
        analyze_parquet_columns(file_to_analyze)
    else:
        print("사용법: python inspect_parquet.py <parquet_file_path>", file=sys.stderr)