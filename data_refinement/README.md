# 데이터 정제 스크립트 모음 (Data Refinement Scripts)

이 디렉토리에는 로봇 원격 조종으로 기록된 데이터셋을 후처리하고 정제하는 데 사용되는 파이썬 스크립트들이 포함되어 있습니다. 주요 기능은 데이터셋 병합, 마스크 적용, 데이터 차원 수정 등입니다.

---

### 주요 스크립트 및 사용법

#### 1. `apply_mask_to_videos.py`

ArUco 마커를 기반으로 데이터셋의 비디오에 워크스페이스 마스크를 적용합니다. 이 스크립트는 지정된 데이터셋 폴더 내의 `ego_left_view` 및 `ego_right_view` 비디오를 찾아 마스크가 적용된 새로운 비디오(`_masked` 접미사 폴더에 저장)를 생성합니다.

**사용법:**
```bash
python apply_mask_to_videos.py --path <데이터셋_루트_경로>
```
**예시:**
```bash
python apply_mask_to_videos.py --path /home/ansur/Ansur_unitree_teleop/record/1125_pick_and_place
```

---

#### 2. `sequential_merge.py`

여러 개의 데이터셋 폴더를 순차적으로 병합하여 하나의 새로운 데이터셋으로 만듭니다. 스크립트 내부에 병합할 데이터셋 목록을 지정하면, 모든 에피소드를 수집하여 0번부터 순차적으로 번호를 다시 매겨 새로운 폴더에 저장합니다. 원본 데이터셋은 유지됩니다.

**사용법:**
1.  스크립트 파일(`sequential_merge.py`)을 열어 `dataset_names` 리스트에 병합할 데이터셋 폴더 이름을 순서대로 입력합니다.
2.  스크립트를 실행합니다.

```bash
python sequential_merge.py
```

---

#### 3. `modify_action_dimension.py` & `modify_kistar_dim.py`

Parquet 파일 내의 `action` 및 `observation.state` 데이터의 차원을 수정합니다. 특정 관절 데이터를 제거하거나 여러 관절의 평균값을 계산하여 차원을 축소하는 로직이 포함되어 있습니다.

**사용법:**
```bash
python modify_action_dimension.py <parquet_파일들이_있는_디렉토리_경로>
```
**예시:**
```bash
python modify_action_dimension.py /home/ansur/Ansur_unitree_teleop/record/my_dataset/data/chunk-000/
```

---

#### 4. `merge_datasets.py`

두 개의 데이터셋(`SOURCE`와 `TARGET`)을 병합합니다. `TARGET` 데이터셋의 에피소드들을 `SOURCE` 데이터셋에 추가하고, 모든 에피소드의 번호를 0번부터 다시 매겨 `SOURCE` 폴더를 덮어씁니다. 실행 전에 원본 `SOURCE` 데이터는 자동으로 백업됩니다.

**사용법:**
1.  스크립트 파일(`merge_datasets.py`)을 열어 `SOURCE_PATH`와 `TARGET_PATH` 변수에 병합할 데이터셋 경로를 지정합니다.
2.  스크립트를 실행합니다.

```bash
python merge_datasets.py
```

