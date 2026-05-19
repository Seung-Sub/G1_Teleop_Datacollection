# storage/parquet_sink.py
from __future__ import annotations
import os
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from utils.record_config import BASE_FOLDER, CHUNK_SIZE

class ParquetSink:
    """
    에피소드 단위로 observation.state / action / timestamp를 모아
    data/chunk-XXX/episode_XXXXXX.parquet 로 저장.

    Phase D 이후: 추가로 raw_ts_* 등의 per-frame 메타 컬럼을 add_extra_column
    으로 등록하면 close_episode 시 parquet 에 함께 저장된다 (학습/디버그에서
    멀티모달 phase offset 분석 용도).
    """
    def __init__(self, logger):
        self.logger = logger
        self._states_action: Optional[List[List[float]]] = None
        self._states_sensor: Optional[List[List[float]]] = None
        self._actions: Optional[List[List[float]]] = None
        self._timestamps: Optional[List[float]] = None
        self._extra: Dict[str, np.ndarray] = {}
        self._task: Optional[str] = None
        self._ep_idx: Optional[int] = None

    def start_episode(self, task_name: str, ep_idx: int):
        self._task = task_name
        self._ep_idx = ep_idx
        self._states_action, self._states_sensor, self._actions, self._timestamps = [], [], [], []
        self._extra = {}
        self.logger.info(f"[PARQUET] start episode={ep_idx} (task={task_name})")

    def append(self, state_vec_action: np.ndarray, state_vec_sensor: np.ndarray, action_vec: np.ndarray, t_sec: float):
        # float32 + list 변환 (원본 코드 호환)
        self._states_action.append(state_vec_action.astype(np.float32).tolist())
        self._states_sensor.append(state_vec_sensor.astype(np.float32).tolist())
        self._actions.append(action_vec.astype(np.float32).tolist())
        self._timestamps.append(float(t_sec))

    def add_extra_column(self, name: str, arr: np.ndarray) -> None:
        """Register a per-frame metadata column for the upcoming close_episode.
        len(arr) must equal the number of appended frames at close time (validated)."""
        self._extra[name] = np.asarray(arr)

    def close_episode(self) -> str:
        assert self._task is not None and self._ep_idx is not None
        task_folder = os.path.join(BASE_FOLDER, self._task)
        chunk_id = self._ep_idx // CHUNK_SIZE
        data_chunk_folder = os.path.join(task_folder, "data", f"chunk-{chunk_id:03d}")
        os.makedirs(data_chunk_folder, exist_ok=True)

        n_rows = len(self._states_action)
        df_dict = {
            "observation.state": self._states_action,
            "observation.sensor": self._states_sensor,
            "action": self._actions,
            "timestamp": self._timestamps,
            "episode_index": self._ep_idx,
            "index": list(range(n_rows)),
            "frame_index": list(range(n_rows)),
        }
        for name, arr in self._extra.items():
            if len(arr) != n_rows:
                self.logger.warning(
                    f"[PARQUET] extra column '{name}' length {len(arr)} != frames {n_rows}; dropping."
                )
                continue
            # 1-D scalar column 으로 저장. dtype 그대로 (int64 ts 등).
            df_dict[name] = list(arr)
        df = pd.DataFrame(df_dict)

        parquet_name = f"episode_{self._ep_idx:06d}.parquet"
        parquet_path = os.path.join(data_chunk_folder, parquet_name)
        # pyarrow/snappy는 pyarrow 엔진 필요
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        self.logger.info(f"[PARQUET] Episode {self._ep_idx} saved to {parquet_path}")

        # 내부 상태 정리
        self._states_action = self._states_sensor = self._actions = self._timestamps = None
        self._extra = {}