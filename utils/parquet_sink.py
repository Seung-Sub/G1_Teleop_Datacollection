# storage/parquet_sink.py
from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
import pandas as pd
from utils.record_config import BASE_FOLDER, CHUNK_SIZE

class ParquetSink:
    """
    에피소드 단위로 observation.state / action / timestamp를 모아
    data/chunk-XXX/episode_XXXXXX.parquet 로 저장
    """
    def __init__(self, logger):
        self.logger = logger
        self._states_action: Optional[List[List[float]]] = None
        self._states_sensor: Optional[List[List[float]]] = None
        self._actions: Optional[List[List[float]]] = None
        self._timestamps: Optional[List[float]] = None
        self._task: Optional[str] = None
        self._ep_idx: Optional[int] = None

    def start_episode(self, task_name: str, ep_idx: int):
        self._task = task_name
        self._ep_idx = ep_idx
        self._states_action, self._states_sensor, self._actions, self._timestamps = [], [], [], []
        self.logger.info(f"[PARQUET] start episode={ep_idx} (task={task_name})")

    def append(self, state_vec_action: np.ndarray, state_vec_sensor: np.ndarray, action_vec: np.ndarray, t_sec: float):
        # float32 + list 변환 (원본 코드 호환)
        self._states_action.append(state_vec_action.astype(np.float32).tolist())
        self._states_sensor.append(state_vec_sensor.astype(np.float32).tolist())
        self._actions.append(action_vec.astype(np.float32).tolist())
        self._timestamps.append(float(t_sec))

    def close_episode(self) -> str:
        assert self._task is not None and self._ep_idx is not None
        task_folder = os.path.join(BASE_FOLDER, self._task)
        chunk_id = self._ep_idx // CHUNK_SIZE
        data_chunk_folder = os.path.join(task_folder, "data", f"chunk-{chunk_id:03d}")
        os.makedirs(data_chunk_folder, exist_ok=True)

        df = pd.DataFrame({
            "observation.state": self._states_action,
            "observation.sensor": self._states_sensor,
            "action": self._actions,
            "timestamp": self._timestamps,
            "episode_index": self._ep_idx,
            "index": list(range(len(self._states_action))),
            "frame_index": list(range(len(self._states_action))),
        })

        parquet_name = f"episode_{self._ep_idx:06d}.parquet"
        parquet_path = os.path.join(data_chunk_folder, parquet_name)
        # pyarrow/snappy는 pyarrow 엔진 필요
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        self.logger.info(f"[PARQUET] Episode {self._ep_idx} saved to {parquet_path}")

        # 내부 상태 정리
        self._states_action = self._states_sensor = self._actions = self._timestamps = None