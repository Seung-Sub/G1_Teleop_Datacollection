# storage/video_sink.py
from __future__ import annotations
import os
from typing import List, Optional, Dict
import numpy as np
import imageio
from utils.record_config import BASE_FOLDER, CHUNK_SIZE
from utils.frame_utils import process_frames

class VideoSink:
    """
    좌/우/RealSense 프레임을 모아
    videos/chunk-XXX/observation.images.ego_*_view/episode_XXXXXX.mp4 로 저장
    """
    def __init__(self, logger, fps: float):
        self.logger = logger
        self.fps = float(fps)
        self._task: Optional[str] = None
        self._ep_idx: Optional[int] = None
        self._left: Optional[List[np.ndarray]] = None
        self._right: Optional[List[np.ndarray]] = None
        self._realsense: Optional[List[np.ndarray]] = None
        self._left_masked: Optional[List[np.ndarray]] = None
        self._right_masked: Optional[List[np.ndarray]] = None

    def start_episode(self, task_name: str, ep_idx: int):
        self._task = task_name
        self._ep_idx = ep_idx
        self._left, self._right, self._realsense = [], [], []
        self._left_masked, self._right_masked = [], []
        self.logger.info(f"[VIDEO] start episode={ep_idx} (task={task_name})")

    def append(self, img_left: np.ndarray, img_right: np.ndarray, img_realsense: np.ndarray,
               img_left_masked: np.ndarray = None, img_right_masked: np.ndarray = None):
        # None 인 view 는 그 episode 동안 비활성 (close_episode 가 빈 list 를 skip 해 mp4 생성 안 함).
        if img_left      is not None: self._left.append(img_left.copy())
        if img_right     is not None: self._right.append(img_right.copy())
        if img_realsense is not None: self._realsense.append(img_realsense.copy())
        if img_left_masked  is not None: self._left_masked.append(img_left_masked.copy())
        if img_right_masked is not None: self._right_masked.append(img_right_masked.copy())

    def close_episode(self) -> Dict[str, str]:
        assert self._task is not None and self._ep_idx is not None
        task_folder = os.path.join(BASE_FOLDER, self._task)
        chunk_id = self._ep_idx // CHUNK_SIZE
        video_base = os.path.join(task_folder, "videos", f"chunk-{chunk_id:03d}")
        video_name = f"episode_{self._ep_idx:06d}.mp4"

        # imageio FFMPEG writer; skip silently when there is nothing to write.
        def _write_view(view_name: str, frames_buf: Optional[List], side_label: str):
            if not frames_buf:
                return
            frames = process_frames(self.logger, frames_buf, side=side_label)
            if not frames:
                return
            view_dir  = os.path.join(video_base, f"observation.images.{view_name}")
            os.makedirs(view_dir, exist_ok=True)
            view_path = os.path.join(view_dir, video_name)
            with imageio.get_writer(
                view_path, format="FFMPEG", mode="I", fps=self.fps,
                codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"],
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            self.logger.info(f"[VIDEO] Episode {self._ep_idx} {side_label} 저장: {view_path}")

        _write_view("ego_left_view",         self._left,         "왼쪽")
        _write_view("ego_right_view",        self._right,        "오른쪽")
        _write_view("ego_realsense",         self._realsense,    "RealSense")
        _write_view("ego_left_masked_view",  self._left_masked,  "왼쪽 마스크")
        _write_view("ego_right_masked_view", self._right_masked, "오른쪽 마스크")

        # 내부 버퍼 정리
        self._left = self._right = self._realsense = None
        self._left_masked = self._right_masked = None

