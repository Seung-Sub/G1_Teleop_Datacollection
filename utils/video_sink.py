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
        # 원본 코드처럼 복사본 보관
        self._left.append(img_left.copy())
        self._right.append(img_right.copy())
        self._realsense.append(img_realsense.copy() if img_realsense is not None else None)

        # 마스크 적용된 이미지 보관 (None일 수 있음)
        if img_left_masked is not None:
            self._left_masked.append(img_left_masked.copy())
        if img_right_masked is not None:
            self._right_masked.append(img_right_masked.copy())

    def close_episode(self) -> Dict[str, str]:
        assert self._task is not None and self._ep_idx is not None
        task_folder = os.path.join(BASE_FOLDER, self._task)
        chunk_id = self._ep_idx // CHUNK_SIZE
        video_base = os.path.join(task_folder, "videos", f"chunk-{chunk_id:03d}")

        left_dir = os.path.join(video_base, "observation.images.ego_left_view")
        right_dir = os.path.join(video_base, "observation.images.ego_right_view")
        realsense_dir = os.path.join(video_base, "observation.images.ego_realsense")
        left_masked_dir = os.path.join(video_base, "observation.images.ego_left_masked_view")
        right_masked_dir = os.path.join(video_base, "observation.images.ego_right_masked_view")

        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)
        os.makedirs(realsense_dir, exist_ok=True)
        os.makedirs(left_masked_dir, exist_ok=True)
        os.makedirs(right_masked_dir, exist_ok=True)

        # 프레임 전처리
        frames_left = process_frames(self.logger, self._left, side="왼쪽")
        frames_right = process_frames(self.logger, self._right, side="오른쪽")
        frames_rs = process_frames(self.logger, self._realsense, side="RealSense")

        # 마스크된 프레임 전처리 (있는 경우에만)
        frames_left_masked = process_frames(self.logger, self._left_masked, side="왼쪽 마스크") if self._left_masked else []
        frames_right_masked = process_frames(self.logger, self._right_masked, side="오른쪽 마스크") if self._right_masked else []

        video_name = f"episode_{self._ep_idx:06d}.mp4"

        left_path = os.path.join(left_dir, video_name)
        right_path = os.path.join(right_dir, video_name)
        rs_path = os.path.join(realsense_dir, video_name)
        left_masked_path = os.path.join(left_masked_dir, video_name)
        right_masked_path = os.path.join(right_masked_dir, video_name)

        # imageio FFMPEG writer
        def _write(path: str, frames: List[np.ndarray]):
            with imageio.get_writer(
                path, format="FFMPEG", mode="I", fps=self.fps,
                codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"]
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)

        _write(left_path, frames_left)
        self.logger.info(f"[VIDEO] Episode {self._ep_idx} 왼쪽 비디오 저장: {left_path}")

        _write(right_path, frames_right)
        self.logger.info(f"[VIDEO] Episode {self._ep_idx} 오른쪽 비디오 저장: {right_path}")

        _write(rs_path, frames_rs)
        self.logger.info(f"[VIDEO] Episode {self._ep_idx} RealSense 비디오 저장: {rs_path}")

        # 마스크된 비디오 저장 (프레임이 있는 경우에만)
        if frames_left_masked:
            _write(left_masked_path, frames_left_masked)
            self.logger.info(f"[VIDEO] Episode {self._ep_idx} 왼쪽 마스크 비디오 저장: {left_masked_path}")

        if frames_right_masked:
            _write(right_masked_path, frames_right_masked)
            self.logger.info(f"[VIDEO] Episode {self._ep_idx} 오른쪽 마스크 비디오 저장: {right_masked_path}")

        # 내부 버퍼 정리
        self._left = self._right = self._realsense = None
        self._left_masked = self._right_masked = None

