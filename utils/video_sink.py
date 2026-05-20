# storage/video_sink.py
from __future__ import annotations
import os
from typing import List, Optional, Dict
import numpy as np
import imageio
from utils.record_config import BASE_FOLDER, CHUNK_SIZE
from utils.frame_utils import process_frames

class VideoSink:
    """View-dynamic mp4 writer.

    `videos/chunk-XXX/<view_name>/episode_XXXXXX.mp4`

    Phase K7-A 이후: start_episode(views=[...]) 로 active view 명을 받아 동적으로
    buffer 를 생성. append_views(dict) 가 한 axis tick 의 view 별 frame 을 push.
    backward-compat: 옛 append(left, right, realsense, ...) 시그니처는 그대로 유지.
    """
    def __init__(self, logger, fps: float):
        self.logger = logger
        self.fps = float(fps)
        self._task: Optional[str] = None
        self._ep_idx: Optional[int] = None
        # 동적 view buffers: view_name → list of frames. None 인 frame 은 skip.
        self._buffers: Dict[str, List[np.ndarray]] = {}

    def start_episode(self, task_name: str, ep_idx: int, views: Optional[List[str]] = None):
        """views: ['observation.images.ego', ...] active view 이름 리스트.
        backward-compat: views=None 이면 옛 5-view (left/right/realsense/_masked) 모두 준비."""
        self._task = task_name
        self._ep_idx = ep_idx
        self._buffers = {}
        if views is None:
            views = [
                'observation.images.ego_left_view',
                'observation.images.ego_right_view',
                'observation.images.ego_realsense',
                'observation.images.ego_left_masked_view',
                'observation.images.ego_right_masked_view',
            ]
        for v in views:
            self._buffers[v] = []
        self.logger.info(f"[VIDEO] start episode={ep_idx} (task={task_name}) views={views}")

    def append_views(self, view_payload: Dict[str, Optional[np.ndarray]]):
        """한 axis tick 의 view 별 frame 을 push. None 인 view 는 skip.
        새 view name 이 들어오면 자동 등록."""
        for v, frame in view_payload.items():
            if frame is None:
                continue
            self._buffers.setdefault(v, []).append(frame.copy())

    # ----- backward-compat: 5-view 고정 append -----------------------------
    def append(self, img_left: np.ndarray, img_right: np.ndarray, img_realsense: np.ndarray,
               img_left_masked: np.ndarray = None, img_right_masked: np.ndarray = None):
        payload = {}
        if img_left      is not None: payload['observation.images.ego_left_view']         = img_left
        if img_right     is not None: payload['observation.images.ego_right_view']        = img_right
        if img_realsense is not None: payload['observation.images.ego_realsense']         = img_realsense
        if img_left_masked  is not None: payload['observation.images.ego_left_masked_view']  = img_left_masked
        if img_right_masked is not None: payload['observation.images.ego_right_masked_view'] = img_right_masked
        self.append_views(payload)

    def close_episode(self) -> Dict[str, str]:
        assert self._task is not None and self._ep_idx is not None
        task_folder = os.path.join(BASE_FOLDER, self._task)
        chunk_id = self._ep_idx // CHUNK_SIZE
        video_base = os.path.join(task_folder, "videos", f"chunk-{chunk_id:03d}")
        video_name = f"episode_{self._ep_idx:06d}.mp4"

        saved: Dict[str, str] = {}

        def _write_view(view_full_name: str, frames_buf: Optional[List]):
            if not frames_buf:
                return
            frames = process_frames(self.logger, frames_buf, side=view_full_name)
            if not frames:
                return
            view_dir  = os.path.join(video_base, view_full_name)
            os.makedirs(view_dir, exist_ok=True)
            view_path = os.path.join(view_dir, video_name)
            with imageio.get_writer(
                view_path, format="FFMPEG", mode="I", fps=self.fps,
                codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"],
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            self.logger.info(f"[VIDEO] Episode {self._ep_idx} {view_full_name} saved: {view_path}")
            saved[view_full_name] = view_path

        for view_full_name, frames_buf in self._buffers.items():
            _write_view(view_full_name, frames_buf)

        # 내부 버퍼 정리
        self._buffers = {}
        return saved
