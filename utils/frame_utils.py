# utils/frame_utils.py
import numpy as np
import cv2

def process_frames(logger, frames, side: str = ""):
    """
    1) 검증: ndarray, HxWxC, C==3 여부 확인. 잘못된 프레임 제거
    2) 전처리: 1채널/2차원→3채널, 4채널→RGB
    3) BGR->RGB 변환, uint8 캐스팅
    """
    bad_idxs = []
    for i, f in enumerate(frames):
        if not (isinstance(f, np.ndarray) and f.ndim == 3 and f.shape[2] == 3):
            bad_idxs.append((i, getattr(f, "shape", None)))
    if bad_idxs:
        logger.error(f"[VIDEO] {side} 프레임 잘못됨: 인덱스/shape = {bad_idxs}")
        frames = [f for idx, f in enumerate(frames)
                  if (idx, getattr(f, "shape", None)) not in bad_idxs]
        logger.info(f"[VIDEO] {side} 잘못된 프레임 삭제 완료")

    processed = []
    for f in frames:
        if f.ndim == 2:
            img = np.stack([f]*3, axis=-1)
        elif f.ndim == 3 and f.shape[2] == 1:
            img = np.repeat(f, 3, axis=2)
        elif f.ndim == 3 and f.shape[2] == 4:
            img = f[..., :3]
        elif f.ndim == 3 and f.shape[2] == 3:
            img = f
        else:
            logger.error(f"[VIDEO] {side} 지원하지 않는 프레임 shape: {f.shape}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed.append(img.astype(np.uint8))
    return processed