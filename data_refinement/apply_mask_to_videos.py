import cv2
import cv2.aruco as aruco
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def _create_workspace_mask(image, corners, ids):
    """
    관심 영역 바깥을 검정색으로 마스킹합니다.
    """
    height, width = image.shape[:2]

    # 기본값: 전체를 마스크된 상태로 시작
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    if ids is not None and len(ids) > 0:
        detected_markers = {marker_id[0]: corner_set[0] for marker_id, corner_set in zip(ids, corners)}

        # 모든 마커 감지 시
        if all(i in detected_markers for i in range(4)):
            # 마커 ID 순서대로 점들 재정렬 (0,1,2,3 순서)
            points = np.array([
                detected_markers[0][0], # 0번 마커의 첫번째 코너
                detected_markers[1][0], # 1번 마커의 첫번째 코너
                detected_markers[2][0], # 2번 마커의 첫번째 코너
                detected_markers[3][0], # 3번 마커의 첫번째 코너
            ], dtype=np.float32)

            # 0-1 직선, 1-2 반직선, 0-3 반직선으로 영역 생성
            p1, p2 = points[1], points[2]
            direction_12 = (p2 - p1)
            extended_p2 = p1 + direction_12 * 1000

            p0, p3 = points[0], points[3]
            direction_03 = (p3 - p0)
            extended_p3 = p0 + direction_03 * 1000

            region_points = np.array([
                points[0],
                points[1],
                extended_p2,
                extended_p3
            ], dtype=np.int32)
            
            # 마스크 적용: 관심 영역 내부는 흰색 (통과), 바깥은 검정색 (차단)
            cv2.fillPoly(mask, [region_points], 0)

    return 255 - mask

def process_video(video_path: Path, output_dir: Path):
    """
    단일 비디오 파일을 처리하여 마스크를 씌운 새 비디오를 저장합니다.
    """
    print(f"  - 처리 시작: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [오류] 비디오 파일을 열 수 없습니다: {video_path.name}")
        return

    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 비디오 설정
    output_path = output_dir / video_path.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # ArUco 설정
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    previous_mask = None
    
    # 프레임 처리
    for _ in tqdm(range(frame_count), desc=f"    - {video_path.name}", unit="frame", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        # 4개의 마커가 모두 감지되면, 새로운 마스크를 생성하여 저장합니다.
        if ids is not None and all(i in np.ravel(ids) for i in range(4)):
            mask = _create_workspace_mask(frame, corners, ids)
            previous_mask = mask
        else:
            # 마커가 모두 감지되지 않으면 이전 마스크를 사용합니다.
            mask = previous_mask

        # 마스크가 있으면 적용하고, 없으면 검은색 프레임을 출력합니다.
        if mask is not None:
            final_frame = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            final_frame = np.zeros_like(frame)

        out.write(final_frame)

    cap.release()
    out.release()
    print(f"  - 처리 완료 및 저장: {output_path}")

def main(root_dir: Path):
    """
    지정된 루트 디렉토리에서 비디오를 찾아 마스킹 작업을 수행합니다.
    """
    print(f"데이터셋 루트 디렉토리 처리 시작: {root_dir}")
    videos_dir = root_dir / "videos/chunk-000"
    if not videos_dir.is_dir():
        print(f"[오류] 'videos/chunk-000' 폴더를 찾을 수 없습니다: {videos_dir}")
        return

    view_dirs = [
        "observation.images.ego_left_view",
        "observation.images.ego_right_view"
    ]

    for view in view_dirs:
        input_dir = videos_dir / view
        output_dir = videos_dir / f"{view}_masked"
        
        if not input_dir.is_dir():
            print(f"[경고] 입력 뷰 폴더를 찾을 수 없습니다. 건너뛰었습니다: {input_dir}")
            continue

        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n>> 뷰 폴더 처리 중: {input_dir}")
        print(f"   출력은 다음 위치에 저장됩니다: {output_dir}")

        video_files = sorted(list(input_dir.glob("*.mp4")))
        if not video_files:
            print(f"   mp4 파일을 찾을 수 없습니다.")
            continue

        for video_path in video_files:
            process_video(video_path, output_dir)

    print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED 카메라 영상에 ArUco 마커 기반 마스크를 적용합니다.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="데이터셋의 루트 경로 (예: 'record/1016_mask/1016_pick_and_place_apple_masked')"
    )
    args = parser.parse_args()
    
    root_path = Path(args.path)
    if not root_path.is_dir():
        print(f"[에러] 제공된 경로가 디렉토리가 아니거나 존재하지 않습니다: {root_path}")
    else:
        main(root_path)
