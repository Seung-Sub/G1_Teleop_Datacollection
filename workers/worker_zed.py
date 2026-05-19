import os
import time

import numpy as np
import cv2
import pyzed.sl as sl
# [추가] Path 모듈: 디버그 프레임 저장 경로 생성용
from pathlib import Path
# [추가] matplotlib.pyplot: 실시간 깊이 맵 플롯용
import matplotlib.pyplot as plt

from sharedmemory.shmManager import SharedMemoryManager
# [변경] 공유 메모리 스키마 추가: ARUCO_MARKERS(ArUco 마커 정보), WORKSPACE_MASK(작업 공간 마스크),
#        MASK_CONTROL_LAYOUT(마스크 생성 제어), DEPTH_MAP(깊이 맵)
from sharedmemory.shm_schema import CAMERA, ARUCO_MARKERS, WORKSPACE_MASK, WORKER_FREQ, MASK_CONTROL_LAYOUT, DEPTH_MAP

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# [추가] 마스크 갱신 주기 상수: 마스크를 주기적으로 업데이트하는 시간 간격 (초)
MASK_UPDATE_INTERVAL = 0.02

# [추가] 마스크 생성 토글: 마스크 생성 여부를 제어 (실제 적용은 각 프로세스에서 결정)
GENERATE_MASK = True

# [추가] 실시간 깊이 맵 플롯 관련 상수: 디버깅/시각화용
# --- 실시간 뎁스맵 플롯 설정 ---
PLOT_DEPTH_MAP = False # 실시간 플롯 활성화
PLOT_INTERVAL = 0.1     # 0.1초 (10 FPS) 간격으로 플롯 업데이트
# -----------------------------

# [추가] 작업 공간 마스크 생성 함수: ArUco 마커 4개를 감지하여 작업 공간 영역 마스크 생성
# 원본 코드에는 없었던 기능으로, ArUco 마커로 정의된 영역만 통과하는 마스크 생성
def _create_workspace_mask(image, corners, ids):
    """특정 카메라 이미지에 대한 작업 공간 마스크 생성"""
    height, width = image.shape[:2]

    # 기본값: 빈 마스크
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_contour = np.zeros(8, dtype=np.float64)  # 4 points * 2 coords
    marker_corners_flat = np.zeros(8, dtype=np.float64)  # 4 points * 2 coords

    if ids is not None and len(ids) > 0:
        num_markers = min(len(ids), 4)

        # 4개 마커 모두 감지되었을 때 박스 테두리 좌표 저장
        box_corners = []

        for i in range(num_markers):
            marker_id = ids[i][0]
            corner_points = corners[i][0].astype(int)
            first_corner = corner_points[0]
            box_corners.append(first_corner)

        # 4개 마커 모두 감지되었을 때 마스크 생성
        if num_markers == 4 and len(box_corners) == 4:
            # 마커 ID 순서대로 점들 재정렬 (0,1,2,3 순서)
            sorted_corners = []
            for marker_idx in range(4):
                for i, marker_id in enumerate(ids[:num_markers]):
                    if marker_id[0] == marker_idx:
                        sorted_corners.append(box_corners[i])
                        break

            if len(sorted_corners) >= 4:
                points = np.array(sorted_corners, dtype=np.float32)

                # 0-1 직선, 1-2 반직선, 0-3 반직선으로 영역 생성
                # 1-2 반직선 방향으로 이미지 경계까지 연장
                p1, p2 = points[1], points[2]  # 마커 1과 2
                direction_12 = p2 - p1
                extended_p2 = p1 + direction_12 * 1000

                # 0-3 반직선 방향으로 이미지 경계까지 연장
                p0, p3 = points[0], points[3]  # 마커 0과 3
                direction_03 = p3 - p0
                extended_p3 = p0 + direction_03 * 1000

                # 영역을 정의하는 점들: 0, 1, extended_2, extended_3
                region_points = np.array([
                    points[0],  # 마커 0
                    points[1],  # 마커 1
                    extended_p2,  # 1-2 반직선의 연장점
                    extended_p3   # 0-3 반직선의 연장점
                ], dtype=np.int32)

                # 마스크 생성 (흑백 마스크: 255=작업공간, 0=배경)
                cv2.fillPoly(mask, [region_points], 255)

                # 결과 준비
                mask_contour = region_points.astype(np.float64).flatten()
                marker_corners_flat = np.array(sorted_corners, dtype=np.float64).flatten()

    return mask, mask_contour, marker_corners_flat

def worker_zed(shared_event, shm_name, shared_lock, serial=None, zed_mode='direct'):
    """ZED stereo worker.

    Args:
        serial: ZED device serial (int 또는 str). None 이면 첫 번째 device.
        zed_mode: 'direct' (sl.Camera.open USB direct) | 'stream' (set_from_stream).
                  stream 모드는 Jetson 등 외부 PC 가 ZED 를 들고 송신 중일 때만.
    """
    # 3) SharedMemoryManager 초기화 (필요하다면 사용)
    camera_shm = SharedMemoryManager(CAMERA, shared_lock["camera_lock"], shm_name["camera_shm"])
    # [추가] ArUco 마커 정보 공유 메모리: 감지된 마커의 ID, 코너 좌표, 중심점 등 저장
    aruco_shm = SharedMemoryManager(ARUCO_MARKERS, shared_lock["aruco_lock"], shm_name["aruco_shm"])
    # [추가] 작업 공간 마스크 공유 메모리: 생성된 마스크 데이터를 다른 워커와 공유
    workspace_mask_shm = SharedMemoryManager(WORKSPACE_MASK, shared_lock["workspace_mask_lock"], shm_name["workspace_mask_shm"])
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])
    # [추가] 마스크 제어 공유 메모리: UI에서 마스크 생성 ON/OFF 제어용
    mask_control_shm = SharedMemoryManager(MASK_CONTROL_LAYOUT, shared_lock["record_lock"], shm_name["mask_control_shm"])
    # [추가] 깊이 맵 공유 메모리: ZED 카메라로부터 받은 깊이 정보 저장
    depth_map_shm = SharedMemoryManager(DEPTH_MAP, shared_lock["depth_map_lock"], shm_name["depth_map_shm"])


    # ZED 초기화 — mode 에 따라 direct (USB) 또는 stream 분기
    zed = sl.Camera()
    init_params = sl.InitParameters()
    if zed_mode == 'stream':
        # 외부 송신자 PC 가 set_from_stream 으로 ZED 영상 송출 중인 경우.
        init_params.set_from_stream("192.168.5.11", 30000)
        logger_mp.info(f"[ZED] stream mode: receiving from 192.168.5.11:30000")
    else:
        # direct USB 연결. serial 명시 시 해당 device, 아니면 첫 device.
        if serial is not None:
            try:
                init_params.set_from_serial_number(int(serial))
                logger_mp.info(f"[ZED] direct mode: open serial={serial}")
            except Exception as e:
                logger_mp.warning(f"[ZED] set_from_serial_number({serial}) 실패: {e} — fallback any device")
        else:
            logger_mp.info("[ZED] direct mode: open first available device")
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.sdk_verbose = 1  # 디버깅용
    init_params.coordinate_units = sl.UNIT.METER


    # logger_mp 설정 직후에
    # console = logging_mp.StreamHandler()
    # console.setLevel(logging_mp.INFO)
    # logger_mp.addHandler(console)


    logger_mp.info("[ZED] About to open ZED camera...")
    err = zed.open(init_params)
    logger_mp.info(f"[ZED] zed.open() returned: {err}")
    if err != sl.ERROR_CODE.SUCCESS:
        logger_mp.error("[ZED] ZED open failed")
        shared_event['emergency'].set()
        return
    logger_mp.info("[ZED] ZED camera opened.")  # 이제 확실히 찍힐 겁니다.

    runtime_params = sl.RuntimeParameters()
    left_mat  = sl.Mat()
    right_mat = sl.Mat()
    # [추가] 깊이 맵 저장용 Mat 객체: 원본에는 없었음
    depth_mat = sl.Mat()

    # [추가] Matplotlib 실시간 플롯 초기화: 깊이 맵을 실시간으로 시각화 (디버깅용)
    # --- Matplotlib 실시간 플롯 초기화 ---
    fig, ax = None, None
    if PLOT_DEPTH_MAP:
        plt.ion()  # 대화형 모드 활성화
        fig, ax = plt.subplots()
        ax.set_title("Real-time Depth Map")
        print("[ZED-DEBUG] Real-time depth map plotting enabled.")
    # -----------------------------------

    # [추가] 디버그 프레임 저장 폴더 생성: 처리된 이미지나 마스크를 파일로 저장할 경로
    # --- 디버그 프레임 저장 폴더 생성 ---
    debug_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)
    # -----------------------------------

    try:
        missed_count = 0
        max_missed = 5
        last_time = time.perf_counter_ns()  # 최초 타임스탬프

        # 마스크 제어 상태 변수들
        saved_mask_l = None
        saved_mask_r = None
        saved_mask_contour_l = None
        saved_mask_contour_r = None
        saved_marker_corners_l = None
        saved_marker_corners_r = None
        prev_mask_control_enabled = False
        last_mask_update_time = 0.0
        last_plot_time = 0.0


        while not shared_event['shutdown'].is_set() and not shared_event['emergency'].is_set():

            now = time.perf_counter_ns()
            delta_ns = now - last_time
            actual_hz = 1.0 / (delta_ns / 1e9) if delta_ns > 0 else 0.0

            # freq_data = freq_shm.read_data()
            # freq_data["camera_freq"] = actual_hz
            freq_shm.write_data(camera_freq = actual_hz)
            
            # logger_mp.warning(f"[Camera60Hz] actual_hz : {actual_hz}")

            last_time = now


            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # ← 이 두 줄이 반드시 필요합니다!
                zed.retrieve_image(left_mat,  sl.VIEW.LEFT)
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

                # [추가] 깊이 맵 데이터 수집: 원본에는 없었던 기능
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

                # 그 다음에야 get_data() 로 안전하게 읽어서 처리
                bgra_l  = left_mat.get_data()

                bgr_l   = cv2.cvtColor(bgra_l, cv2.COLOR_BGRA2BGR)
                small_l = cv2.resize(bgr_l, (640,480), interpolation=cv2.INTER_AREA)
                small_l = np.ascontiguousarray(small_l, dtype=np.uint8)
                # [추가] 원본 이미지 복사본 저장: 마스킹 전 원본을 보존하기 위해 (원본은 small_l_raw, 마스킹된 것은 small_l)
                small_l_raw = small_l.copy()

                bgra_r  = right_mat.get_data()

                bgr_r   = cv2.cvtColor(bgra_r, cv2.COLOR_BGRA2BGR)
                small_r = cv2.resize(bgr_r, (640,480), interpolation=cv2.INTER_AREA)
                small_r = np.ascontiguousarray(small_r, dtype=np.uint8)
                # [추가] 오른쪽 이미지 원본 복사본 저장
                small_r_raw = small_r.copy()

                # [추가] 깊이 맵 데이터 처리 및 저장
                depth_image_float = depth_mat.get_data()
                depth_image_float = np.nan_to_num(depth_image_float, nan=0, posinf=10, neginf=0)  #(1200, 1920)
                
                # 공유 메모리 저장을 위해 480x640 크기로 리사이즈
                depth_resized = cv2.resize(depth_image_float, (640, 480), interpolation=cv2.INTER_AREA)
                
                # 뎁스맵 공유 메모리에 쓰기 (deploy 등 다른 워커에서 사용)
                depth_map_shm.write_data(depth_map=depth_resized, depth_map_ts=np.int64(time.perf_counter_ns()))

                # [추가] Matplotlib으로 깊이 맵 실시간 플롯: 디버깅용 시각화
                # --- Matplotlib으로 뎁스맵 실시간 플롯 ---
                current_time = time.time()
                if PLOT_DEPTH_MAP and (current_time - last_plot_time > PLOT_INTERVAL):
                    # 시각화를 위해 뎁스 값 정규화 (0~10m 범위)
                    depth_for_display = np.clip(depth_image_float, 0, 10) / 10.0
                    
                    # 플롯 업데이트
                    ax.clear()
                    ax.imshow(depth_for_display, cmap='viridis')
                    plt.pause(0.001)  # GUI가 업데이트될 시간을 줌

                    last_plot_time = current_time
                # -----------------------------------------

                # [추가] 마스크 관련 로직 전체 블록: 원본 코드에는 없었던 기능
                # ArUco 마커를 이용한 작업 공간 마스크 생성 및 관리
                ####################### 마스크 관련 로직 시작 ########################

                # [추가] ArUco 마커 인식 및 마스크 생성 로직 (양쪽 카메라 모두)
                # ArUco 마커 인식 및 마스크 생성 (양쪽 카메라 모두)
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters()

                # 그레이스케일 변환 (양쪽 모두)
                gray_l = cv2.cvtColor(small_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(small_r, cv2.COLOR_BGR2GRAY)

                # ArUco 마커 감지 (양쪽 모두): 4개 마커(ID: 0,1,2,3)로 작업 공간 영역 정의
                corners_l, ids_l, rejected_l = cv2.aruco.detectMarkers(gray_l, aruco_dict, parameters=parameters)
                corners_r, ids_r, rejected_r = cv2.aruco.detectMarkers(gray_r, aruco_dict, parameters=parameters)

                # 초기화
                num_markers = 0
                marker_ids = np.full(4, -1, dtype=np.int32)
                marker_corners = np.zeros((4, 4, 2), dtype=np.float32)
                marker_centers = np.zeros((4, 2), dtype=np.float32)

                # [추가] 마스크 제어 상태 확인: UI에서 마스크 생성 ON/OFF 상태를 공유 메모리로부터 읽음
                # 마스크 제어 상태 확인
                try:
                    mask_control_data = mask_control_shm.read_data()
                    mask_control_enabled = bool(mask_control_data["mask_control_enabled"].item())
                except Exception as e:
                    logger_mp.warning(f"[ZED] 마스크 제어 상태 읽기 실패: {e}")
                    mask_control_enabled = False

                # [추가] 마스크 생성/유지 로직: 마커가 감지되면 마스크 생성, 없으면 이전 마스크 재사용
                # 마스크 생성/유지 로직
                if mask_control_enabled:
                    current_time = time.time()
                    time_to_update = (current_time - last_mask_update_time) > MASK_UPDATE_INTERVAL
                    toggled_on = not prev_mask_control_enabled

                    if toggled_on or time_to_update:
                        if toggled_on:
                            logger_mp.info("[ZED] Mask control ON. Attempting to create initial mask...")
                        # else:
                            #logger_mp.info(f"[ZED] {MASK_UPDATE_INTERVAL}s elapsed. Attempting periodic mask update...")

                        # 현재 프레임에서 새 마스크 생성을 시도
                        temp_mask_l, temp_contour_l, temp_corners_l = _create_workspace_mask(small_l, corners_l, ids_l)
                        temp_mask_r, temp_contour_r, temp_corners_r = _create_workspace_mask(small_r, corners_r, ids_r)

                        # ArUco 마커가 성공적으로 감지되었는지 확인 (코너 좌표가 있는지)
                        if np.any(temp_corners_l):
                            #logger_mp.info("[ZED] Mask update successful. Storing new mask.")
                            saved_mask_l = temp_mask_l
                            saved_mask_r = temp_mask_r
                            saved_mask_contour_l = temp_contour_l
                            saved_mask_contour_r = temp_contour_r
                            saved_marker_corners_l = temp_corners_l
                            saved_marker_corners_r = temp_corners_r
                            last_mask_update_time = current_time
                        else:
                            logger_mp.warning("[ZED] Mask update failed (markers not found).")
                            if saved_mask_l is None:
                                #logger_mp.warning("[ZED] No previous mask available. Using an empty mask.")
                                height, width = small_l.shape[:2]
                                saved_mask_l = np.zeros((height, width), dtype=np.uint8)
                                saved_mask_r = np.zeros((height, width), dtype=np.uint8)
                                saved_mask_contour_l = np.zeros(8, dtype=np.float64)
                                saved_mask_contour_r = np.zeros(8, dtype=np.float64)
                                saved_marker_corners_l = np.zeros(8, dtype=np.float64)
                                saved_marker_corners_r = np.zeros(8, dtype=np.float64)
                            #else:
                                #logger_mp.info("[ZED] Reusing previously stored mask.")

                    # 가장 마지막에 성공적으로 저장된 마스크를 현재 프레임에 사용
                    if saved_mask_l is not None:
                        mask_l, mask_r = saved_mask_l, saved_mask_r
                        mask_contour_l, mask_contour_r = saved_mask_contour_l, saved_mask_contour_r
                        marker_corners_l, marker_corners_r = saved_marker_corners_l, saved_marker_corners_r
                    else:
                        # 안전 장치: 만약 저장된 마스크가 없다면 빈 마스크로 초기화
                        height, width = small_l.shape[:2]
                        mask_l = np.zeros((height, width), dtype=np.uint8)
                        mask_r = np.zeros((height, width), dtype=np.uint8)
                        mask_contour_l = np.zeros(8, dtype=np.float64)
                        mask_contour_r = np.zeros(8, dtype=np.float64)
                        marker_corners_l = np.zeros(8, dtype=np.float64)
                        marker_corners_r = np.zeros(8, dtype=np.float64)

                else:
                    # [추가] 마스크 제어 OFF 시 처리: 모든 영역이 통과되는 마스크(255 값) 사용
                    # 마스크 제어 OFF: 디폴트 마스크 (모든 영역 통과)
                    height, width = small_l.shape[:2]
                    mask_l = np.ones((height, width), dtype=np.uint8) * 255
                    mask_r = np.ones((height, width), dtype=np.uint8) * 255
                    mask_contour_l = np.zeros(8, dtype=np.float64)
                    mask_contour_r = np.zeros(8, dtype=np.float64)
                    marker_corners_l = np.zeros(8, dtype=np.float64)
                    marker_corners_r = np.zeros(8, dtype=np.float64)
                    
                    # 마스크 제어가 꺼졌을 때, 저장된 마스크와 타이머를 리셋
                    if prev_mask_control_enabled:
                        logger_mp.info("[ZED] Mask control OFF. Clearing saved mask.")
                        saved_mask_l = None
                        last_mask_update_time = 0.0

                # 이전 프레임의 마스크 제어 상태 업데이트
                prev_mask_control_enabled = mask_control_enabled

                # [추가] 왼쪽 카메라 ArUco 정보 저장: UI에 마커 위치 표시용
                # 왼쪽 카메라 ArUco 정보 저장 (UI 표시용)
                if ids_l is not None and len(ids_l) > 0:
                    num_markers = min(len(ids_l), 4)

                    #print(f"[ZED] 🎯 왼쪽 카메라 ArUco 마커 감지됨! 개수: {num_markers}, ID들: {ids_l.flatten()[:num_markers]}")

                    for i in range(num_markers):
                        marker_id = ids_l[i][0]
                        marker_ids[i] = marker_id
                        marker_corners[i] = corners_l[i][0]
                        center_x = np.mean(corners_l[i][0][:, 0])
                        center_y = np.mean(corners_l[i][0][:, 1])
                        marker_centers[i] = [center_x, center_y]
                else:
                    num_markers = 0

                # [추가] 마스크 평탄화 준비: 공유 메모리에 저장하기 위해 2D 배열을 1D로 변환
                # 마스크 평탄화 준비
                mask_left_flat = mask_l.astype(np.float64).flatten()  # 왼쪽 마스크 평탄화
                mask_right_flat = mask_r.astype(np.float64).flatten()  # 오른쪽 마스크 평탄화


                # [추가] 작업 공간 마스크를 공유 메모리에 저장 (deploy 등 다른 워커에서 사용)
                # 작업 공간 마스크를 공유 메모리에 저장
                workspace_mask_shm.write_data(
                    mask_left_flat=mask_left_flat,
                    mask_right_flat=mask_right_flat,
                    mask_timestamp=time.time(),
                    mask_contour_left=mask_contour_l.astype(np.float64).flatten(),
                    mask_contour_right=mask_contour_r.astype(np.float64).flatten(),
                    marker_corners_left=marker_corners_l.astype(np.float64).flatten(),
                    marker_corners_right=marker_corners_r.astype(np.float64).flatten()
                )

                #print(f"[ZED]   📦 작업 공간 마스크 생성됨 ({np.sum(mask_l > 0)} 픽셀)")

                # [변경] camera_shm.write_data 변경: 원본에는 small_l, small_r를 직접 저장했지만,
                #        여기서는 원본 이미지(small_l_raw, small_r_raw)를 저장 (마스킹은 다른 워커에서 수행)
                # 공유 메모리에 카메라와 ArUco 데이터 저장
                camera_shm.write_data(
                    camera_left=small_l_raw,
                    camera_right=small_r_raw,
                    camera_zed_ts=np.int64(time.perf_counter_ns()),
                )
                # [추가] ArUco 마커 정보를 공유 메모리에 저장: UI에 마커 위치 표시용
                aruco_shm.write_data(
                    num_markers=num_markers,
                    marker_ids=marker_ids,
                    marker_corners=marker_corners,
                    marker_centers=marker_centers,
                    detection_timestamp=time.time()
                )

                missed_count = 0

            else:
                missed_count += 1
                logger_mp.warning(f"[ZED] Missed frame ({missed_count}/{max_missed})")
                if missed_count >= max_missed:
                    logger_mp.error("[ZED] Too many missed frames. Triggering emergency.")
                    shared_event['emergency'].set()
                    break
                time.sleep(0.01)

    except Exception as e:
        logger_mp.error(f"[ZED] Unexpected error: {e}")
        shared_event['emergency'].set()

    finally:
        try:
            zed.close()
            logger_mp.info("[ZED] ZED camera closed.")
        except:
            pass
        camera_shm.worker_close()
        # [추가] 추가된 공유 메모리 종료 처리
        aruco_shm.worker_close()
        workspace_mask_shm.worker_close()
        freq_shm.worker_close()
        # [추가] 깊이 맵 공유 메모리 종료
        depth_map_shm.worker_close()

        logger_mp.info("[ZED] Worker exiting cleanly.")
