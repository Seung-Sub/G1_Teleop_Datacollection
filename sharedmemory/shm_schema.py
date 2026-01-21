#shm_schema.py
import numpy as np

# (field_name, shape, dtype)

MODE_MAPPING = {
    'teleop': 0,
    'gr00t': 1,
    'amo': 2,
    'gr00t_zed': 3,
    'kistar_teleop': 4,
    'kistar_only': 5,
    'kistar_inspire_teleop': 6
}

ROBOT_OBS = [
    ("obs_leg",        (12,),    np.float64),
    ("obs_waist",        (3,),    np.float64),
    ("obs_head",        (2,),    np.float64),
    ("obs_arm",        (14,),    np.float64),
    ("obs_hand",        (12,),    np.float64),
]

ROBOT_ACTION = [
    ("action_leg",        (12,),    np.float64),
    ("action_leg_tauff",        (12,),    np.float64),
    ("action_waist",        (3,),    np.float64),
    ("action_waist_tauff",        (3,),    np.float64),
    ("action_head",        (2,),    np.float64),
    ("action_arm",        (14,),    np.float64),
    ("action_arm_tauff",        (14,),    np.float64),
    ("action_hand",        (12,),    np.float64),
]

ROBOT_AMO_OBS = [
    ("amo_q",        (23,),    np.float64),
    ("amo_dq",        (23,),    np.float64),
    ("quat",        (4,),    np.float64),
    ("ang_vel",        (3,),    np.float64),
]

ROBOT_AMO_INPUT = [
    ("pelvis_pose",        (7,),    np.float64),    # xyz, xyzw
    ("pelvis_height",        (1,),    np.float64),
    ("torso_quat",        (4,),    np.float64),
    ("vel_command",        (3,),    np.float64),
]

# KISTAR Hand Action 전용 스키마 (전송된 제어 명령 저장)
KISTAR_HAND_ACTION = [
    ("hand_action",        (16,),   np.float32),    # 16개 조인트 액션 (action - 전송된 값)
]

# ASUS NUC로부터 받는 KISTAR Hand 데이터
KISTAR_HAND_RECEIVED = [
    ("hand_q_pos",         (16,),   np.float32),    # 16개 조인트 위치
    ("play_cnt",           (),      np.int32),      # 플레이 카운트
]

GR00T_TASK_LAYOUT = [
    ("task_name",      (),    np.dtype("U64")),
]

WORKER_FREQ = [
    ("g1_freq",        (),    np.float64),
    ("hand_freq",        (),    np.float64),
    ("vr_freq",        (),    np.float64),
    ("camera_freq",        (),    np.float64),
    ("aruco_freq",     (),    np.float64),
    ("record_freq",        (),    np.float64),
]


CAMERA = [
    ("camera_left", (480, 640, 3), np.uint8),
    ("camera_right", (480, 640, 3), np.uint8),
    ("realsense", (480, 640, 3), np.uint8),
]

DEPTH_MAP = [
    ("depth_map", (480, 640), np.float32),
]

# ArUco 마커 인식 결과 공유 메모리 스키마
ARUCO_MARKERS = [
    ("num_markers", (), np.int32),  # 감지된 마커 개수
    ("marker_ids", (4,), np.int32),  # 최대 4개 마커 ID (책상 네 모서리)
    ("marker_corners", (4, 4, 2), np.float32),  # 각 마커의 4개 코너 좌표 (x,y)
    ("marker_centers", (4, 2), np.float32),  # 각 마커 중심 좌표
    ("detection_timestamp", (), np.float64),  # 감지 타임스탬프
]

# 작업 공간 마스크 공유 메모리 스키마 (모두 float64로 통일)
WORKSPACE_MASK = [
    ("mask_timestamp", (), np.float64),      # 마스크 생성 타임스탬프
    ("mask_left_flat", (307200,), np.float64),    # 왼쪽 카메라 마스크 평탄화 (480*640)
    ("mask_right_flat", (307200,), np.float64),  # 오른쪽 카메라 마스크 평탄화 (480*640)
    ("mask_contour_left", (8,), np.float64),     # 왼쪽 마스크 테두리 좌표 평탄화 (4*2)
    ("mask_contour_right", (8,), np.float64),    # 오른쪽 마스크 테두리 좌표 평탄화 (4*2)
    ("marker_corners_left", (8,), np.float64),   # 왼쪽 마커 꼭지점 좌표 평탄화 (4*2)
    ("marker_corners_right", (8,), np.float64),  # 오른쪽 마커 꼭지점 좌표 평탄화 (4*2)
]

TELEVISION = [
    ("head_rmat",        (4, 4),    np.float64),
    ("left_wrist_mat",   (4, 4),    np.float64),
    ("right_wrist_mat",  (4, 4),    np.float64),
    ("left_hand",        (5, 3),    np.float64),
    ("right_hand",       (5, 3),    np.float64),
    ("right_distal",     (5, 3),    np.float64),
    ("right_proximal",   (5, 3),    np.float64),
]

RECORD_TASK_LAYOUT = [
    ("task_name",      (),    np.dtype("U64")),
]

RECORD_EPISODE_LAYOUT = [
    ("num_episodes",   (),    np.int32),
    ("episode_len",    (),    np.int32),
    ("delete_idx",     (),    np.int32),
    ("replay_idx",     (),    np.int32),
    ("episode_index",    (),    np.int32),
    ("logging_progress",    (),    np.int32),
]

RECORD_MODE_LAYOUT = [
    ("start",          (),    np.bool_),
    ("done",          (),    np.bool_),
    ("reset",          (),    np.bool_),
    ("replay",         (),    np.bool_),
    ("home",          (),    np.bool_),
    ("deploy",          (),    np.bool_),
]

MASK_CONTROL_LAYOUT = [
    ("mask_control_enabled", (), np.bool_),  # 마스크 제어 활성화 상태
    ("generate_new_mask",    (), np.bool_),  # 새 마스크 생성 요청
]

# 모드 정보 전용 공유 메모리 (정수형 모드 사용)
CURRENT_MODE_LAYOUT = [
    ("mode",           (),    np.int32),  # 모드 정보 저장 (0: teleop, 1: kistar_teleop, 2: kistar_only 등)
]

LEFT_TOUCH_SENSOR_LAYOUT = [
    # 왼손 (left)
    ("l_fingerone_tip_touch",    (3,  3),  np.int16),  # 작은손가락 끝 (3×3)
    ("l_fingerone_top_touch",    (12, 8),  np.int16),  # 작은손가락 손톱 (12×8)
    ("l_fingerone_palm_touch",   (10, 8),  np.int16),  # 작은손가락 패드 (10×8)

    ("l_fingertwo_tip_touch",    (3,  3),  np.int16),  # 무명지 끝 (3×3)
    ("l_fingertwo_top_touch",    (12, 8),  np.int16),  # 무명지 손톱 (12×8)
    ("l_fingertwo_palm_touch",   (10, 8),  np.int16),  # 무명지 패드 (10×8)

    ("l_fingerthree_tip_touch",  (3,  3),  np.int16),  # 중지 끝 (3×3)
    ("l_fingerthree_top_touch",  (12, 8),  np.int16),  # 중지 손톱 (12×8)
    ("l_fingerthree_palm_touch", (10, 8),  np.int16),  # 중지 패드 (10×8)

    ("l_fingerfour_tip_touch",   (3,  3),  np.int16),  # 검지 끝 (3×3)
    ("l_fingerfour_top_touch",   (12, 8),  np.int16),  # 검지 손톱 (12×8)
    ("l_fingerfour_palm_touch",  (10, 8),  np.int16),  # 검지 패드 (10×8)

    ("l_fingerfive_tip_touch",   (3,  3),  np.int16),  # 엄지 끝 (3×3)
    ("l_fingerfive_top_touch",   (12, 8),  np.int16),  # 엄지 손톱 (12×8)
    ("l_fingerfive_middle_touch",(3,  3),  np.int16),  # 엄지 중간 섹션 (3×3)
    ("l_fingerfive_palm_touch",  (12, 8),  np.int16),  # 엄지 패드 (12×8)

    ("l_palm_touch",             (8, 14),  np.int16),  # 왼손 손바닥 (8×14)
]

RIGHT_TOUCH_SENSOR_LAYOUT = [

    ("r_fingerone_tip_touch",    (3,  3),  np.int16),  # 작은손가락 끝 (3×3)
    ("r_fingerone_top_touch",    (12, 8),  np.int16),  # 작은손가락 손톱 (12×8)
    ("r_fingerone_palm_touch",   (10, 8),  np.int16),  # 작은손가락 패드 (10×8)

    ("r_fingertwo_tip_touch",    (3,  3),  np.int16),  # 무명지 끝 (3×3)
    ("r_fingertwo_top_touch",    (12, 8),  np.int16),  # 무명지 손톱 (12×8)
    ("r_fingertwo_palm_touch",   (10, 8),  np.int16),  # 무명지 패드 (10×8)

    ("r_fingerthree_tip_touch",  (3,  3),  np.int16),  # 중지 끝 (3×3)
    ("r_fingerthree_top_touch",  (12, 8),  np.int16),  # 중지 손톱 (12×8)
    ("r_fingerthree_palm_touch", (10, 8),  np.int16),  # 중지 패드 (10×8)

    ("r_fingerfour_tip_touch",   (3,  3),  np.int16),  # 검지 끝 (3×3)
    ("r_fingerfour_top_touch",   (12, 8),  np.int16),  # 검지 손톱 (12×8)
    ("r_fingerfour_palm_touch",  (10, 8),  np.int16),  # 검지 패드 (10×8)

    ("r_fingerfive_tip_touch",   (3,  3),  np.int16),  # 엄지 끝 (3×3)
    ("r_fingerfive_top_touch",   (12, 8),  np.int16),  # 엄지 손톱 (12×8)
    ("r_fingerfive_middle_touch",(3,  3),  np.int16),  # 엄지 중간 섹션 (3×3)
    ("r_fingerfive_palm_touch",  (12, 8),  np.int16),  # 엄지 패드 (12×8)

    ("r_palm_touch",             (8, 14),  np.int16),  # 오른손 손바닥 (8×14)
]