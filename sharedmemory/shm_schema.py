#shm_schema.py
import numpy as np

# (field_name, shape, dtype)

# ---- 직교 옵션 매핑 (main.py CLI flag <-> int code) ----
# 텔레옵은 항상 동일한 골격(arm IK + Inspire 양손 + ZED|RealSense 카메라).
# 정책 분기는 이 세 축으로만 결정한다.
HAND_MAPPING       = {'inspire':    0, 'dex3':    1}
CAMERA_MAPPING     = {'zed':        0, 'realsense': 1, 'auto': 2, 'none': 3}
VR_INPUT_MAPPING   = {'hand':       0, 'controller': 1}
# Phase F: HMD→waist 매핑 on/off, head DXL on/off
WAIST_MAPPING      = {'hmd':        0, 'fixed':    1}   # 0=HMD R_delta 적용, 1=고정 (init q 유지)
HEAD_MAPPING       = {'dxl':        0, 'off':      1}   # 0=Dynamixel 사용, 1=DXL skip

HAND_MAPPING_INV     = {v: k for k, v in HAND_MAPPING.items()}
CAMERA_MAPPING_INV   = {v: k for k, v in CAMERA_MAPPING.items()}
VR_INPUT_MAPPING_INV = {v: k for k, v in VR_INPUT_MAPPING.items()}
WAIST_MAPPING_INV    = {v: k for k, v in WAIST_MAPPING.items()}
HEAD_MAPPING_INV     = {v: k for k, v in HEAD_MAPPING.items()}


# NOTE: 모든 streaming SHM 에 ts (monotonic nanosecond) 필드 추가 (Phase D).
# time.perf_counter_ns() 값. writer 가 write_data 직전 캡처해서 함께 publish.
# reader / recorder 는 ts 를 이용해 stream 간 정렬 (epsiode close 시 보간).
# value 0 은 "아직 published 되지 않음" 을 의미.
ROBOT_OBS = [
    ("obs_leg",        (12,),    np.float64),
    ("obs_waist",        (3,),    np.float64),
    ("obs_head",        (2,),    np.float64),
    ("obs_arm",        (14,),    np.float64),
    # 양손 hand state — max DOF = DEX3 14 (7+7). Inspire 의 경우 [:12] 만 의미가 있고
    # 나머지 [12:14] 는 0. 학습/배포 측 modality.json 에서 hand 종류에 맞게 슬라이싱.
    ("obs_hand",        (14,),    np.float64),
    # body obs ts (worker_g1_ctrl.do_fast write 시점 nanosecond)
    ("obs_body_ts",    (),       np.int64),
    # hand obs ts (worker_hand_ctrl 가 dual_hand_state_array 를 SHM 에 쓰는 시점)
    ("obs_hand_ts",    (),       np.int64),
]

ROBOT_ACTION = [
    ("action_leg",        (12,),    np.float64),
    ("action_leg_tauff",        (12,),    np.float64),
    ("action_waist",        (3,),    np.float64),
    ("action_waist_tauff",        (3,),    np.float64),
    ("action_head",        (2,),    np.float64),
    ("action_arm",        (14,),    np.float64),
    ("action_arm_tauff",        (14,),    np.float64),
    # 같은 이유로 14 로 통일. Inspire 는 [:12], DEX3 는 [:14] 모두 사용.
    ("action_hand",        (14,),    np.float64),
    # body action ts (worker_g1_ik / worker_deploy_policy write 시점)
    ("action_body_ts", (),       np.int64),
    # hand action ts (worker_hand_ctrl / worker_deploy_policy hand 쓰기 시점)
    ("action_hand_ts", (),       np.int64),
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
    # ZED stereo frame ts (left/right 같이 grab 했을 때의 host 수신 ts)
    ("camera_zed_ts",       (), np.int64),
    # RealSense color frame ts
    ("camera_realsense_ts", (), np.int64),
]

DEPTH_MAP = [
    ("depth_map", (480, 640), np.float32),
    ("depth_map_ts", (), np.int64),
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
    # worker_vr SHM write 시점 ts
    ("television_ts",    (),        np.int64),
]

# Quest3 controller 입력 (Vuer CONTROLLER_MOVE 이벤트로부터)
# - 좌/우 4x4 SE(3) pose (Robot Convention 으로 변환 후 저장)
# - trigger / squeeze(=grip) / thumbstick / buttons (a, b, thumbstick_click)
# - connected: 한 번이라도 컨트롤러 이벤트가 들어왔는지 여부
QUEST_CONTROLLER = [
    ("left_ctrl_mat",     (4, 4),  np.float64),
    ("right_ctrl_mat",    (4, 4),  np.float64),
    ("left_trigger",      (),      np.float32),
    ("left_squeeze",      (),      np.float32),
    ("left_thumbstick",   (2,),    np.float32),
    ("left_buttons",      (3,),    np.float32),  # [a/x, b/y, thumbstick_click]
    ("right_trigger",     (),      np.float32),
    ("right_squeeze",     (),      np.float32),
    ("right_thumbstick",  (2,),    np.float32),
    ("right_buttons",     (3,),    np.float32),
    ("connected",         (),      np.bool_),
    # CONTROLLER_MOVE 이벤트 처리 직후 worker_vr 가 SHM 쓴 시점 ts
    ("controller_ts",     (),      np.int64),
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

# 텔레옵 구성(직교 옵션) 공유 메모리.
# main.py 가 시작 시 1회 채우고, record/log 등에서 참고만 한다 (분기 X).
TELEOP_CONFIG = [
    ("hand_type",  (), np.int32),  # HAND_MAPPING       (0=inspire, 1=dex3)
    ("camera_type",(), np.int32),  # CAMERA_MAPPING     (0=zed, 1=realsense, 2=auto, 3=none)
    ("vr_input",   (), np.int32),  # VR_INPUT_MAPPING   (0=hand, 1=controller)
    ("waist_mode", (), np.int32),  # WAIST_MAPPING      (0=hmd, 1=fixed)
    ("head_mode",  (), np.int32),  # HEAD_MAPPING       (0=dxl, 1=off)
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
    ("l_touch_ts",               (),       np.int64),
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
    ("r_touch_ts",               (),       np.int64),
]