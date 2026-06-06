"""G1 Teleoperation entrypoint.

Three orthogonal options decide the worker set:
    --hand      {inspire}          (default: inspire; future hands plug in here)
    --camera    {zed, realsense}   (default: zed)
    --vr-input  {hand, controller} (default: hand)

The recording pipeline (worker_record + parquet/video sinks) is *always*
spawned. Pressing the record button in the UI/keyboard during teleop
captures an episode under record/<task>/.

Deploy / policy evaluation is intentionally not part of this entrypoint;
it will be re-introduced as a separate `evaluate.py` (Phase 6).
"""

import sys
import os
import time
import errno
import atexit
import signal
import argparse
import multiprocessing as mp
from multiprocessing import Event, Lock, Process
import numpy as np

import logging_mp
from sharedmemory.shmManager import SharedMemoryManager

from sharedmemory.shm_schema import (
    CAMERA, CAMERA_VIEW, TELEVISION, ARUCO_MARKERS, WORKSPACE_MASK,
    RECORD_TASK_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_MODE_LAYOUT,
    LEFT_TOUCH_SENSOR_LAYOUT, RIGHT_TOUCH_SENSOR_LAYOUT,
    WORKER_FREQ, GR00T_TASK_LAYOUT, ROBOT_OBS, ROBOT_ACTION,
    MASK_CONTROL_LAYOUT, DEPTH_MAP, TELEOP_CONFIG, QUEST_CONTROLLER,
    HAND_MAPPING, CAMERA_MAPPING, VR_INPUT_MAPPING, WAIST_MAPPING, HEAD_MAPPING, TACTILE_MAPPING,
    LOWER_BODY_MAPPING,
)
# NOTE: worker_vr / television / Vuer imports are deferred into
# get_worker_specs() because the params-proto library (a Vuer
# transitive dep) hijacks the global argparse at module-load time --
# importing worker_vr here makes `python main.py --help` print Vuer's
# CLI instead of ours. Loading it inside get_worker_specs() keeps
# argparse owned by us.
from gui.ui_launcher import run_ui
from workers.worker_record import worker_record
from workers.keyboard_listener import keyboard_listener

os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = " ".join([
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--enable-gpu-rasterization",
    "--use-gl=desktop",
    "--no-sandbox",
])

# Logging — try/except 로 fail-soft. (worker module 들이 module-level 에서 이미
# get_logger 를 호출했을 가능성이 있어, logging_mp 가 "이미 시작됨" 으로 raise 할
# 수 있다. 그 경우 logger 는 default 설정으로 진행.)
try:
    logging_mp.basic_config(level=logging_mp.INFO)
except RuntimeError:
    pass
logger_mp = logging_mp.get_logger(__name__)

# Shared-memory configurations: layout constant, shm name, lock key.
# Phase K7-A (P0-3 + P1-4): camera_shm 은 backward-compat 만 — CAMERA_VIEW 기반의
# role 별 SHM (rs_ego_shm / rs_wrist_l_shm / rs_wrist_r_shm) 이 실제 사용 SHM.
# main() 에서 cameras 설정에 따라 동적으로 SHM_CONFIG 에 추가.
SHM_CONFIG = {
    'camera_shm':           (CAMERA,                    'camera_shm',           'camera_lock'),
    'television_shm':       (TELEVISION,                'television_shm',       'television_lock'),
    'quest_controller_shm': (QUEST_CONTROLLER,          'quest_controller_shm', 'quest_controller_lock'),
    'aruco_shm':            (ARUCO_MARKERS,             'aruco_shm',            'aruco_lock'),
    'workspace_mask_shm':   (WORKSPACE_MASK,            'workspace_mask_shm',   'workspace_mask_lock'),
    'record_task_shm':      (RECORD_TASK_LAYOUT,        'record_task_shm',      'record_lock'),
    'record_episode_shm':   (RECORD_EPISODE_LAYOUT,     'record_episode_shm',   'record_lock'),
    'record_mode_shm':      (RECORD_MODE_LAYOUT,        'record_mode_shm',      'record_lock'),
    'teleop_config_shm':    (TELEOP_CONFIG,             'teleop_config_shm',    'record_lock'),
    'left_touch_shm':       (LEFT_TOUCH_SENSOR_LAYOUT,  'left_touch_shm',       'left_touch_lock'),
    'right_touch_shm':      (RIGHT_TOUCH_SENSOR_LAYOUT, 'right_touch_shm',      'right_touch_lock'),
    'freq_shm':             (WORKER_FREQ,               'freq_shm',             'record_lock'),
    'gr00t_shm':            (GR00T_TASK_LAYOUT,         'gr00t_shm',            'gr00t_lock'),
    'robot_obs_shm':        (ROBOT_OBS,                 'robot_obs_shm',        'robot_obs_lock'),
    'robot_action_shm':     (ROBOT_ACTION,              'robot_action_shm',     'robot_action_lock'),
    'mask_control_shm':     (MASK_CONTROL_LAYOUT,       'mask_control_shm',     'record_lock'),
    'depth_map_shm':        (DEPTH_MAP,                 'depth_map_shm',        'depth_map_lock'),
}


def create_events():
    return {
        'set_start': Event(),
        'shutdown':  Event(),
        'go_home':   Event(),
        'emergency': Event(),
        'set_g1':    Event(),
        'set_hand':  Event(),
    }


def create_locks():
    return {
        'robot_data_lock':       Lock(),
        'robot_lock':            Lock(),
        'camera_lock':           Lock(),
        'television_lock':       Lock(),
        'quest_controller_lock': Lock(),
        'aruco_lock':            Lock(),
        'workspace_mask_lock':   Lock(),
        'record_lock':           Lock(),
        'left_touch_lock':       Lock(),
        'right_touch_lock':      Lock(),
        'freq_lock':             Lock(),
        'visual_lock':           Lock(),
        'gr00t_lock':            Lock(),
        'robot_obs_lock':        Lock(),
        'robot_action_lock':     Lock(),
        'depth_map_lock':        Lock(),
        # Phase K7-A: 카메라 role 별 lock (단일 운용 시 ego 만 사용).
        'rs_ego_lock':           Lock(),
        'rs_wrist_l_lock':       Lock(),
        'rs_wrist_r_lock':       Lock(),
    }


def create_shm_managers(locks):
    managers = {}
    for key, (layout, name, lock_key) in SHM_CONFIG.items():
        managers[key] = SharedMemoryManager(layout, locks[lock_key], name)
    return managers


def get_shm_names():
    return {key: name for key, (_, name, _) in SHM_CONFIG.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="G1 teleoperation + data collection")
    parser.add_argument('--hand',     choices=list(HAND_MAPPING.keys()),
                        default='inspire', help='Hand hardware')
    parser.add_argument('--camera',   default='auto',
                        help="Camera source: 'zed' | 'realsense' | 'auto' (RealSense 우선 자동감지) | "
                             "'none' | <serial-number> (특정 device serial)")
    parser.add_argument('--camera-role', dest='camera_role', default='ego',
                        help='Camera role label (ego, wrist_l, wrist_r, ...). 기본 ego; 시리얼 지정 시 추후 yaml 매핑.')
    parser.add_argument('--zed-mode',  dest='zed_mode', choices=['direct', 'stream'], default='direct',
                        help="ZED 연결 방식: 'direct' = sl.Camera.open(USB), 'stream' = set_from_stream(외부 송신).")
    parser.add_argument('--cameras-config', dest='cameras_config',
                        default='utils/cameras.yaml',
                        help='cameras.yaml 경로. 비어있으면 --camera 단일 카메라 모드.')
    parser.add_argument('--vr-input', dest='vr_input',
                        choices=list(VR_INPUT_MAPPING.keys()),
                        default='hand',
                        help='Quest3 input mode (hand tracking vs motion controller)')
    # Phase F: waist / head 제어 on/off
    parser.add_argument('--waist', choices=list(WAIST_MAPPING.keys()), default='hmd',
                        help="Waist 제어 모드: 'hmd' = HMD 변위로 waist 제어, 'fixed' = init q 고정")
    parser.add_argument('--head',  choices=list(HEAD_MAPPING.keys()),  default='dxl',
                        help="Head Dynamixel: 'dxl' = 사용, 'off' = 비활성 (Dynamixel 없는 G1 / 데이터 수집 시 head 고정)")
    parser.add_argument('--tactile', choices=list(TACTILE_MAPPING.keys()), default='off',
                        help="DEX3 압력센서 (HandState_.press_sensor_state): 'off' = 미사용 (default), "
                             "'on' = subscribe thread 가 메시지의 press_sensor_state.length 를 로깅. "
                             "실 device 의 sequence length 확정 후 SHM/parquet 컬럼 추가는 후속 작업.")
    # Phase N (PART4): lower-body 모드. 안전상 hoist default.
    parser.add_argument('--lower-body', dest='lower_body', choices=list(LOWER_BODY_MAPPING.keys()),
                        default='hoist',
                        help="lower body 제어: 'hoist'(default) = 기존 동작 (rt/lowcmd, 전체 직접 "
                             "제어, 호이스트 전제). 'loco' = motion mode (rt/arm_sdk, 상체만 명령, "
                             "내장 LocoClient 가 밸런싱/보행). loco 는 motion control 진입 (G1 "
                             "리모컨 L2+B→L2+UP→R1+X) 후 사용. 검증 절차 docs/REMAINING §L11.")
    parser.add_argument('--gait', choices=['off', 'thumbstick'], default='off',
                        help="(loco 모드 전용) 'thumbstick' = Quest3 thumbstick 으로 LocoClient.Move "
                             "보행 제어. 'off' = 보행 없음 (밸런싱만).")
    parser.add_argument('--gait-stick', dest='gait_stick',
                        choices=['split', 'left', 'right'], default='split',
                        help="(gait=thumbstick 전용) stick 매핑: 'split' = 왼쪽 병진 + 오른쪽 회전 "
                             "(xr_teleoperate 공식). 'left' = 왼쪽 스틱만 (X→vyaw 추가). "
                             "'right' = 오른쪽 스틱만.")
    # Inspire thumb 사전 자세 (vr_input=controller 일 때만 사용; 손가락 4개는 trigger로 토글)
    # 값 범위: 0.0(굽힘/안쪽) ~ 1.0(펼침/바깥쪽). 물체에 따라 잡기 편한 자세를 사전 설정.
    # Inspire controller-mode 그립 — 상황별 프로파일 메뉴 (hand_control/inspire_grip_profiles.yaml).
    # --grip-profile 로 "손가락 수 + 엄지 각도 + force/speed" 묶음 선택. 아래 개별 플래그는 override.
    # (DEX3 는 thumb_bend/thumb_yaw 만 사용, 나머지는 무시.)
    parser.add_argument('--grip-profile', dest='grip_profile', default=None,
                        help="(Inspire) 그립 프로파일 이름 (inspire_grip_profiles.yaml). "
                             "예: full_oppose|tripod|pinch|lateral|hook. 미지정 시 파일의 default_profile.")
    parser.add_argument('--thumb-bend', dest='thumb_bend', type=float, default=None,
                        help='Inspire 엄지 굽힘 0..1 (controller). 미지정 시 프로파일 값.')
    parser.add_argument('--thumb-yaw',  dest='thumb_yaw',  type=float, default=None,
                        help='Inspire 엄지 회전(대향 각도) 0..1 (controller). 미지정 시 프로파일 값.')
    parser.add_argument('--grasp-fingers', dest='grasp_fingers', default=None,
                        help="(Inspire) 파지 시 닫히는 손가락 subset, comma 구분. "
                             "pinky,ring,middle,index (+thumb 포함 시 엄지도 grasp 때 굽힘). "
                             "예: --grasp-fingers thumb,index,middle. 미지정 시 프로파일 값.")
    parser.add_argument('--close-depth', dest='close_depth', type=float, default=None,
                        help="(Inspire) 파지 깊이 0..1 (1.0=완전 폐쇄). 미지정 시 프로파일 값.")
    parser.add_argument('--grip-force', dest='grip_force', type=int, default=None,
                        help="(Inspire) force_set 0..1000(g). 도달 시 펌웨어 정지(STATUS=3)=과부하 차단. "
                             "미지정 시 프로파일 값. (deploy 에서도 적용되는 안전 envelope.)")
    parser.add_argument('--grip-speed', dest='grip_speed', type=int, default=None,
                        help="(Inspire) speed_set 0..1000 (1000=full≈800ms). 미지정 시 프로파일 값.")
    # Phase F: G1/Hand 하드웨어 없이 Quest3 입력 + IK 계산만 검증
    parser.add_argument('--no-robot', dest='no_robot', action='store_true',
                        help='G1 / hand 워커 spawn 생략 + set_g1/set_hand 자동 set (Quest3 + IK 검증용)')
    return parser.parse_args()


def resolve_camera(args):
    """Resolve --camera 값(zed/realsense/auto/none/<serial>) 을 (type, serial, name) 으로 반환.

    type ∈ {'zed', 'realsense', 'none'}. serial 은 worker_camera/worker_zed 가 사용.
    Phase K7-A 이후: 본 함수는 단일 카메라 (ego 1대) 경로용. 멀티-카메라는
    resolve_cameras_config() 가 cameras.yaml 을 읽어 처리.
    """
    cam = (args.camera or 'auto').strip()
    if cam == 'none':
        return ('none', None, None)
    if cam == 'auto':
        from utils.camera_discovery import auto_select
        ct, sn, nm = auto_select(prefer='realsense')
        if ct is None:
            logger_mp.warning("[main] --camera auto: no device detected → fall back to 'none'")
            return ('none', None, None)
        return (ct, sn, nm)
    if cam in ('zed', 'realsense'):
        # 종류만 명시 — auto-select 로 첫 device 잡기 (없으면 worker 가 실패 처리)
        from utils.camera_discovery import discover_realsense, discover_zed
        devs = discover_realsense() if cam == 'realsense' else discover_zed()
        if devs:
            d = devs[0]
            return (cam, d['serial'], d['name'])
        return (cam, None, None)
    # serial 가정 — 양쪽 discover 모두 시도
    from utils.camera_discovery import find_by_serial
    res = find_by_serial(cam)
    if res is None:
        raise ValueError(f"--camera={cam!r} 가 연결된 device 와 매칭 안 됨")
    return res


def resolve_cameras_config(args):
    """cameras.yaml 을 읽어 활성 카메라 리스트 반환.

    Returns:
        list of dict: [{'role', 'type', 'serial', 'name'}, ...]
        - yaml 의 cameras 가 비어 있거나 파일 없으면 → 단일 카메라 모드.
          resolve_camera(args) 결과를 ego 1개로 wrap.
        - --camera none 이면 빈 list 반환.

    설계 (사용자 지시서 K7,8,9_Instruction.md §25 채택):
        "단일=ego 1개짜리 멀티뷰" 로 통일. SHM 이름 체계 / record / deploy 에서
        분기 없이 동일 경로로 처리. 단일 운용 = cameras 리스트 길이 1.
    """
    import yaml as _yaml
    import os as _os
    yaml_path = args.cameras_config
    if yaml_path and _os.path.exists(yaml_path):
        try:
            with open(yaml_path) as f:
                cfg = _yaml.safe_load(f) or {}
            cams = cfg.get('cameras') or []
        except Exception as e:
            logger_mp.warning(f"[main] {yaml_path} parse 실패: {e} — 단일 카메라 fallback")
            cams = []
    else:
        cams = []

    if cams:
        # YAML driven multi-camera (or single via yaml)
        out = []
        for cam in cams:
            role = cam.get('role', 'ego')
            typ  = cam.get('type', 'realsense')
            sn   = cam.get('serial')
            if sn in (None, '', 'auto', 'AUTO_OR_FILL_HERE'):
                # auto-select 첫 device
                from utils.camera_discovery import discover_realsense, discover_zed
                devs = discover_realsense() if typ == 'realsense' else discover_zed()
                if devs:
                    sn = devs[0]['serial']
                    nm = devs[0]['name']
                else:
                    logger_mp.warning(f"[main] cameras.yaml role={role} type={typ}: device 없음 — skip")
                    continue
            else:
                nm = None
            out.append({'role': role, 'type': typ, 'serial': str(sn), 'name': nm})
        return out

    # YAML 미사용 → 단일 카메라 (ego 1대)
    typ, sn, nm = resolve_camera(args)
    if typ == 'none':
        return []
    return [{'role': 'ego', 'type': typ, 'serial': str(sn) if sn else None, 'name': nm}]


# Phase K7-A: role 별 SHM key / lock key 매핑.
ROLE_TO_SHM_KEY  = {'ego': 'rs_ego_shm',  'wrist_l': 'rs_wrist_l_shm',  'wrist_r': 'rs_wrist_r_shm'}
ROLE_TO_LOCK_KEY = {'ego': 'rs_ego_lock', 'wrist_l': 'rs_wrist_l_lock', 'wrist_r': 'rs_wrist_r_lock'}


def write_teleop_config(locks, shm_names, args):
    """Write the chosen options into TELEOP_CONFIG SHM (info-only)."""
    # camera_type 은 cameras 리스트의 첫 카메라 type 기준 (단일/멀티 모두).
    if args.cameras:
        cam_type_str = args.cameras[0]['type']
    else:
        cam_type_str = 'none'
    cfg = SharedMemoryManager(TELEOP_CONFIG, locks["record_lock"], shm_names["teleop_config_shm"])
    cfg.write_data(
        hand_type   =np.int32(HAND_MAPPING[args.hand]),
        camera_type =np.int32(CAMERA_MAPPING.get(cam_type_str, CAMERA_MAPPING['none'])),
        vr_input    =np.int32(VR_INPUT_MAPPING[args.vr_input]),
        waist_mode  =np.int32(WAIST_MAPPING[args.waist]),
        head_mode   =np.int32(HEAD_MAPPING[args.head]),
        tactile_mode=np.int32(TACTILE_MAPPING[args.tactile]),
        lower_body  =np.int32(LOWER_BODY_MAPPING[args.lower_body]),
    )
    cfg.worker_close()

    # 기존 record_mode_shm 초기화
    rm = SharedMemoryManager(RECORD_MODE_LAYOUT, locks["record_lock"], shm_names["record_mode_shm"])
    rm.write_data(
        start =np.bool_(False),
        done  =np.bool_(False),
        reset =np.bool_(False),
        replay=np.bool_(False),
        home  =np.bool_(False),
        deploy=np.bool_(False),
    )
    rm.worker_close()


def get_worker_specs(args, events, locks, shm_names):
    """Build the worker spec list for the chosen options."""
    specs = []

    write_teleop_config(locks, shm_names, args)

    # ---- Robot core (G1 + IK) ---------------------------------------
    # --no-robot 모드: G1 / hand 워커 spec 자체를 미포함 (DDS subscribe 시도 안 함).
    # set_g1 / set_hand 는 main() 에서 미리 set 되어 worker_g1_ik 등이 FSM RUN 진입.
    if not args.no_robot:
        from workers.worker_g1_ctrl import worker_g1_ctrl
        # Phase N: lower_body 인자 전달. loco 면 worker_g1_ctrl 가 G1_29_ArmController
        # 를 motion mode (rt/arm_sdk) 로 init + engage_arm_sdk.
        specs += [
            {'target': worker_g1_ctrl,
             'args':   (events, shm_names, locks, 'teleop', args.head, args.lower_body),
             'name':   'worker_g1_ctrl'},
        ]
        # gait=thumbstick + loco 시 보행 워커 spawn.
        if args.lower_body == 'loco' and args.gait == 'thumbstick':
            from workers.worker_loco import worker_loco
            specs += [
                {'target': worker_loco,
                 'args':   (events, shm_names, locks, args.gait_stick),
                 'name':   'WORKER_LOCO'},
            ]
    from workers.worker_g1_ik import worker_g1_ik
    specs += [
        {'target': worker_g1_ik,
         'args':   (events, shm_names, locks, args.vr_input, args.waist),
         'name':   'worker_g1_ik'},
    ]

    # ---- Hand --------------------------------------------------------
    if not args.no_robot:
        if args.hand == 'inspire':
            from workers.worker_hand_ctrl import worker_hand_ctrl
            from workers.worker_hand_dds  import worker_hand_r_dds, worker_hand_l_dds
            specs += [
                {'target': worker_hand_ctrl,
                 'args':   (events, shm_names, locks, args.hand, args.vr_input, args.thumb_bend, args.thumb_yaw, args.tactile,
                            args.grasp_fingers, args.close_depth, args.grip_force, args.grip_speed),
                 'name':   'WORKER_HAND'},
                # Inspire 전용 터치센서 Modbus DDS
                {'target': worker_hand_r_dds, 'args': ('192.168.123.210', 'r', 'Right-hand process', shm_names, locks),
                 'name': 'WORKER_HAND_R_DDS'},
                {'target': worker_hand_l_dds, 'args': ('192.168.123.211', 'l', 'Left-hand process',  shm_names, locks),
                 'name': 'WORKER_HAND_L_DDS'},
            ]
        elif args.hand == 'dex3':
            from workers.worker_hand_ctrl import worker_hand_ctrl
            specs += [
                {'target': worker_hand_ctrl,
                 'args':   (events, shm_names, locks, args.hand, args.vr_input, args.thumb_bend, args.thumb_yaw, args.tactile),
                 'name':   'WORKER_HAND'},
                # DEX3 는 별도 터치센서 DDS 가 없다 — press_sensor_state 는 HandState_ 메시지에 함께 포함됨.
                # tactile=on 시 robot_hand_dex3 가 length 로깅.
            ]
        else:
            raise ValueError(f"Unsupported hand hardware: {args.hand}")

    # ---- Camera (Phase K7-A: cameras 리스트 기반 N개 worker spawn) --------
    # args.cameras = [{'role','type','serial','name'}, ...] (0..3개)
    if not args.cameras:
        logger_mp.info("[main] camera disabled — no camera worker spawned.")
    else:
        for cam in args.cameras:
            role     = cam['role']
            typ      = cam['type']
            sn       = cam.get('serial')
            shm_key  = ROLE_TO_SHM_KEY[role]
            lock_key = ROLE_TO_LOCK_KEY[role]
            if typ == 'realsense':
                from workers.worker_camera import worker_camera
                specs += [{'target': worker_camera,
                           'args':   (events, shm_names, locks, sn, role, shm_key, lock_key),
                           'name':   f'WORKER_RS_{role.upper()}'}]
            elif typ == 'zed':
                from workers.worker_zed import worker_zed
                specs += [{'target': worker_zed,
                           'args':   (events, shm_names, locks, sn, args.zed_mode, shm_key, lock_key),
                           'name':   f'WORKER_ZED_{role.upper()}'}]
            else:
                raise ValueError(f"Unsupported camera type: {typ}")
            logger_mp.info(f"[main] camera spawn: role={role} type={typ} serial={sn} → shm={shm_key}")

    # ---- Recording (always-on) --------------------------------------
    specs += [{'target': worker_record, 'args': (events, shm_names, locks),
               'name': 'WORKER_RECORD'}]

    # ---- Common: VR / GUI / keyboard --------------------------------
    # worker_vr import는 여기서 (vuer/params-proto 가 argparse 가로채는 것 회피)
    from workers.worker_vr import worker_vr
    # Part5: GUI 가 표시할 카메라 role 목록 (활성 cameras 의 role). 단일/멀티/없음
    # 모두 동일 코드 경로. 빈 리스트면 GUI 가 "신호 없음" 표시.
    active_camera_roles = [c['role'] for c in args.cameras]
    specs += [
        {'target': worker_vr,         'args': (events, shm_names, locks, args.vr_input),
         'name': 'WORKER_VR'},
        {'target': run_ui,            'args': (events, shm_names, locks, active_camera_roles),
         'name': 'UI'},
        {'target': keyboard_listener, 'args': (events,),
         'name': 'KEYBOARD'},
    ]

    return specs


AFFINITY = {
    'worker_g1_ctrl':    {20, 21},
    'WORKER_VR':         {23},
    'WORKER_HAND_R_DDS': {18},
    'WORKER_HAND_L_DDS': {19},
}
# Phase M5 (SUPPLEMENT §L1 보강): K7 이후 카메라 워커 이름이 'WORKER_RS_<ROLE>' /
# 'WORKER_ZED_<ROLE>' 패턴으로 바뀌었으므로 prefix 매칭으로 처리. 동일 prefix 의 모든
# 카메라 워커가 같은 core set 을 공유 (다중 카메라일 때 CPU 분산 시 후속 작업).
AFFINITY_PREFIX = {
    'WORKER_RS_':  {22},   # RealSense 카메라들 (ego/wrist_l/wrist_r)
    'WORKER_ZED_': {22},   # ZED 카메라들
}


def _resolve_affinity(name: str):
    """worker name → cpu set. 정확 매칭 우선, 그 다음 prefix 매칭."""
    if name in AFFINITY:
        return AFFINITY[name]
    for prefix, cpus in AFFINITY_PREFIX.items():
        if name.startswith(prefix):
            return cpus
    return None


def launch_processes(specs):
    processes = []
    for spec in specs:
        p = Process(target=spec['target'], args=spec['args'], daemon=True, name=spec.get('name'))
        p.start()
        cpus = _resolve_affinity(p.name)
        if cpus:
            try:
                os.sched_setaffinity(p.pid, cpus)
            except Exception:
                pass
        processes.append(p)
    return processes


def wait_for_shutdown(event):
    try:
        while not event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        event.set()


def cleanup(processes, managers):
    """Graceful 3단 종료: join(2s) → terminate(1s) → kill, 그 다음 SHM unlink.

    이전엔 join(timeout=2) 만 했어서 worker 가 shutdown event 를 못 봤거나 hang
    상태일 때 좀비 + SHM 누수가 남았다. 이제 모든 경우에 깨끗이 정리.
    """
    # 1) graceful — workers 가 shutdown event 보고 자기 정리.
    #    G1_Ctrl worker(hoist) 는 종료 시 damp_to_release(~3s: ramp 2.5s + settle 0.5s)
    #    로 팔 힘을 점진적으로 빼므로, join timeout 을 그보다 넉넉히(5s) 줘서 damp 가
    #    완료되기 전에 terminate 로 잘리지 않게 한다. (잘리면 팔이 갑자기 떨어짐.)
    for p in processes:
        try:
            p.join(timeout=5)
        except Exception:
            pass

    # 2) SIGTERM (terminate) 후 1초 추가 대기
    for p in processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
    for p in processes:
        if p.is_alive():
            try:
                p.join(timeout=1)
            except Exception:
                pass

    # 3) SIGKILL — 그래도 안 죽으면 강제
    for p in processes:
        if p.is_alive():
            try:
                p.kill()
            except Exception:
                pass

    # SHM unlink (owner) / close (non-owner)
    for mgr in managers.values():
        if getattr(mgr, '_owner', False):
            try:
                mgr.main_unlink()
            except FileNotFoundError:
                pass
        else:
            mgr.worker_close()


# -----------------------------------------------------------------------------
# Pre-flight cleanup — 이전 run 잔존물 자동 정리.
# 매 main.py 실행 시 호출 (main() 의 가장 첫 단계). 사용자가 종료 절차를 매번
# 신경쓸 필요 없도록, 다음을 자동 수행:
#   1) /tmp/teleop_main.pid 가 가리키는 stale main.py process group 을 정리
#      (그 cmdline 이 우리 main.py 인지 검증 후에만 kill — 다른 사용자 process
#       오살(誤殺) 방지).
#   2) 우리 워크스페이스가 owner 인 알려진 SHM 이름들을 /dev/shm 에서 unlink.
#      (SHM_CONFIG 정적 항목 + 카메라 role-keyed 동적 항목 다 포함.)
# -----------------------------------------------------------------------------
PIDFILE = "/tmp/teleop_main.pid"

# 카메라 role-keyed SHM 은 main() 안에서 SHM_CONFIG 에 동적 추가되지만, preflight
# 는 그 전에 돌아 SHM_CONFIG 에 없는 것까지 일괄 청소해야 함. 그래서 명시 나열.
_ROLE_SHM_NAMES = ['rs_ego_shm', 'rs_wrist_l_shm', 'rs_wrist_r_shm']


def _known_shm_names():
    static = [v[1] for v in SHM_CONFIG.values()]
    return static + _ROLE_SHM_NAMES


def _proc_is_our_main(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().decode("utf-8", "ignore")
        return ("python" in cmd) and ("main.py" in cmd)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def preflight_cleanup():
    """이전 main.py run 잔존물 (process + SHM) 정리.

    호출 시점: main() 최상단 (SHM owner-create 직전).
    """
    # 1) stale process group kill
    killed = False
    if os.path.isfile(PIDFILE):
        prev = None
        try:
            with open(PIDFILE) as f:
                prev = int(f.read().strip())
        except Exception:
            prev = None
        if prev and _proc_is_our_main(prev):
            try:
                pgid = os.getpgid(prev)
                print(f"[preflight] killing stale main.py pgid={pgid} (pid={prev})",
                      file=sys.stderr)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(1.5)
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                killed = True
            except (ProcessLookupError, PermissionError) as e:
                print(f"[preflight] killpg failed: {e}", file=sys.stderr)
        try:
            os.remove(PIDFILE)
        except FileNotFoundError:
            pass

    # 2) stale SHM unlink
    unlinked = []
    for name in _known_shm_names():
        path = f"/dev/shm/{name}"
        if os.path.exists(path):
            try:
                os.remove(path)
                unlinked.append(name)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    print(f"[preflight] unlink {path} failed: {e}", file=sys.stderr)
    if unlinked:
        sample = unlinked if len(unlinked) <= 8 else unlinked[:8] + ['...']
        print(f"[preflight] unlinked {len(unlinked)} stale shm: {sample}",
              file=sys.stderr)
    if killed or unlinked:
        # 잠깐 텀 — kill 직후 OS 가 fd close 다 처리하도록.
        time.sleep(0.3)


def _write_pidfile():
    """이번 run 의 PID 를 PIDFILE 에 기록 + atexit 으로 자동 제거."""
    try:
        with open(PIDFILE, "w") as f:
            f.write(str(os.getpid()))
    except OSError as e:
        print(f"[preflight] pidfile write failed: {e}", file=sys.stderr)
        return

    def _remove_pidfile():
        try:
            os.remove(PIDFILE)
        except FileNotFoundError:
            pass
    atexit.register(_remove_pidfile)


def _install_signal_handlers(events):
    """SIGTERM 시에도 (Ctrl-C 와 동일하게) graceful shutdown 트리거."""
    def _on_sig(signum, frame):
        events['shutdown'].set()
    try:
        signal.signal(signal.SIGTERM, _on_sig)
    except Exception:
        pass  # 일부 환경(예: thread 안) 에서는 등록 불가, 무시


_GRIP_PROFILE_PATH = "hand_control/inspire_grip_profiles.yaml"
# dex3 / 프로파일 로드 실패 시의 안전 기본값 (= 기존 동작).
_HAND_SHAPE_FALLBACK = {
    'grasp_fingers': 'pinky,ring,middle,index', 'close_depth': 1.0,
    'thumb_bend': 0.5, 'thumb_yaw': 0.5, 'grip_force': 800, 'grip_speed': 1000,
}


def resolve_hand_shape(args):
    """Inspire 그립 프로파일 + 개별 플래그 override 를 args 에 in-place 반영.

    우선순위: 명시적 CLI 플래그 > --grip-profile 프로파일 > 파일 default_profile > _HAND_SHAPE_FALLBACK.
    dex3 는 프로파일 무관 — None 인 항목만 fallback 으로 채운다 (thumb_bend/yaw 만 의미).
    grasp_fingers 는 리스트로 들어오면 comma 문자열로 정규화.
    """
    prof = {}
    if args.hand == 'inspire':
        try:
            import yaml
            doc = yaml.safe_load(open(_GRIP_PROFILE_PATH))
            profiles = doc.get('profiles', {})
            name = args.grip_profile or doc.get('default_profile')
            if name not in profiles:
                logger_mp.warning(
                    f"[main] grip-profile '{name}' 없음 (선택: {list(profiles)}). fallback 사용.")
            else:
                prof = dict(profiles[name])
                logger_mp.info(f"[main] grip-profile '{name}': {prof}")
        except Exception as e:
            logger_mp.warning(f"[main] grip-profile 로드 실패({e}). fallback 사용.")

    def pick(key):
        v = getattr(args, key)
        if v is not None:          # 명시적 CLI override
            return v
        if key in prof:            # 프로파일 값
            return prof[key]
        return _HAND_SHAPE_FALLBACK[key]

    for key in _HAND_SHAPE_FALLBACK:
        setattr(args, key, pick(key))
    # grasp_fingers 리스트 -> comma 문자열 (worker/controller 는 문자열/리스트 모두 파싱하나 통일).
    if isinstance(args.grasp_fingers, (list, tuple)):
        args.grasp_fingers = ",".join(str(x) for x in args.grasp_fingers)


def main():
    # 이전 run 잔존물 자동 정리 (process + SHM). 사용자가 매번 신경쓸 필요 없게.
    preflight_cleanup()
    _write_pidfile()

    args      = parse_args()
    resolve_hand_shape(args)

    # Phase N — lower_body / vr_input / waist / gait 안전 검증.
    if args.lower_body == 'loco':
        if args.vr_input != 'controller':
            raise SystemExit("[main] --lower-body loco 는 --vr-input controller 만 지원.")
        if args.waist == 'hmd':
            logger_mp.warning(
                "[main] loco + --waist hmd: motion mode 는 보통 waist 직접 제어 비권장. "
                "기본은 --waist fixed 권장 (PART4 §3)."
            )
    elif args.gait == 'thumbstick':
        logger_mp.warning("[main] --gait thumbstick 은 --lower-body loco 필요 — gait off 로 강제.")
        args.gait = 'off'

    # Phase K7-A: cameras.yaml 우선 → 단일 카메라 fallback.
    # args.cameras = [{'role','type','serial','name'}, ...] (멀티/단일 모두 동일 구조).
    args.cameras = resolve_cameras_config(args)
    # role 별 SHM 을 SHM_CONFIG 에 동적 추가 (owner-create 대상에 포함).
    for cam in args.cameras:
        role = cam['role']
        shm_key  = ROLE_TO_SHM_KEY[role]
        lock_key = ROLE_TO_LOCK_KEY[role]
        SHM_CONFIG[shm_key] = (CAMERA_VIEW, shm_key, lock_key)

    events    = create_events()
    locks     = create_locks()
    shm_names = get_shm_names()
    managers  = create_shm_managers(locks)
    # SIGTERM (kill <pid>) 도 SIGINT (Ctrl-C) 와 동일하게 graceful shutdown.
    _install_signal_handlers(events)

    # --no-robot: set_g1/set_hand 사전 set 으로 worker_g1_ik 가 FSM RUN 진입
    if args.no_robot:
        events['set_g1'].set()
        events['set_hand'].set()
        logger_mp.warning(
            "[main] --no-robot: G1/hand 워커 spawn skip + set_g1/set_hand pre-set. "
            "worker_g1_ik 는 정상 spawn 되어 Quest3 입력에 대한 IK 계산만 수행."
        )

    cams_summary = ', '.join(f"{c['role']}={c['type']}:{c.get('serial')}" for c in args.cameras) or 'none'
    logger_mp.info(
        f"[main] hand={args.hand} cameras=[{cams_summary}] "
        f"vr_input={args.vr_input} waist={args.waist} head={args.head} "
        f"lower_body={args.lower_body} gait={args.gait} "
        f"no_robot={args.no_robot}"
    )
    if args.hand == 'dex3' and args.vr_input == 'hand':
        logger_mp.warning(
            "[main] DEX3 + vr-input=hand: hand-tracking 경로는 25 landmark 입력이 "
            "필요해 현재 안전 default(release pose)만 publish합니다. controller 모드 사용을 권장합니다.")

    specs     = get_worker_specs(args, events, locks, shm_names)
    processes = launch_processes(specs)

    wait_for_shutdown(events['shutdown'])
    cleanup(processes, managers)
    # PIDFILE 은 atexit 으로 이미 제거 등록되어 있음. 명시적으로 한 번 더.
    try:
        os.remove(PIDFILE)
    except FileNotFoundError:
        pass
    sys.exit(0)


if __name__ == '__main__':
    main()
