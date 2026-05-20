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
    HAND_MAPPING, CAMERA_MAPPING, VR_INPUT_MAPPING, WAIST_MAPPING, HEAD_MAPPING,
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
    # Inspire thumb 사전 자세 (vr_input=controller 일 때만 사용; 손가락 4개는 trigger로 토글)
    # 값 범위: 0.0(굽힘/안쪽) ~ 1.0(펼침/바깥쪽). 물체에 따라 잡기 편한 자세를 사전 설정.
    parser.add_argument('--thumb-bend', dest='thumb_bend', type=float, default=0.5,
                        help='Inspire thumb bend angle (controller mode only, 0..1)')
    parser.add_argument('--thumb-yaw',  dest='thumb_yaw',  type=float, default=0.5,
                        help='Inspire thumb yaw   angle (controller mode only, 0..1)')
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
        hand_type  =np.int32(HAND_MAPPING[args.hand]),
        camera_type=np.int32(CAMERA_MAPPING.get(cam_type_str, CAMERA_MAPPING['none'])),
        vr_input   =np.int32(VR_INPUT_MAPPING[args.vr_input]),
        waist_mode =np.int32(WAIST_MAPPING[args.waist]),
        head_mode  =np.int32(HEAD_MAPPING[args.head]),
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
        specs += [
            {'target': worker_g1_ctrl,
             'args':   (events, shm_names, locks, 'teleop', args.head),
             'name':   'worker_g1_ctrl'},
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
                 'args':   (events, shm_names, locks, args.hand, args.vr_input, args.thumb_bend, args.thumb_yaw),
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
                 'args':   (events, shm_names, locks, args.hand, args.vr_input, args.thumb_bend, args.thumb_yaw),
                 'name':   'WORKER_HAND'},
                # DEX3 는 별도 터치센서 DDS 가 없다 — worker_hand_*_dds 미동작
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
    specs += [
        {'target': worker_vr,         'args': (events, shm_names, locks, args.vr_input),
         'name': 'WORKER_VR'},
        {'target': run_ui,            'args': (events, shm_names, locks),
         'name': 'UI'},
        {'target': keyboard_listener, 'args': (events,),
         'name': 'KEYBOARD'},
    ]

    return specs


AFFINITY = {
    'worker_g1_ctrl':    {20, 21},
    'WORKER_ZED':        {22},
    'WORKER_VR':         {23},
    'WORKER_HAND_R_DDS': {18},
    'WORKER_HAND_L_DDS': {19},
}


def launch_processes(specs):
    processes = []
    for spec in specs:
        p = Process(target=spec['target'], args=spec['args'], daemon=True, name=spec.get('name'))
        p.start()
        cpus = AFFINITY.get(p.name)
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
    for p in processes:
        p.join(timeout=2)

    for mgr in managers.values():
        if getattr(mgr, '_owner', False):
            try:
                mgr.main_unlink()
            except FileNotFoundError:
                pass
        else:
            mgr.worker_close()


def main():
    args      = parse_args()

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
    sys.exit(0)


if __name__ == '__main__':
    main()
