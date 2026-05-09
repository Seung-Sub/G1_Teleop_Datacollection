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
    CAMERA, TELEVISION, ARUCO_MARKERS, WORKSPACE_MASK,
    RECORD_TASK_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_MODE_LAYOUT,
    LEFT_TOUCH_SENSOR_LAYOUT, RIGHT_TOUCH_SENSOR_LAYOUT,
    WORKER_FREQ, GR00T_TASK_LAYOUT, ROBOT_OBS, ROBOT_ACTION,
    MASK_CONTROL_LAYOUT, DEPTH_MAP, TELEOP_CONFIG, QUEST_CONTROLLER,
    HAND_MAPPING, CAMERA_MAPPING, VR_INPUT_MAPPING,
)
from gui.ui_launcher import run_ui
from workers.worker_record import worker_record
from workers.worker_vr import worker_vr
from workers.keyboard_listener import keyboard_listener

os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = " ".join([
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--enable-gpu-rasterization",
    "--use-gl=desktop",
    "--no-sandbox",
])

# Logging
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

# Shared-memory configurations: layout constant, shm name, lock key
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
    parser.add_argument('--camera',   choices=list(CAMERA_MAPPING.keys()),
                        default='zed',     help='Egocentric camera source')
    parser.add_argument('--vr-input', dest='vr_input',
                        choices=list(VR_INPUT_MAPPING.keys()),
                        default='hand',
                        help='Quest3 input mode (hand tracking vs motion controller)')
    # Inspire thumb 사전 자세 (vr_input=controller 일 때만 사용; 손가락 4개는 trigger로 토글)
    # 값 범위: 0.0(굽힘/안쪽) ~ 1.0(펼침/바깥쪽). 물체에 따라 잡기 편한 자세를 사전 설정.
    parser.add_argument('--thumb-bend', dest='thumb_bend', type=float, default=0.5,
                        help='Inspire thumb bend angle (controller mode only, 0..1)')
    parser.add_argument('--thumb-yaw',  dest='thumb_yaw',  type=float, default=0.5,
                        help='Inspire thumb yaw   angle (controller mode only, 0..1)')
    return parser.parse_args()


def write_teleop_config(locks, shm_names, args):
    """Write the chosen options into TELEOP_CONFIG SHM (info-only)."""
    cfg = SharedMemoryManager(TELEOP_CONFIG, locks["record_lock"], shm_names["teleop_config_shm"])
    cfg.write_data(
        hand_type =np.int32(HAND_MAPPING[args.hand]),
        camera_type=np.int32(CAMERA_MAPPING[args.camera]),
        vr_input  =np.int32(VR_INPUT_MAPPING[args.vr_input]),
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
    """Build the worker spec list for the chosen --hand/--camera/--vr-input."""
    specs = []

    write_teleop_config(locks, shm_names, args)

    # ---- Robot core (G1 + IK) ---------------------------------------
    from workers.worker_g1_ctrl import worker_g1_ctrl
    from workers.worker_g1_ik   import worker_g1_ik
    specs += [
        {'target': worker_g1_ctrl, 'args': (events, shm_names, locks, 'teleop'),
         'name': 'worker_g1_ctrl'},
        {'target': worker_g1_ik,   'args': (events, shm_names, locks, args.vr_input),
         'name': 'worker_g1_ik'},
    ]

    # ---- Hand --------------------------------------------------------
    if args.hand == 'inspire':
        from workers.worker_hand_ctrl import worker_hand_ctrl
        from workers.worker_hand_dds  import worker_hand_r_dds, worker_hand_l_dds
        specs += [
            {'target': worker_hand_ctrl,
             'args':   (events, shm_names, locks, args.vr_input, args.thumb_bend, args.thumb_yaw),
             'name':   'WORKER_HAND'},
            {'target': worker_hand_r_dds, 'args': ('192.168.123.210', 'r', 'Right-hand process', shm_names, locks),
             'name': 'WORKER_HAND_R_DDS'},
            {'target': worker_hand_l_dds, 'args': ('192.168.123.211', 'l', 'Left-hand process',  shm_names, locks),
             'name': 'WORKER_HAND_L_DDS'},
        ]
    else:
        raise ValueError(f"Unsupported hand hardware: {args.hand}")

    # ---- Camera (ZED OR RealSense; never both) -----------------------
    if args.camera == 'zed':
        from workers.worker_zed import worker_zed
        specs += [{'target': worker_zed, 'args': (events, shm_names, locks),
                   'name': 'WORKER_ZED'}]
    elif args.camera == 'realsense':
        from workers.worker_camera import worker_camera
        specs += [{'target': worker_camera, 'args': (events, shm_names, locks),
                   'name': 'WORKER_Realsense'}]
    else:
        raise ValueError(f"Unsupported camera: {args.camera}")

    # ---- Recording (always-on) --------------------------------------
    specs += [{'target': worker_record, 'args': (events, shm_names, locks),
               'name': 'WORKER_RECORD'}]

    # ---- Common: VR / GUI / keyboard --------------------------------
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
    events    = create_events()
    locks     = create_locks()
    shm_names = get_shm_names()
    managers  = create_shm_managers(locks)

    logger_mp.info(f"[main] hand={args.hand} camera={args.camera} vr_input={args.vr_input}")

    specs     = get_worker_specs(args, events, locks, shm_names)
    processes = launch_processes(specs)

    wait_for_shutdown(events['shutdown'])
    cleanup(processes, managers)
    sys.exit(0)


if __name__ == '__main__':
    main()
