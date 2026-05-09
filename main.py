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
    RECORD_TASK_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_MODE_LAYOUT, CURRENT_MODE_LAYOUT,
    LEFT_TOUCH_SENSOR_LAYOUT, RIGHT_TOUCH_SENSOR_LAYOUT,
    WORKER_FREQ, GR00T_TASK_LAYOUT, ROBOT_OBS, ROBOT_ACTION,
    MASK_CONTROL_LAYOUT, DEPTH_MAP, MODE_MAPPING
)
from gui.ui_launcher import run_ui
from workers.worker_record import worker_record
from workers.worker_vr import worker_vr
from workers.keyboard_listener import keyboard_listener

# Configure multiprocessing
# mp.set_start_method('spawn')  # Uncomment if needed for Windows compatibility
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = " ".join([
    "--ignore-gpu-blocklist",   # 블랙리스트 무시
    "--enable-webgl",
    "--enable-gpu-rasterization",
    "--use-gl=desktop",         # 안 되면 'egl' 또는 'swiftshader' 로 바꿔 테스트
    "--no-sandbox",
])

# Logging
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

# Shared-memory configurations: layout constant, shm name, corresponding lock key
SHM_CONFIG = {
    'camera_shm':          (CAMERA,                    'camera_shm',          'camera_lock'),
    'television_shm':      (TELEVISION,                'television_shm',      'television_lock'),
    'aruco_shm':           (ARUCO_MARKERS,             'aruco_shm',           'aruco_lock'),
    'workspace_mask_shm':  (WORKSPACE_MASK,            'workspace_mask_shm',  'workspace_mask_lock'),
    'record_task_shm':     (RECORD_TASK_LAYOUT,        'record_task_shm',     'record_lock'),
    'record_episode_shm':  (RECORD_EPISODE_LAYOUT,     'record_episode_shm',  'record_lock'),
    'record_mode_shm':     (RECORD_MODE_LAYOUT,        'record_mode_shm',     'record_lock'),
    'current_mode_shm':    (CURRENT_MODE_LAYOUT,       'current_mode_shm',    'record_lock'),
    'left_touch_shm':      (LEFT_TOUCH_SENSOR_LAYOUT,  'left_touch_shm',      'left_touch_lock'),
    'right_touch_shm':     (RIGHT_TOUCH_SENSOR_LAYOUT, 'right_touch_shm',     'right_touch_lock'),
    'freq_shm':            (WORKER_FREQ,               'freq_shm',            'record_lock'),
    'gr00t_shm':           (GR00T_TASK_LAYOUT,         'gr00t_shm',           'gr00t_lock'),
    'robot_obs_shm':       (ROBOT_OBS,                 'robot_obs_shm',       'robot_obs_lock'),
    'robot_action_shm':    (ROBOT_ACTION,              'robot_action_shm',    'robot_action_lock'),
    'mask_control_shm':    (MASK_CONTROL_LAYOUT,       'mask_control_shm',    'record_lock'),
    'depth_map_shm':       (DEPTH_MAP,                 'depth_map_shm',       'depth_map_lock'),
}


def create_events():
    """Initialize and return a dict of multiprocessing Events."""
    return {
        'set_start': Event(),
        'shutdown':  Event(),
        'go_home':   Event(),
        'emergency': Event(),
        'set_g1':    Event(),
        'set_hand':  Event(),
    }


def create_locks():
    """Initialize and return a dict of multiprocessing Locks."""
    return {
        'robot_data_lock':     Lock(),
        'robot_lock':          Lock(),
        'camera_lock':         Lock(),
        'television_lock':     Lock(),
        'aruco_lock':          Lock(),
        'workspace_mask_lock': Lock(),
        'record_lock':         Lock(),
        'left_touch_lock':     Lock(),
        'right_touch_lock':    Lock(),
        'freq_lock':           Lock(),
        'visual_lock':         Lock(),
        'gr00t_lock':          Lock(),
        'robot_obs_lock':      Lock(),
        'robot_action_lock':   Lock(),
        'depth_map_lock':      Lock(),
    }


def create_shm_managers(locks):
    """Create SharedMemoryManager instances for all configured segments."""
    managers = {}
    for key, (layout, name, lock_key) in SHM_CONFIG.items():
        managers[key] = SharedMemoryManager(layout, locks[lock_key], name)
    return managers


def get_shm_names():
    """Return a dict of shared memory names."""
    return {key: name for key, (_, name, _) in SHM_CONFIG.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['teleop', 'gr00t', 'gr00t_zed'],
        default='teleop',
        help='Select worker set to run'
    )
    return parser.parse_args()


def get_worker_specs(mode, events, locks, shm_names):
    """Return a list of worker specs (target, args, name) based on the selected mode."""
    specs = []

    # 모드 정보를 공유 메모리에 저장
    current_mode_shm = SharedMemoryManager(CURRENT_MODE_LAYOUT, locks["record_lock"], shm_names["current_mode_shm"])
    mode_int = MODE_MAPPING.get(mode, 0)
    current_mode_shm.write_data(mode=np.int32(mode_int))
    current_mode_shm.worker_close()

    # 기존 record_mode_shm 초기화
    record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, locks["record_lock"], shm_names["record_mode_shm"])
    record_mode_shm.write_data(
        start=np.bool_(False),
        done=np.bool_(False),
        reset=np.bool_(False),
        replay=np.bool_(False),
        home=np.bool_(False),
        deploy=np.bool_(False)
    )
    record_mode_shm.worker_close()

    ###### 양손 Inspire hand 텔레옵 (수집)
    if mode == 'teleop':
        from workers.worker_g1_ctrl import worker_g1_ctrl
        from workers.worker_g1_ik import worker_g1_ik
        from workers.worker_hand_ctrl import worker_hand_ctrl
        from workers.worker_zed import worker_zed
        from workers.worker_camera import worker_camera
        from workers.worker_hand_dds import worker_hand_r_dds, worker_hand_l_dds

        specs += [
            {'target': worker_record,    'args': (events, shm_names, locks),       'name': 'WORKER_RECORD'},
            {'target': worker_g1_ctrl,   'args': (events, shm_names, locks, mode), 'name': 'worker_g1_ctrl'},
            {'target': worker_g1_ik,     'args': (events, shm_names, locks, mode), 'name': 'worker_g1_ik'},
            {'target': worker_zed,       'args': (events, shm_names, locks),       'name': 'WORKER_ZED'},
            {'target': worker_camera,    'args': (events, shm_names, locks),       'name': 'WORKER_Realsense'},
            {'target': worker_hand_ctrl, 'args': (events, shm_names, locks),       'name': 'WORKER_HAND'},
            {'target': worker_hand_r_dds, 'args': ('192.168.123.210', 'r', 'Right-hand process', shm_names, locks), 'name': 'WORKER_HAND_R_DDS'},
            {'target': worker_hand_l_dds, 'args': ('192.168.123.211', 'l', 'Left-hand process',  shm_names, locks), 'name': 'WORKER_HAND_L_DDS'},
        ]

    ### gr00t deploy - RealSense 단독 + 양손 Inspire
    elif mode == 'gr00t':
        from workers.worker_g1_ctrl import worker_g1_ctrl
        from workers.worker_g1_ik import worker_g1_ik
        from workers.worker_hand_ctrl import worker_hand_ctrl
        from workers.worker_camera import worker_camera
        from workers.worker_hand_dds import worker_hand_r_dds, worker_hand_l_dds

        specs += [
            {'target': worker_g1_ctrl,   'args': (events, shm_names, locks, mode), 'name': 'worker_g1_ctrl'},
            {'target': worker_g1_ik,     'args': (events, shm_names, locks, mode), 'name': 'worker_g1_ik'},
            {'target': worker_hand_ctrl, 'args': (events, shm_names, locks),       'name': 'WORKER_HAND'},
            {'target': worker_camera,    'args': (events, shm_names, locks),       'name': 'WORKER_Realsense'},
            {'target': worker_hand_r_dds, 'args': ('192.168.123.210', 'r', 'Right-hand process', shm_names, locks), 'name': 'WORKER_HAND_R_DDS'},
            {'target': worker_hand_l_dds, 'args': ('192.168.123.211', 'l', 'Left-hand process',  shm_names, locks), 'name': 'WORKER_HAND_L_DDS'},
        ]

    ### gr00t deploy - ZED + RealSense + 양손 Inspire
    elif mode == 'gr00t_zed':
        from workers.worker_g1_ctrl import worker_g1_ctrl
        from workers.worker_g1_ik import worker_g1_ik
        from workers.worker_hand_ctrl import worker_hand_ctrl
        from workers.worker_zed import worker_zed
        from workers.worker_camera import worker_camera
        from workers.worker_hand_dds import worker_hand_r_dds, worker_hand_l_dds

        specs += [
            {'target': worker_g1_ctrl,   'args': (events, shm_names, locks, mode), 'name': 'worker_g1_ctrl'},
            {'target': worker_g1_ik,     'args': (events, shm_names, locks, mode), 'name': 'worker_g1_ik'},
            {'target': worker_hand_ctrl, 'args': (events, shm_names, locks),       'name': 'WORKER_HAND'},
            {'target': worker_zed,       'args': (events, shm_names, locks),       'name': 'WORKER_ZED'},
            {'target': worker_camera,    'args': (events, shm_names, locks),       'name': 'WORKER_Realsense'},
            {'target': worker_hand_r_dds, 'args': ('192.168.123.210', 'r', 'Right-hand process', shm_names, locks), 'name': 'WORKER_HAND_R_DDS'},
            {'target': worker_hand_l_dds, 'args': ('192.168.123.211', 'l', 'Left-hand process',  shm_names, locks), 'name': 'WORKER_HAND_L_DDS'},
        ]

    common_workers = [
        # VR 입력 (Vuer Quest3)
        {'target': worker_vr,         'args': (events, shm_names, locks), 'name': 'WORKER_VR'},
        # GUI
        {'target': run_ui,            'args': (events, shm_names, locks), 'name': 'UI'},
        # 키보드 입력
        {'target': keyboard_listener, 'args': (events,),                  'name': 'KEYBOARD'},
    ]

    specs += common_workers

    return specs


AFFINITY = {
    'worker_g1_ctrl':       {20, 21},
    'WORKER_ZED':           {22},
    'WORKER_VR':            {23},
    'WORKER_HAND_R_DDS':    {18},
    'WORKER_HAND_L_DDS':    {19},
}


def launch_processes(specs):
    """Start all processes defined in specs."""
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
    """Block until shutdown event is set."""
    try:
        while not event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        event.set()


def cleanup(processes, managers):
    """Join processes and clean up shared memory."""
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

    specs     = get_worker_specs(args.mode, events, locks, shm_names)
    processes = launch_processes(specs)

    wait_for_shutdown(events['shutdown'])
    cleanup(processes, managers)
    sys.exit(0)


if __name__ == '__main__':
    main()
