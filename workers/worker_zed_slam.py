import os
import time

import numpy as np
import math
from collections import deque

import pyzed.sl as sl

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import ROBOT_AMO_INPUT
from utils.rate import Rate

import logging_mp
logger_mp = logging_mp.get_logger(__name__)



# ===== Filtering / Rate limiting helpers =====
def exp_smooth(prev, x, tau, dt):
    """연속시간 1차 저역통과 y += (1-a)*(x-y), a=exp(-dt/tau)"""
    if prev is None:
        return x
    a = math.exp(-dt / max(tau, 1e-6))
    return a*prev + (1.0 - a)*x

def slerp(q0, q1, t):
    """쿼터니언 (x,y,z,w) slerp"""
    q0 = np.array(q0, dtype=float); q1 = np.array(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # 최단호 보장
        q1 = -q1; dot = -dot
    DOT_TH = 0.9995
    if dot > DOT_TH:  # 거의 같은 방향이면 lerp
        out = q0 + t*(q1 - q0)
        return out / np.linalg.norm(out)
    theta0 = math.acos(dot); sin0 = math.sin(theta0)
    theta = theta0 * t; sinT = math.sin(theta)
    s0 = math.cos(theta) - dot * sinT / sin0
    s1 = sinT / sin0
    return s0*q0 + s1*q1

def angle_smooth(prev, meas, tau, dt, max_rate=None):
    """yaw 각도 EMA + 선택적 rate limit (rad/s)"""
    if prev is None:
        return meas
    # unwrap diff to [-pi, pi]
    diff = ( (meas - prev + math.pi) % (2*math.pi) ) - math.pi
    a = math.exp(-dt / max(tau, 1e-6))
    target = prev + (1.0 - a) * diff
    if max_rate is not None:
        max_step = max_rate * dt
        step = max(-max_step, min(max_step, target - prev))
        return prev + step
    return target

def slew_limit(prev, target, max_rate, dt):
    """스칼라/벡터에 대해 d/dt를 |max_rate|로 제한"""
    if prev is None:
        return target
    max_step = max_rate * dt
    return prev + np.clip(np.array(target) - np.array(prev), -max_step, max_step)

def worker_fusion(shared_event, shm_names, shared_lock):

    rate = Rate(30.0)


    # ===== 상수: ZED 비가용 시 더미 출력 =====
    ZERO_PELVIS = np.array([0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 1.0], dtype=float)  # [x,y,z,qx,qy,qz,qw]
    ZERO_VEL    = np.array([0.0, 0.0, 0.0], dtype=float)                         # [vx,vy,yaw_rate]

    # ===== 공유메모리를 가장 먼저 열어, 실패 시에도 바로 쓸 수 있게 함 =====
    robot_amo_input_shm = SharedMemoryManager(
        ROBOT_AMO_INPUT,
        shared_lock["robot_amo_input_lock"],
        shm_names["robot_amo_input_shm"]
    )

    # ===== ZED 카메라 초기화 시도 =====
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.AUTO,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    )

    logger_mp.info("[ZEDFusionWorker] Opening ZED camera...")
    status = zed.open(init_params)
    zed_opened = (status == sl.ERROR_CODE.SUCCESS)
    zed_working = zed_opened  # 이후 단계 실패 시 False로 전환

    if not zed_opened:
        logger_mp.error(f"[ZEDFusionWorker] Camera open failed: {status}. Entering dummy-output mode (no exit).")

    fusion = None
    subscribed = False
    uuid = None

    # ZED가 동작하는 경우에만 나머지 초기화 진행
    if zed_working:
        # shared memory 대신 SDK 퍼블리싱
        comm_params = sl.CommunicationParameters()
        comm_params.set_for_shared_memory()
        zed.start_publishing(comm_params)

        # 워밍업
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            logger_mp.error("[ZEDFusionWorker] Initial grab failed. Falling back to dummy-output mode.")
            zed_working = False

    if zed_working:
        # Positional Tracking
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_imu_fusion = True
        tracking_params.set_gravity_as_origin = True
        err = zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logger_mp.error(f"[ZEDFusionWorker] Positional tracking failed: {err}. Falling back to dummy-output mode.")
            zed_working = False

    if zed_working:
        # Fusion 초기화 및 구독
        camera_info = zed.get_camera_information()
        fusion = sl.Fusion()
        init_fusion_params = sl.InitFusionParameters()
        init_fusion_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        init_fusion_params.coordinate_units = sl.UNIT.METER
        fusion.init(init_fusion_params)
        fusion.enable_positionnal_tracking(sl.PositionalTrackingFusionParameters())

        uuid = sl.CameraIdentifier(camera_info.serial_number)
        st = fusion.subscribe(uuid, comm_params, sl.Transform(0, 0, 0))
        if st != sl.FUSION_ERROR_CODE.SUCCESS:
            logger_mp.error(f"[ZEDFusionWorker] Fusion subscribe failed: {st}. Falling back to dummy-output mode.")
            zed_working = False
        else:
            subscribed = True
            logger_mp.info(f"[ZEDFusionWorker] Subscribed to camera {uuid.serial_number}.")
            logger_mp.info("[ZEDFusionWorker] Positional tracking enabled.")

    # ===== 필터/상태 파라미터 (ZED 동작 시에만 사용) =====
    TAU_POS = 0.35
    TAU_ORI = 0.35
    TAU_VEL = 0.25
    MAX_ACC = 2.0
    MAX_YAWR = 2.5

    f_pos = None
    f_quat = None
    f_vel = np.zeros(2)
    f_yaw = None
    prev_pos_for_deriv = None


    last_t = time.perf_counter()

    camera_pose = sl.Pose() if zed_working else None
    odom_pose = sl.Pose() if zed_working else None
    py_trans = sl.Translation() if zed_working else None

    try:
        while not shared_event['shutdown'].is_set() and not shared_event['emergency'].is_set():
            loop_start = time.perf_counter()
            dt = max(1e-3, loop_start - last_t)

            if not zed_working:
                # ===== ZED가 없을 때: 0값을 계속 퍼블리시 =====
                robot_amo_input_shm.write_data(
                    pelvis_pose=ZERO_PELVIS,
                    vel_command=ZERO_VEL
                )
            else:
                # ===== 정상 경로 =====
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    zed.get_position(odom_pose, sl.REFERENCE_FRAME.WORLD)
                else:
                    logger_mp.warning("[ZEDFusionWorker] Grab failed.")

                # Dummy GNSS 데이터 (기존 로직 유지)
                gnss = sl.GNSSData()
                gnss.ts = sl.get_current_timestamp()
                gnss.set_coordinates(0.0, 0.0, 0.0)
                gnss.position_covariances = [1,0,0, 0,1,0, 0,0,1]
                if fusion is not None:
                    fusion.ingest_gnss_data(gnss)

                if fusion is not None and fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
                    state = fusion.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
                    if state == sl.POSITIONAL_TRACKING_STATE.OK:
                        # --- quaternion (회전벡터 -> quat) ---
                        rot_vec = camera_pose.get_rotation_vector()
                        theta = np.linalg.norm(rot_vec)
                        if theta > 1e-6:
                            axis = rot_vec / theta
                            qw = math.cos(theta / 2.0)
                            q_xyz = axis * math.sin(theta / 2.0)
                        else:
                            qw = 1.0
                            q_xyz = np.zeros(3)
                        quat_meas = (round(q_xyz[0],2), round(q_xyz[1],2), round(q_xyz[2],2), round(qw,2))

                        # --- translation ---
                        trans = camera_pose.get_translation(py_trans)
                        pos_meas = np.array(trans.get())  # [x,y,z]


                        f_pos = exp_smooth(f_pos, pos_meas, TAU_POS, dt)
                        if f_quat is None:
                            f_quat = quat_meas
                        else:
                            a_ori = math.exp(-dt / max(TAU_ORI, 1e-6))
                            f_quat = slerp(f_quat, quat_meas, (1.0 - a_ori))
                            f_quat = f_quat / np.linalg.norm(f_quat)

                        if prev_pos_for_deriv is None:
                            raw_vel = np.array([0.0, 0.0])
                        else:
                            raw_vel = (f_pos[:2] - prev_pos_for_deriv[:2]) / dt
                        prev_pos_for_deriv = f_pos.copy()

                        f_vel = exp_smooth(f_vel, raw_vel, TAU_VEL, dt)
                        f_vel = slew_limit(f_vel, raw_vel, MAX_ACC, dt)

                        euler = camera_pose.get_euler_angles()
                        yaw_meas = float(euler[2])
                        f_yaw = angle_smooth(f_yaw, yaw_meas, TAU_ORI, dt, max_rate=MAX_YAWR)

                        quat_out = tuple(round(v, 2) for v in f_quat)
                        pos_out  = tuple(round(v, 2) for v in f_pos)
                        vel_out  = (round(float(f_vel[0]), 3),
                                    round(float(f_vel[1]), 3),
                                    round(float(f_yaw),   4))

                        pelvis_pose = np.concatenate((np.array(pos_out), np.array(quat_out)))

                        robot_amo_input_shm.write_data(
                            pelvis_pose=pelvis_pose,
                            vel_command=vel_out
                        )
                else:
                    # Fusion 처리 실패시에도 0 출력(안전)
                    robot_amo_input_shm.write_data(
                        pelvis_pose=ZERO_PELVIS,
                        vel_command=ZERO_VEL
                    )

            # 정확한 주기 유지를 위한 sleep
            rate.sleep()

            # ★ dt 기준 업데이트 (누적 방지)
            last_t = loop_start


    except Exception as e:
        logger_mp.error(f"[ZEDFusionWorker] Unexpected exception: {e}")
        shared_event['emergency'].set()

    finally:
        # 안전 정리: 구독이 되었을 때만 unsubscribe
        try:
            if 'subscribed' in locals() and subscribed and (fusion is not None) and (uuid is not None):
                fusion.unsubscribe(uuid)
        except Exception as e:
            logger_mp.warning(f"[ZEDFusionWorker] unsubscribe skip/failed: {e}")

        # 공유메모리 정리
        try:
            robot_amo_input_shm.worker_close()
        except Exception as e:
            logger_mp.warning(f"[ZEDFusionWorker] shm close failed: {e}")

        # 카메라가 실제로 열렸을 때만 close
        try:
            if 'zed_opened' in locals() and zed_opened:
                zed.close()
        except Exception as e:
            logger_mp.warning(f"[ZEDFusionWorker] zed.close() failed: {e}")

        logger_mp.info("[ZEDFusionWorker] Closed and exiting.")
        
if __name__ == "__main__":
    import argparse
    import threading
    import signal
    import sys
    import sharedmemory
    
    parser = argparse.ArgumentParser(description="Run ZED fusion worker standalone.")
    parser.add_argument(
        "--shm-name",
        default=os.environ.get("ROBOT_AMO_INPUT_SHM", "robot_amo_input"),
        help="ROBOT_AMO_INPUT용 공유 메모리 이름 (기본: robot_amo_input 또는 환경변수 ROBOT_AMO_INPUT_SHM)"
    )
    args = parser.parse_args()

    # 이벤트/락/이름 준비
    shutdown_event = threading.Event()
    emergency_event = threading.Event()
    shared_event = {"shutdown": shutdown_event, "emergency": emergency_event}
    shared_lock = {"robot_amo_input_lock": threading.Lock()}
    shm_names = {"robot_amo_input_shm": args.shm_name}

    # SIGINT/SIGTERM에서 정상 종료
    def _handle_signal(signum, frame):
        logger_mp.info(f"[Main] Caught signal {signum}. Requesting shutdown...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            # 일부 OS/스레드 환경에서 SIGTERM 설정이 불가할 수 있음
            pass

    logger_mp.info("[Main] Starting worker_fusion...")
    try:
        worker_fusion(shared_event, shm_names, shared_lock)
    except KeyboardInterrupt:
        logger_mp.info("[Main] KeyboardInterrupt received. Shutting down...")
        shutdown_event.set()
    except Exception as e:
        logger_mp.error(f"[Main] Unhandled exception: {e}")
        emergency_event.set()
        sys.exit(1)
    finally:
        logger_mp.info("[Main] Exited.")