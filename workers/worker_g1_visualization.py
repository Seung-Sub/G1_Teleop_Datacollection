import os
import time
import numpy as np
import logging

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import WORKER_FREQ, ROBOT_OBS


from g1_control.g1_visualize_whole import G1_Visualization

import pandas as pd

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


def expand_hand_joints(hand6: np.ndarray) -> np.ndarray:
    pinky_p, ring_p, middle_p, index_p, thumb_pitch_p, thumb_yaw_p = hand6
    
    # 엄지 mimic
    thumb_i = 1.6 * thumb_pitch_p      # thumb_intermediate_joint
    thumb_d = 2.4 * thumb_pitch_p      # thumb_distal_joint
    
    # 나머지 손가락 mimic(1:1)
    index_i  = index_p                 # index_intermediate_joint
    middle_i = middle_p                # middle_intermediate_joint
    pinky_i  = pinky_p                 # pinky_intermediate_joint
    ring_i   = ring_p                  # ring_intermediate_joint
    
    # 지정된 최종 순서대로 배열 생성
    hand12 = np.array([
        index_p,  index_i,
        middle_p, middle_i,
        pinky_p,  pinky_i,
        ring_p,   ring_i,
        thumb_yaw_p,
        thumb_pitch_p,
        thumb_i,
        thumb_d
    ], dtype=float)
    
    return hand12


def worker_g1_visualization(shared_event, shm_name, shared_lock):



    #Set SharedMemory
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])
    robot_obs_shm = SharedMemoryManager(ROBOT_OBS, shared_lock["robot_obs_lock"], shm_name["robot_obs_shm"])

    #Set loop frequency
    freq = 50.0
    period_ns = int(1e9 / freq)
    next_time = time.perf_counter_ns()
    last_time = next_time
    count = 0

    # G1/Hand 컨트롤러를 아직 생성하지 않은 상태를 추적하는 플래그
    g1_initialized   = False


    # ── 2) 최상위 루프: shutdown이 set 되기 전까지 계속 살아 있음 ───────────
    while not shared_event['shutdown'].is_set():

        # ── 2-1) G1과 Hand 연결을 둘 다 확인 → 둘 다 눌릴 때까지 대기 ─────────
        logger_mp.info("[Meshcat] Waiting for G1/Hand connections (both).")
        while True:
            # (1) Emergency나 Shutdown이 먼저 들어온 경우 즉시 종료
            if shared_event['emergency'].is_set():
                logger_mp.warning("[Meshcat] Emergency received before G1/Hand. Exiting.")
                break
            if shared_event['shutdown'].is_set():
                logger_mp.info("[Meshcat] Shutdown received before G1/Hand. Exiting.")
                break

            # (2) G1 연결 신호가 들어왔고, 아직 초기화되지 않았다면 초기화
            if shared_event['set_g1'].is_set() and not g1_initialized:
                try:
                    # 컨트롤러 인스턴스 생성
                    g1_vis = G1_Visualization()

                    g1_initialized = True
                    logger_mp.info("[Meshcat] G1 controller initialized.")

                except Exception as e:
                    logger_mp.error(f"[Meshcat] G1 init failed: {e}")
                    shared_event['emergency'].set()
                    break

            # (4) 둘 다 초기화되면 이 연결 대기 루프를 벗어남
            if g1_initialized :
                logger_mp.info("[Meshcat] G1 connected. Proceeding to START wait.")
                break

            # (5) 아직 조건 만족 못 했으면 짧게 sleep
            time.sleep(0.01)

        # 2-1 루프 종료 시점: 
        #    (A) 둘 다 connected 되었거나, (B) emergency/shutdown이 set되었거나
        if shared_event['shutdown'].is_set():
            # Shutdown으로 인해 빠져나온 경우
            break
        if shared_event['emergency'].is_set():
            # Emergency로 인해 빠져나온 경우
            break

        # ── 2-2) 이제 “두 연결 완료” 후, Start 버튼을 기다리는 단계 ─────────────────
        logger_mp.info("[Meshcat] Waiting for START signal (after G1/Hand).")
        while True:
            # (1) Emergency 혹은 Shutdown이 먼저 들어오면 즉시 종료
            if shared_event['emergency'].is_set():
                logger_mp.warning("[Meshcat] Emergency received before START. Exiting.")
                break
            if shared_event['shutdown'].is_set():
                logger_mp.info("[Meshcat] Shutdown received before START. Exiting.")
                break

            # (2) Start 버튼이 눌린 경우에만 다음 단계로 넘어감
            if shared_event['set_start'].is_set():
                logger_mp.info("[Meshcat] START received. Entering run loop.")
                break

            time.sleep(0.01)

        # 2-2 루프 종료 시점: (A) Start 눌림, (B) Emergency, (C) Shutdown 중 하나
        if shared_event['shutdown'].is_set():
            break
        if shared_event['emergency'].is_set():
            break


        while shared_event['set_start'].is_set() and not shared_event['shutdown'].is_set() and not shared_event['emergency'].is_set():
                  
            now = time.perf_counter_ns()
            delta_ms = (now - last_time) / 1e6
            actual_hz = 1.0 / ((now - last_time) / 1e9)

            obs_data = robot_obs_shm.read_data()
            obs_leg = obs_data["obs_leg"]
            obs_waist = obs_data["obs_waist"]
            obs_head = obs_data["obs_head"]
            obs_arm = obs_data["obs_arm"]
            obs_hand = obs_data["obs_hand"]

            g1_vis.visual_leg_q(
                left=obs_leg[:6],
                right=obs_leg[6:]
                )

            g1_vis.visual_waist_q(
                waist = obs_waist
            )

            # 양팔 동시에(각 7개 값)
            g1_vis.visual_arm_q(
                left =obs_arm[:7],
                right=obs_arm[7:]
            )

            hand_L12 = expand_hand_joints(obs_hand[:6])     # 왼손
            hand_R12 = expand_hand_joints(obs_hand[6:])   # 오른손
            g1_vis.visual_hand_q(
                right=hand_R12,
                left=hand_L12
            )

            g1_vis.visual_head_q(obs_head)

            last_time = now
            count += 1

            next_time += period_ns
            sleep_ns = next_time - time.perf_counter_ns()
            if sleep_ns > 0:
                if sleep_ns > 1_000_000:
                    time.sleep((sleep_ns - 500_000) / 1e9)
                while time.perf_counter_ns() < next_time:
                    pass
            else:
                next_time = time.perf_counter_ns()


        if shared_event['shutdown'].is_set():
            logger_mp.info("[Meshcat] Shutdown signal received during run. Exiting.")
            break
        if shared_event['emergency'].is_set():
            logger_mp.warning("[Meshcat] Emergency signal received during run. Exiting.")
            break

        # ‘Start’ 버튼이 다시 눌려 “Pause” 상태가 되었을 때
        if not shared_event['set_start'].is_set():
            logger_mp.info("[Meshcat] PAUSE signal received. Returning to wait-for-start...")
            # (이 시점에서 최상위 루프 while문 맨 위로 올라가, 다시 대기 모드로 진입)


    logger_mp.info("[Meshcat] 종료 신호 수신. 종료합니다.")
    freq_shm.worker_close()
    robot_obs_shm.worker_close()
    # if shared_event['set_g1'].is_set():
    #     g1_ik_solver.close_vis()    Meshcat