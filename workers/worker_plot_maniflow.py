import os
import time

import numpy as np

from collections import deque

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import ROBOT_ACTION, ROBOT_OBS


# import logging_mp
# logger_mp = logging_mp.get_logger(__name__)

def worker_plot(shared_event, shm_name, shared_lock):
    import matplotlib 
    matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt 

    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['savefig.dpi'] = 150    # 저장 시 해상도도 함께 설정 가능

    robot_action_shm = SharedMemoryManager(ROBOT_ACTION, shared_lock["robot_action_lock"], shm_name["robot_action_shm"])
    robot_obs_shm = SharedMemoryManager(ROBOT_OBS, shared_lock["robot_obs_lock"], shm_name["robot_obs_shm"])

    # --- Matplotlib 설정 ---
    plt.ion()
    # qpos/action: 3개 그룹, hand: 2개 그룹
    q_groups = {
        "waist":     slice(0, 3),
        "left_arm":  slice(5, 12),
        "right_arm": slice(12, 19),
    }
    h_groups = {
        "left_hand":  slice(0, 6),
        "right_hand": slice(6, 12),
    }

    # 전체 Figure: 세로 2행 (qpos/action, hand), 가로 최대 그룹 수
    n_q = len(q_groups)
    n_h = len(h_groups)
    fig_q, axes_q = plt.subplots(1, n_q, figsize=(4*n_q, 4), sharex=True)
    fig_h, axes_h = plt.subplots(1, n_h, figsize=(4*n_h, 6), sharex=True)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 라인 객체 생성
    lines_q, lines_a = {}, {}
    for ix, (name, sl) in enumerate(q_groups.items()):
        ax = axes_q[ix] if n_q>1 else axes_q
        ax.set_title(f"{name}: qpos & action")
        lines_q[name], lines_a[name] = [], []
        for idx, d in enumerate(range(sl.start, sl.stop)):
            c = color_cycle[idx % len(color_cycle)]  # 같은 인덱스엔 같은 색
            # qpos: 실선
            ln_q, = ax.plot([], [], color=c, label=f"qpos d{d}")
            # action: 점선
            ln_a, = ax.plot([], [], '--', color=c, label=f"action d{d}")
            lines_q[name].append(ln_q)
            lines_a[name].append(ln_a)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize="small")

    lines_hq, lines_ha = {}, {}
    for ix, (name, sl) in enumerate(h_groups.items()):
        ax = axes_h[ix] if n_h > 1 else axes_h
        ax.set_title(f"{name}: hand_qpos & hand_action")
        lines_hq[name], lines_ha[name] = [], []
        for idx, d in enumerate(range(sl.start, sl.stop)):
            c = color_cycle[idx % len(color_cycle)]
            ln_hq, = ax.plot([], [], color=c, label=f"qpos d{d}")
            ln_ha, = ax.plot([], [], '--', color=c, label=f"action d{d}")
            lines_hq[name].append(ln_hq)
            lines_ha[name].append(ln_ha)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize="small")

    # 데이터 버퍼 (5초치 = 50Hz * 5s)
    maxlen = 50
    times = deque(maxlen=maxlen)
    buf_qpos, buf_act = deque(maxlen=maxlen), deque(maxlen=maxlen)
    buf_hqpos, buf_ha = deque(maxlen=maxlen), deque(maxlen=maxlen)

    start_t = time.perf_counter()
    freq = 50.0
    period_ns = int(1e9 / freq)
    next_ns = time.perf_counter_ns()

    while not shared_event['shutdown'].is_set():
        now_ns = time.perf_counter_ns()
        # 데이터 읽기
        robot_obs = robot_obs_shm.read_data()
        robot_action = robot_action_shm.read_data()

        obs_leg = robot_obs["obs_leg"]
        obs_waist = robot_obs["obs_waist"]
        obs_head = robot_obs["obs_head"]
        obs_arm = robot_obs["obs_arm"]
        obs_hand = robot_obs["obs_hand"]

        action_leg = robot_action["action_leg"]
        action_waist = robot_action["action_waist"]
        action_head = robot_action["action_head"]
        action_arm = robot_action["action_arm"]
        action_hand = robot_action["action_hand"]

        qpos = np.concatenate((obs_waist,obs_head,obs_arm))
        action = np.concatenate((action_waist,action_head,action_arm))
        hqp = obs_hand
        ha = action_hand



        # 타임스탬프
        t = time.perf_counter() - start_t
        times.append(t)
        buf_qpos.append(qpos)
        buf_act.append(action)
        buf_hqpos.append(hqp)
        buf_ha.append(ha)

        # qpos/action 업데이트
        for name, sl in q_groups.items():
            for idx, d in enumerate(range(sl.start, sl.stop)):
                ys = [qp[d] for qp in buf_qpos]
                lines_q[name][idx].set_data(times, ys)
                ys2 = [ac[d] for ac in buf_act]
                lines_a[name][idx].set_data(times, ys2)
        # hand 업데이트
        for name, sl in h_groups.items():
            for idx, d in enumerate(range(sl.start, sl.stop)):
                ys = [hq[d] for hq in buf_hqpos]
                lines_hq[name][idx].set_data(times, ys)
                ys2 = [ha_[d] for ha_ in buf_ha]
                lines_ha[name][idx].set_data(times, ys2)

        # 축 재조정 & 그리기
        for ax in axes_q.flatten(): 
            ax.relim(); ax.autoscale_view()
        fig_q.canvas.draw_idle()
        for ax in axes_h.flatten(): 
            ax.relim(); ax.autoscale_view()
        fig_h.canvas.draw_idle()

        plt.pause(0.001)

        # 주기 제어
        next_ns += period_ns
        sleep_ns = next_ns - time.perf_counter_ns()
        if sleep_ns > 0:
            if sleep_ns > 1_000_000:
                time.sleep((sleep_ns - 500_000)/1e9)
            while time.perf_counter_ns() < next_ns:
                pass
        else:
            next_ns = time.perf_counter_ns()

    # logger_mp.info("[Plot] 종료 신호 수신. 종료합니다.")
    robot_obs_shm.worker_close()
    robot_action_shm.worker_close()
