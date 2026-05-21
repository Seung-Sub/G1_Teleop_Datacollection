"""worker_vr: pump Quest3 input (hand-tracking OR motion controller) into SHM.

vr_input = "hand"
    -> tv_wrapper.get_data_with_segments() -> TELEVISION SHM
       (head_rmat, left/right wrist mat, hand keypoints, right distal/proximal)
    -> 'home' 트리거 시 left/right wrist의 translation 을 home 기준 상대값으로 보정.

vr_input = "controller"
    -> tv_wrapper.get_controller_data() ->
       TELEVISION SHM (head_rmat, left/right_wrist_mat = controller pose,
                       hand keypoints는 0으로 채움) and
       QUEST_CONTROLLER SHM (per-side trigger/squeeze/thumbstick/buttons + connected).
    -> 절대 좌표 그대로 적재한다. clutch (grip 누른 동안만 추종) 와 home rebase는
       worker_g1_ik (Phase 1-C) 에서 처리한다.
"""
import os
import time
import numpy as np

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    TELEVISION, RECORD_MODE_LAYOUT, WORKER_FREQ, QUEST_CONTROLLER,
)

from open_television.tv_wrapper import TeleVisionWrapper

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


def worker_vr(shared_event, shm_name, shared_lock, vr_input="hand"):
    # Part5: 표시 경로 정합 — Vuer 가 ego 카메라 (rs_ego_shm, CAMERA_VIEW) 를 attach.
    # main.py 가 owner-create. cameras=[] (카메라 없음) 시 'rs_ego_shm' 키 미존재
    # → TeleVision 측 SharedMemoryManager 가 자동으로 owner-create (빈 frame 만) 후
    # main_image_* 의 frame_left.any()==0 가드로 표시 skip. 즉 worker_vr 자체는 그대로
    # 동작 (controller pose 채널은 카메라 유무와 독립).
    television_shm       = SharedMemoryManager(TELEVISION,         shared_lock["television_lock"],       shm_name["television_shm"])
    record_mode_shm      = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"],           shm_name["record_mode_shm"])
    freq_shm             = SharedMemoryManager(WORKER_FREQ,        shared_lock["freq_lock"],             shm_name["freq_shm"])
    quest_controller_shm = SharedMemoryManager(QUEST_CONTROLLER,   shared_lock["quest_controller_lock"], shm_name["quest_controller_shm"])

    freq      = 50.0
    period_ns = int(1e9 / freq)
    next_time = time.perf_counter_ns()
    last_time = next_time

    img_shape  = (640, 480, 3)
    # Part5: TeleVisionWrapper 가 직접 attach 할 SHM 이름 — ego role 의 rs_ego_shm 우선,
    # 없으면 legacy 'camera_shm' fallback (cameras=[] 일 때).
    camera_shm_name = shm_name.get("rs_ego_shm", shm_name.get("camera_shm", "rs_ego_shm"))
    tv_wrapper = TeleVisionWrapper(True, img_shape, camera_shm_name,
                                   vr_input=vr_input,
                                   camera_shm_key=camera_shm_name)

    home_left_wrist  = None
    home_right_wrist = None

    zero_keypoints = np.zeros((5, 3), dtype=np.float64)

    # Record-control controller mapping (controller mode 만 해당):
    #   Left X  (left_buttons[0]  aButton) rising-edge → record_mode.start toggle
    #     - IDLE 상태 → start=True (새 에피소드 녹화 시작)
    #     - RECORDING 상태 → start=True (worker_record 가 early-stop + save 처리)
    #   Left Y  (left_buttons[1]  bButton) rising-edge → record_mode.reset = True (현 에피 폐기)
    #   Right B (right_buttons[1] bButton) rising-edge → SET (start/reset/replay/done/deploy 전부 False).
    #     worker_record 는 record_task_shm.task_name 이 비어있지 않으면 WAIT_FOR_SET→IDLE 로 자동 전이.
    RECORD_BTN_THRESH = 0.5
    prev_lx = False   # left aButton (= X)
    prev_ly = False   # left bButton (= Y)
    prev_rb = False   # right bButton (= B)

    logger_mp.info(f"[VR] start. vr_input={vr_input}")

    while not shared_event['shutdown'].is_set():
        now = time.perf_counter_ns()
        actual_hz = 1.0 / max((now - last_time) / 1e9, 1e-6)
        freq_shm.write_data(vr_freq=actual_hz)

        try:
            mode_data = record_mode_shm.read_data()
            home = bool(mode_data["home"])
        except Exception:
            home = False

        if vr_input == "hand":
            head_rmat, left_wrist, right_wrist, left_hand, right_hand, right_distal, right_proximal = \
                tv_wrapper.get_data_with_segments()

            if home:
                home_left_wrist  = left_wrist.copy()
                home_right_wrist = right_wrist.copy()

            if home_left_wrist is not None and home_right_wrist is not None:
                rel_left_wrist  = left_wrist.copy()
                rel_right_wrist = right_wrist.copy()
                rel_left_wrist[:3, 3]  = left_wrist[:3, 3]  - home_left_wrist[:3, 3]
                rel_right_wrist[:3, 3] = right_wrist[:3, 3] - home_right_wrist[:3, 3]
            else:
                rel_left_wrist  = left_wrist
                rel_right_wrist = right_wrist

            television_shm.write_data(
                head_rmat=head_rmat,
                left_wrist_mat=rel_left_wrist,
                right_wrist_mat=rel_right_wrist,
                left_hand=left_hand,
                right_hand=right_hand,
                right_distal=right_distal,
                right_proximal=right_proximal,
                television_ts=np.int64(time.perf_counter_ns()),
            )

        else:  # vr_input == "controller"
            head_mat, left_ctrl_mat, right_ctrl_mat, left_state, right_state, connected = \
                tv_wrapper.get_controller_data()

            # TELEVISION SHM: wrist target은 controller pose 그대로 (clutch/IK는 worker_g1_ik에서)
            # hand keypoints 필드들은 controller 모드에선 사용하지 않으므로 0 채움.
            ts_ctrl = np.int64(time.perf_counter_ns())
            television_shm.write_data(
                head_rmat=head_mat,
                left_wrist_mat=left_ctrl_mat,
                right_wrist_mat=right_ctrl_mat,
                left_hand=zero_keypoints,
                right_hand=zero_keypoints,
                right_distal=zero_keypoints,
                right_proximal=zero_keypoints,
                television_ts=ts_ctrl,
            )

            # QUEST_CONTROLLER SHM에 button/trigger state 적재.
            # state layout: [trigger, squeeze, thumb_x, thumb_y, a, b, thumb_click]
            quest_controller_shm.write_data(
                left_ctrl_mat=left_ctrl_mat,
                right_ctrl_mat=right_ctrl_mat,
                left_trigger    =np.float32(left_state[0]),
                left_squeeze    =np.float32(left_state[1]),
                left_thumbstick =np.asarray(left_state[2:4], dtype=np.float32),
                left_buttons    =np.asarray(left_state[4:7], dtype=np.float32),
                right_trigger   =np.float32(right_state[0]),
                right_squeeze   =np.float32(right_state[1]),
                right_thumbstick=np.asarray(right_state[2:4], dtype=np.float32),
                right_buttons   =np.asarray(right_state[4:7], dtype=np.float32),
                connected       =np.bool_(connected),
                controller_ts   =ts_ctrl,
            )

            # ---- Record-control rising-edge → record_mode_shm ----
            # left_state / right_state index: [trigger, squeeze, tx, ty, a, b, thumb_click]
            lx_now = float(left_state[4])  >= RECORD_BTN_THRESH
            ly_now = float(left_state[5])  >= RECORD_BTN_THRESH
            rb_now = float(right_state[5]) >= RECORD_BTN_THRESH

            if lx_now and not prev_lx:
                # Toggle: worker_record 가 IDLE/RECORDING 양쪽에서 start=True 를 처리한다
                # (RECORDING 중에는 early-stop + save 동작).
                rm = record_mode_shm.read_data()
                rm["start"]  = np.bool_(True)
                rm["reset"]  = np.bool_(False)
                rm["replay"] = np.bool_(False)
                rm["done"]   = np.bool_(False)
                record_mode_shm.write_data(**rm)
                logger_mp.info("[VR] Left-X edge -> Record START toggle (worker_record handles).")
            prev_lx = lx_now

            if ly_now and not prev_ly:
                rm = record_mode_shm.read_data()
                rm["reset"]  = np.bool_(True)
                rm["start"]  = np.bool_(False)
                rm["replay"] = np.bool_(False)
                rm["done"]   = np.bool_(False)
                record_mode_shm.write_data(**rm)
                logger_mp.info("[VR] Left-Y edge -> Record DROP (reset current episode).")
            prev_ly = ly_now

            if rb_now and not prev_rb:
                rm = record_mode_shm.read_data()
                rm["start"]  = np.bool_(False)
                rm["reset"]  = np.bool_(False)
                rm["replay"] = np.bool_(False)
                rm["done"]   = np.bool_(False)
                rm["deploy"] = np.bool_(False)
                record_mode_shm.write_data(**rm)
                logger_mp.info("[VR] Right-B edge -> SET (worker_record WAIT_FOR_SET→IDLE if task_name set in GUI).")
            prev_rb = rb_now

        last_time = now

        next_time += period_ns
        sleep_ns = next_time - time.perf_counter_ns()
        if sleep_ns > 0:
            if sleep_ns > 1_000_000:
                time.sleep((sleep_ns - 500_000) / 1e9)
            while time.perf_counter_ns() < next_time:
                pass
        else:
            next_time = time.perf_counter_ns()

    logger_mp.info("[VR] 종료 신호 수신. 종료합니다.")
    television_shm.worker_close()
    record_mode_shm.worker_close()
    freq_shm.worker_close()
    quest_controller_shm.worker_close()


if __name__ == "__main__":
    import multiprocessing as mp
    import signal
    import sys

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    mp.freeze_support()

    shutdown_event = mp.Event()
    shared_event = {"shutdown": shutdown_event}

    shared_lock = {
        "camera_lock":           mp.Lock(),
        "television_lock":       mp.Lock(),
        "record_lock":           mp.Lock(),
        "freq_lock":             mp.Lock(),
        "quest_controller_lock": mp.Lock(),
    }

    shm_name = {
        "camera_shm":           "camera_shm",
        "television_shm":       "television_shm",
        "record_mode_shm":      "record_mode_shm",
        "freq_shm":             "freq_shm",
        "quest_controller_shm": "quest_controller_shm",
    }

    def _handle_signal(signum, frame):
        try:
            logger_mp.info(f"[MAIN] Received signal {signum}. Requesting shutdown...")
        except Exception:
            pass
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass

    proc = mp.Process(
        target=worker_vr,
        args=(shared_event, shm_name, shared_lock, "hand"),
        name="VRWorker",
        daemon=False,
    )
    proc.start()

    try:
        while proc.is_alive():
            proc.join(timeout=0.5)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        shutdown_event.set()
        proc.join(timeout=5.0)
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            proc.join(timeout=2.0)

    sys.exit(0)
