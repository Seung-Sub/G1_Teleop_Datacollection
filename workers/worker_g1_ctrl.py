# g1_ctrl_worker_split.py
import os
import time
import threading
import numpy as np

from workers.A_dual_rate_worker import DualRateWorker  # 기존 DualRateWorker 사용
from utils.state import State, EventsSnapshot, next_state
from utils.rate import Rate

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    WORKER_FREQ, RECORD_MODE_LAYOUT, ROBOT_ACTION, ROBOT_OBS, ROBOT_AMO_OBS
)

from g1_control.g1_whole_control import G1_29_ArmController
from g1_control.g1_head_dynamixel import Dynamixel_Controller
from g1_control.remote_controller import RemoteController, KeyMap

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# 이중 주파수 사용/ 관측은 빠르게 제어는 느리게 
OBS_HZ = 300.0
ACT_HZ = 50.0  # 느린 루프 주파수로 사용

class G1CtrlWorker(DualRateWorker):
    """
    빠른 루프(OBS_HZ): RUN 상태에서만 관측 읽고 robot_obs_shm에 씀
    느린 루프(ACT_HZ): RUN 상태에서만 액션 읽고 실제 제어 수행
    상태 전이/초기화도 느린 루프에서 처리
    """
    def __init__(self, shared_event, shm_name, shared_lock, mode):
        super().__init__(slow_hz=ACT_HZ, fast_hz=OBS_HZ)

        # 외부 이벤트
        self._shared_event = shared_event
        self.mode = mode

        # ── SHM ───────────────────────────────────────────
        self.freq_shm         = SharedMemoryManager(WORKER_FREQ,        shared_lock["freq_lock"],         shm_name["freq_shm"])
        self.record_mode_shm  = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"],       shm_name["record_mode_shm"])
        self.robot_action_shm = SharedMemoryManager(ROBOT_ACTION,       shared_lock["robot_action_lock"], shm_name["robot_action_shm"])
        self.robot_obs_shm    = SharedMemoryManager(ROBOT_OBS,          shared_lock["robot_obs_lock"],    shm_name["robot_obs_shm"])
        self.robot_amo_obs_shm    = SharedMemoryManager(ROBOT_AMO_OBS,          shared_lock["robot_amo_obs_lock"],    shm_name["robot_amo_obs_shm"])

        # ── G1/Head 컨트롤 ────────────────────────────────
        self.g1_initialized = False
        self.g1_ctrl = None
        self.d_ctrl  = None

        # ── FSM 상태 ─────────────────────────────────────
        self.state = State.WAIT_CONNECT
        self._state_lock = threading.Lock()


        self._buf_leg_q   = np.empty(12, dtype=np.float32)
        self._buf_leg_dq  = np.empty(12, dtype=np.float32)
        self._buf_waist_q = np.empty(3,  dtype=np.float32)
        self._buf_waist_dq= np.empty(3,  dtype=np.float32)
        self._buf_arm_q   = np.empty(14, dtype=np.float32)
        self._buf_arm_dq  = np.empty(14, dtype=np.float32)
        self._buf_amo_q   = np.empty(23, dtype=np.float32)
        self._buf_amo_dq  = np.empty(23, dtype=np.float32)
        self._buf_quat    = np.empty(4, dtype=np.float32)
        self._buf_gyro    = np.empty(3, dtype=np.float32)


        logger_mp.info("[G1_Ctrl] FSM start: WAIT_CONNECT")

    # 이벤트 스냅샷
    def _events_snapshot(self) -> EventsSnapshot:
        select_pressed = False
        if self.g1_ctrl is not None and getattr(self.g1_ctrl, "remote", None) is not None:
            select_pressed = (self.g1_ctrl.remote.button[KeyMap.select] == 1)

        return EventsSnapshot(
            shutdown       = self._shared_event['shutdown'].is_set(),
            emergency      = self._shared_event['emergency'].is_set(),
            g1_ready       = self.g1_initialized,
            hand_ready     = self._shared_event['set_hand'].is_set(),
            start          = self._shared_event['set_start'].is_set(),
            select_pressed = select_pressed
        )

    # 느린 루프: 상태 전이 + (RUN일 때) 액션→제어 입력
    def do_slow(self) -> None:
        # --- 상태 전이(모든 사이클) ---
        with self._state_lock:
            cur = self.state
            nxt = next_state(cur, self._events_snapshot())
            if nxt != cur:
                logger_mp.info(f"[G1_Ctrl] {cur.name} -> {nxt.name}")
                self.state = nxt
                if nxt is State.EXIT:
                    self._stop_event.set()
                    return
        
        # --- WAIT_CONNECT: 초기화 트리거 ---
        if self.state is State.WAIT_CONNECT:
            if self._shared_event['set_g1'].is_set() and not self.g1_initialized:
                logger_mp.info("[G1_Ctrl] Initializing G1/DXL...")
                try:
                    
                    self.g1_ctrl = G1_29_ArmController(self.mode)
                    self.d_ctrl  = Dynamixel_Controller()
                    
                    self.g1_ctrl.speed_gradual_max()
                    self.g1_initialized = True
                    logger_mp.info("[G1_Ctrl] G1/DXL initialized.")
                except Exception as e:
                    logger_mp.exception("[G1_Ctrl] G1 init failed: %s", e)
                    self._shared_event['emergency'].set()
            return  # 제어는 RUN에서만

        # --- WAIT_START / PAUSE: 아무 것도 안 함 ---
        if self.state in (State.WAIT_START, State.PAUSE):
            
            return

        # --- RUN: 액션 읽고 제어 입력(50 Hz) ---
        if self.state is State.RUN and self.g1_initialized:
            
            mode_data = self.record_mode_shm.read_data()
            home = mode_data["home"]

            if home:
                # 홈 포즈 명령
                self.g1_ctrl.ctrl_defualt_pose()
                mode_data["home"] = False
                mode_data["start"] = False
                self.record_mode_shm.write_data(**mode_data)
                self._shared_event['set_start'].clear()
                return
            
            # 액션 읽기
            action = self.robot_action_shm.read_data()
            
            action_leg         = action["action_leg"]
            action_waist       = action["action_waist"]
            action_waist_tauff = action["action_waist_tauff"]
            action_head        = action["action_head"]
            action_arm         = action["action_arm"]
            action_arm_tauff   = action["action_arm_tauff"]
            # 제어 입력
            self.g1_ctrl.ctrl_leg(action_leg)
            self.g1_ctrl.ctrl_waist(action_waist, action_waist_tauff)

            # 머리 제어 (Dynamixel 사용)
            if hasattr(self, 'd_ctrl') and self.d_ctrl is not None:
                d_tick = np.array([self.d_ctrl.rad_to_dxl_tick(x) for x in action_head], dtype=int)
                # 실제 머리 제어 명령
                self.d_ctrl.ctrl_dynamixel(d_tick)
            else:
                logger_mp.warning("[G1_Ctrl] d_ctrl not available for head control")

            self.g1_ctrl.ctrl_arm(action_arm, action_arm_tauff)


    # 빠른 루프: RUN에서만 관측(250 Hz) → SHM 기록 (+옵션: 주파수 측정)
    def do_fast(self) -> None:
        if not (self.g1_initialized and self.state is State.RUN):
            return

        # (옵션) 현재 fast 루프 실제 Hz 기록
        try:
            g1_freq = self.fast_rate.tick_hz()
            self.freq_shm.write_data(g1_freq=g1_freq)
        except AttributeError:
            pass

        # 관측 읽기
        obs_leg   = self.g1_ctrl.get_current_leg_q(out=self._buf_leg_q)
        obs_waist = self.g1_ctrl.get_current_waist_q(out=self._buf_waist_q)
        obs_arm   = self.g1_ctrl.get_current_arm_q(out=self._buf_arm_q)
        
        # gr00t_kistar 모드에서는 다이나믹셀 관측 읽기 생략
        if self.d_ctrl is not None:
            obs_dxl   = self.d_ctrl.get_current_motor_q()
            obs_head  = np.array([ self.d_ctrl.dxl_tick_to_rad(x) for x in obs_dxl ],dtype=np.float32)
        else:
            obs_head  = np.zeros(2, dtype=np.float32)  # gr00t_kistar 모드에서는 0으로 채움

        amo_q     = self.g1_ctrl.get_obs_motor_q(out=self._buf_amo_q)
        amo_dq    = self.g1_ctrl.get_obs_motor_dq(out=self._buf_amo_dq)
        quat, ang_vel = self.g1_ctrl.get_imu_state_data(out_quat=self._buf_quat, out_gyro=self._buf_gyro)

        # SHM 쓰기
        self.robot_obs_shm.write_data(
            obs_leg=obs_leg,
            obs_waist=obs_waist,
            obs_head=obs_head,
            obs_arm=obs_arm
        )
        self.robot_amo_obs_shm.write_data(
            amo_q = amo_q,
            amo_dq = amo_dq,
            quat = quat,
            ang_vel = ang_vel,
        )

    # 리소스 정리
    def stop(self) -> None:
        super().stop()
        try:
            if self.d_ctrl is not None:
                self.d_ctrl.disconnect_dynamixel()
            if self.g1_ctrl is not None:
                self.g1_ctrl.stop()
            self.freq_shm.worker_close()
            self.robot_action_shm.worker_close()
            self.robot_obs_shm.worker_close()
            self.record_mode_shm.worker_close()
            self.robot_amo_obs_shm.worker_close()
        finally:
            logger_mp.info("[G1_Ctrl] 종료 및 SHM 정리 완료")


# ────────────── 실행 진입점 예시 ──────────────
def worker_g1_ctrl(shared_event, shm_name, shared_lock,mode):
    w = G1CtrlWorker(shared_event, shm_name, shared_lock, mode)
    try:
        w.start()
        while not shared_event['shutdown'].is_set():
            time.sleep(0.1)
    finally:
        w.stop()
