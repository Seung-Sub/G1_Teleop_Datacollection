"""worker_loco — Quest3 thumbstick → LocoClient.Move 보행 워커 (Phase N).

PART4 §2 적용. main.py 가 `--lower-body loco --gait thumbstick` 일 때만 spawn.

핵심:
  - LocoClient.Move(vx, vy, vyaw, continous_move=False) 는 SetVelocity(...,
    duration=1) 와 동치 → 매 루프 반복 호출하면 명령 끊김 시 1s 후 자동 정지 (안전).
    continous_move=True 는 duration=864000s (≈10일) — **절대 사용 금지**.
  - thumbstick 매핑 (xr_teleoperate teleop_hand_and_arm.py 319-341 검증):
      split: vx = -left_y, vy = -left_x, vyaw = -right_x. 스케일 0.3 공식 상한.
      left : 왼쪽 스틱만 (vyaw 도 left_x 로).
      right: 오른쪽 스틱만 (vx/vy/vyaw 모두 right).
  - 초기 검증은 보수적으로 vx_scale=0.15 권장. 검증 후 0.3 까지 상향.
  - 양쪽 thumb-click → Damp() (소프트 비상정지).
  - emergency event → Move(0,0,0) + Damp().

ChannelFactoryInitialize 주의:
  worker_loco 는 별도 프로세스 (mp.Process spawn) 에서 동작. G1_29_ArmController
  가 worker_g1_ctrl 프로세스에서 ChannelFactoryInitialize(0, network_interface)
  를 호출하지만, 이는 *그 프로세스* 한정. worker_loco 는 자기 프로세스에서 다시
  init 필요. 같은 domain (0) + 같은 interface 사용 → DDS 토픽 공유.
"""
from __future__ import annotations
import time
from typing import Tuple

import numpy as np
import yaml

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import QUEST_CONTROLLER

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


# 보행 스케일 — xr_teleoperate 의 0.3 이 공식 상한. 초기 검증은 0.15 보수적.
# 사용자 운용 검증 후 _SCALE_VX 등을 0.3 까지 상향 가능.
_SCALE_VX   = 0.15
_SCALE_VY   = 0.15
_SCALE_VYAW = 0.3
_DEADZONE   = 0.15        # |stick| <= 0.15 → 0 (중립)
_LOCO_LOOP_HZ = 20.0      # 저속 루프 — 0.05 s 간격. duration=1s 명령이라 충분.
_BUTTON_THRESH = 0.5
_THUMB_CLICK_IDX = 2      # QUEST_CONTROLLER.left_buttons[2] = thumbstick click


def _deadzone(v: float) -> float:
    return 0.0 if abs(v) < _DEADZONE else v


def _map_thumbstick(lx: float, ly: float, rx: float, ry: float,
                    mode: str) -> Tuple[float, float, float]:
    """thumbstick (x, y) ∈ [-1, 1] → (vx, vy, vyaw). mode = 'split'/'left'/'right'."""
    if mode == 'left':
        vx   = -ly * _SCALE_VX
        vy   = 0.0
        vyaw = -lx * _SCALE_VYAW
    elif mode == 'right':
        vx   = -ry * _SCALE_VX
        vy   = 0.0
        vyaw = -rx * _SCALE_VYAW
    else:
        # split (공식 기본): 왼쪽 = 병진, 오른쪽 X = 회전.
        vx   = -ly * _SCALE_VX
        vy   = -lx * _SCALE_VY
        vyaw = -rx * _SCALE_VYAW
    return _deadzone(vx), _deadzone(vy), _deadzone(vyaw)


def worker_loco(shared_event, shm_name, shared_lock, gait_stick: str = 'split'):
    """LocoClient 보행 워커.

    Args:
        gait_stick: 'split' | 'left' | 'right' — thumbstick 매핑 (PART4 §1.1).
    """
    # SHM attach. main.py 가 quest_controller_shm owner.
    ctrl_shm = SharedMemoryManager(QUEST_CONTROLLER, shared_lock["quest_controller_lock"],
                                   shm_name["quest_controller_shm"])

    # ChannelFactoryInitialize — 본 프로세스에서 1회.
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    except ImportError as e:
        logger_mp.error(f"[Loco] unitree_sdk2py import 실패: {e} — gait 미동작.")
        return

    with open("utils/lan_config.yaml") as f:
        cfg = yaml.safe_load(f)
    try:
        ChannelFactoryInitialize(0, cfg["network_interface"])
    except Exception as e:
        logger_mp.info(f"[Loco] ChannelFactoryInitialize: {e} (이미 init 일 수 있음)")

    loco = LocoClient()
    try:
        loco.SetTimeout(0.0001)
        loco.Init()
        logger_mp.info(
            f"[Loco] LocoClient Init OK (gait_stick={gait_stick}, "
            f"scales vx={_SCALE_VX} vy={_SCALE_VY} vyaw={_SCALE_VYAW})"
        )
    except Exception as e:
        logger_mp.error(f"[Loco] LocoClient Init 실패: {e} — gait 미동작")
        return

    period = 1.0 / _LOCO_LOOP_HZ
    last_damp_time = 0.0
    damp_cooldown_sec = 1.0
    started = False

    try:
        while not shared_event['shutdown'].is_set():
            t0 = time.perf_counter()

            if shared_event.get('emergency', None) is not None and shared_event['emergency'].is_set():
                try:
                    loco.Move(0.0, 0.0, 0.0, continous_move=False)
                    loco.Damp()
                except Exception:
                    pass
                time.sleep(0.5)
                continue

            try:
                cd = ctrl_shm.read_data()
            except Exception:
                time.sleep(period); continue
            if not bool(cd['connected']):
                try:
                    loco.Move(0.0, 0.0, 0.0, continous_move=False)
                except Exception:
                    pass
                time.sleep(period); continue

            ls = np.asarray(cd['left_thumbstick'],  dtype=np.float32)
            rs = np.asarray(cd['right_thumbstick'], dtype=np.float32)
            lx, ly = float(ls[0]), float(ls[1])
            rx, ry = float(rs[0]), float(rs[1])

            l_click = float(cd['left_buttons'][_THUMB_CLICK_IDX])  >= _BUTTON_THRESH
            r_click = float(cd['right_buttons'][_THUMB_CLICK_IDX]) >= _BUTTON_THRESH
            now = time.perf_counter()
            if l_click and r_click and (now - last_damp_time) > damp_cooldown_sec:
                try:
                    loco.Damp()
                    logger_mp.warning("[Loco] both thumbsticks clicked → Damp() (soft e-stop).")
                except Exception as e:
                    logger_mp.error(f"[Loco] Damp 실패: {e}")
                last_damp_time = now
                time.sleep(period); continue

            vx, vy, vyaw = _map_thumbstick(lx, ly, rx, ry, gait_stick)

            try:
                loco.Move(vx, vy, vyaw, continous_move=False)
                if not started and (vx != 0.0 or vy != 0.0 or vyaw != 0.0):
                    try:
                        loco.Start()
                        logger_mp.info("[Loco] Start() called (first non-zero Move).")
                    except Exception:
                        pass
                    started = True
            except Exception as e:
                logger_mp.warning(f"[Loco] Move 실패: {e}")

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

    finally:
        try:
            loco.Move(0.0, 0.0, 0.0, continous_move=False)
            time.sleep(0.1)
            loco.StopMove()
        except Exception:
            pass
        try:
            ctrl_shm.worker_close()
        except Exception:
            pass
        logger_mp.info("[Loco] worker exiting.")
