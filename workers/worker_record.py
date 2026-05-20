# worker_record.py — Phase D refactor: collector-thread + post-align.
#
# 동작 흐름:
#  WAIT_FOR_SET → (task_name 입력) → IDLE
#  IDLE → (start=True) → RECORDING
#          ├─ RecordCollectors 인스턴스 생성, poller thread 3개 start
#          └─ episode_start_t = perf_counter()
#  RECORDING tick (20Hz 외부 루프, FSM 폴링용):
#     - reset=True → 폐기 (collectors 정지 후 폐기) → IDLE
#     - start=True (이미 RECORDING) → early-stop → 저장 → IDLE
#     - elapsed >= episode_len → 정상 종료 → 저장 → IDLE
#
# 저장 (utils/record_collectors.align_and_save_episode):
#   각 RawStreamBuffer 를 dump → 공통 시간축 (50Hz, intersection) 생성 →
#   continuous 신호는 linear, image 는 ZOH 로 정렬 → ParquetSink.append loop +
#   raw_ts_* 메타 컬럼 + VideoSink mp4. LeRobot v2.1 파일 구조 그대로 유지.

import os
import time
import glob
from enum import Enum
from typing import Optional

import numpy as np

from utils.rate import Rate
from utils import ParquetSink, VideoSink
from utils.record_collectors import RecordCollectors, align_and_save_episode, DEFAULT_OUTPUT_HZ

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    CAMERA, RECORD_TASK_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_MODE_LAYOUT,
    WORKER_FREQ, ROBOT_ACTION, ROBOT_OBS, WORKSPACE_MASK,
    TELEOP_CONFIG, CAMERA_MAPPING_INV, HAND_MAPPING_INV,
    TELEVISION, QUEST_CONTROLLER,
)

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


class State(Enum):
    WAIT_FOR_SET = 1
    IDLE         = 2
    RECORDING    = 3


def read_mode_snapshot(shm):
    m = shm.read_data()
    return bool(m["start"]), bool(m["reset"]), bool(m["replay"]), m


def worker_record(shared_event, shm_name, shared_lock):
    # ── SharedMemory 핸들 -----------------------------------------------
    shm_objects = []

    def attach(name, layout, lock_key):
        try:
            s = SharedMemoryManager(layout, shared_lock[lock_key], shm_name[name])
            shm_objects.append(s)
            return s
        except Exception as e:
            logger_mp.error(f"[Record] SHM attach 실패 ({name}): {e}")
            return None

    record_task_shm    = attach("record_task_shm",    RECORD_TASK_LAYOUT,    "record_lock")
    record_episode_shm = attach("record_episode_shm", RECORD_EPISODE_LAYOUT, "record_lock")
    record_mode_shm    = attach("record_mode_shm",    RECORD_MODE_LAYOUT,    "record_lock")
    camera_shm         = attach("camera_shm",         CAMERA,                "camera_lock")
    robot_action_shm   = attach("robot_action_shm",   ROBOT_ACTION,          "robot_action_lock")
    robot_obs_shm      = attach("robot_obs_shm",      ROBOT_OBS,             "robot_obs_lock")
    freq_shm           = attach("freq_shm",           WORKER_FREQ,           "freq_lock")
    workspace_mask_shm = attach("workspace_mask_shm", WORKSPACE_MASK,        "workspace_mask_lock")
    teleop_config_shm  = attach("teleop_config_shm",  TELEOP_CONFIG,         "record_lock")
    television_shm     = attach("television_shm",     TELEVISION,            "television_lock")
    controller_shm     = attach("quest_controller_shm", QUEST_CONTROLLER,    "quest_controller_lock")

    # ── camera_type + hand_type (TELEOP_CONFIG SHM 의 1회 기록 값) ----------
    camera_type_str = "zed"
    hand_type_str   = "inspire"
    if teleop_config_shm is not None:
        try:
            cfg = teleop_config_shm.read_data()
            camera_type_str = CAMERA_MAPPING_INV.get(int(cfg["camera_type"].item()), "zed")
            hand_type_str   = HAND_MAPPING_INV.get(int(cfg["hand_type"].item()),     "inspire")
        except Exception as e:
            logger_mp.warning(f"[Record] teleop_config 읽기 실패: {e}")
    use_zed       = (camera_type_str == "zed")
    use_realsense = (camera_type_str == "realsense")
    logger_mp.info(f"[Record] camera_type={camera_type_str} (use_zed={use_zed}, use_realsense={use_realsense}) hand_type={hand_type_str}")

    # ── 외부 루프 주파수 (FSM 폴링용 — 데이터 수집 thread 는 collectors 가 별도 thread 로 처리)
    outer_hz = 20.0
    rate     = Rate(outer_hz)

    # ── State Machine 초기화 ------------------------------------------
    state = State.WAIT_FOR_SET
    ep_idx = 0
    episode_len_sec  = 0
    episode_start_t  = 0.0
    collectors: Optional[RecordCollectors] = None

    parquet_sink = ParquetSink(logger_mp)
    video_sink   = VideoSink(logger_mp, fps=DEFAULT_OUTPUT_HZ)

    def _start_collectors() -> None:
        nonlocal collectors
        if collectors is not None:
            try: collectors.stop_and_dump()
            except Exception: pass
        collectors = RecordCollectors(
            shm={
                'robot_obs':    robot_obs_shm,
                'robot_action': robot_action_shm,
                'television':   television_shm,
                'controller':   controller_shm,
                'camera':       camera_shm,
            },
            use_zed=use_zed,
            use_realsense=use_realsense,
        )
        collectors.start()

    def _stop_collectors_save(task_name: str, this_ep_idx: int) -> bool:
        """Stop collectors → align → save. Returns True on success."""
        nonlocal collectors
        if collectors is None:
            return False
        dumped = collectors.stop_and_dump()
        collectors = None
        return align_and_save_episode(
            dumped=dumped,
            parquet_sink=parquet_sink,
            video_sink=video_sink,
            task_name=task_name,
            ep_idx=this_ep_idx,
            use_zed=use_zed,
            use_realsense=use_realsense,
            output_hz=DEFAULT_OUTPUT_HZ,
            hand_type=hand_type_str,   # Phase K6: hand 종류별 hand DOF truncation
        )

    def _stop_collectors_discard() -> None:
        nonlocal collectors
        if collectors is None:
            return
        try:
            collectors.stop_and_dump()
        except Exception:
            pass
        collectors = None

    logger_mp.info("[Record] state=WAIT_FOR_SET")

    while not shared_event["shutdown"].is_set():
        try:
            try:
                task_data    = record_task_shm.read_data()
                episode_data = record_episode_shm.read_data()
                start, reset, replay, mode_raw = read_mode_snapshot(record_mode_shm)
            except Exception:
                continue

            # replay 는 record 와 무관 — IDLE 로만 강제 (replay 자체는 worker_g1_ik 가 담당)
            if replay:
                _stop_collectors_discard()
                state = State.IDLE
                continue

            # WAIT_FOR_SET ----------------------------------------------
            if state is State.WAIT_FOR_SET:
                task_name = task_data["task_name"].item().strip()
                if task_name and task_name != "0":
                    data_base = os.path.join("record", task_name, "data")
                    if os.path.isdir(data_base):
                        files = glob.glob(os.path.join(data_base, "chunk-*", "episode_*.parquet"))
                        if files:
                            nums = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files]
                            ep_idx = max(nums) + 1
                        else:
                            ep_idx = 0
                    else:
                        ep_idx = 0
                    os.makedirs(os.path.join("record", task_name), exist_ok=True)
                    state = State.IDLE
                    logger_mp.info(f"[Record] task='{task_name}' → state=IDLE (next ep_idx={ep_idx})")
                continue

            # IDLE -------------------------------------------------------
            if state is State.IDLE:
                if reset:
                    record_mode_shm.write_data(reset=False, start=False)
                    continue
                if start:
                    num_eps    = int(episode_data["num_episodes"].item())
                    ep_len_sec = int(episode_data["episode_len"].item())
                    if ep_idx >= num_eps or ep_len_sec <= 0:
                        record_mode_shm.write_data(reset=False, start=False)
                        continue
                    record_mode_shm.write_data(start=False, reset=False, done=False)
                    episode_len_sec = ep_len_sec
                    episode_start_t = time.perf_counter()
                    _start_collectors()
                    state = State.RECORDING
                    logger_mp.info(f"[Record] Episode {ep_idx} START (len={episode_len_sec}s, collectors on)")
                continue

            # RECORDING -------------------------------------------------
            if state is State.RECORDING:
                # reset 우선 — 폐기
                if reset:
                    record_mode_shm.write_data(reset=False)
                    _stop_collectors_discard()
                    state = State.IDLE
                    logger_mp.info(f"[Record] Episode {ep_idx} discarded (reset).")
                    continue

                elapsed = time.perf_counter() - episode_start_t

                # early-stop (start 다시 누름) or 정상 종료 (elapsed >= len) → 저장
                if start or (elapsed >= float(episode_len_sec)):
                    record_mode_shm.write_data(start=False)
                    task_name = task_data["task_name"].item().strip()
                    ok = _stop_collectors_save(task_name=task_name, this_ep_idx=ep_idx)
                    if ok:
                        logger_mp.info(
                            f"[Record] Episode {ep_idx} saved "
                            f"(elapsed={elapsed:.2f}s, {'early-stop' if start else 'len-done'})."
                        )
                        record_episode_shm.write_data(episode_index=np.int32(ep_idx + 1))
                        record_mode_shm.write_data(done=True)
                        ep_idx += 1
                    state = State.IDLE
                    continue

                # 진행률
                progress = max(0, min(100, int(elapsed * 100.0 / float(episode_len_sec))))
                record_episode_shm.write_data(logging_progress=np.int32(progress))
                continue

        finally:
            try:
                freq_shm.write_data(record_freq=rate.tick_hz())
            except Exception:
                pass
            rate.sleep()

    # ── shutdown ---------------------------------------------------------
    logger_mp.info("[Record] shutdown received; stopping collectors and SHMs.")
    _stop_collectors_discard()
    for s in shm_objects:
        try:
            if hasattr(s, 'worker_close'):
                s.worker_close()
        except Exception as e:
            logger_mp.warning(f"[Record] SHM close 실패: {e}")
    logger_mp.info(f"[Record] {len(shm_objects)} SHM closed.")
