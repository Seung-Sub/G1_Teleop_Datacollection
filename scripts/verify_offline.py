#!/usr/bin/env python3
"""Offline verification — 로봇 / Quest3 / 카메라 없이 코드베이스 무결성 검증.

PC + conda teleop 환경만 있으면 통과해야 하는 항목들:
  - 모든 신규 utils 모듈 import (raw_stream, align, record_collectors,
    camera_discovery, mat_tool)
  - sharedmemory.shm_schema 의 mapping / TS field
  - utils.mat_tool.cosine_ease / se3_interp 수치 검증
  - utils.raw_stream.RawStreamBuffer dedup
  - utils.align.interp_to_axis linear/zoh, common_time_axis
  - utils.camera_discovery 의 lazy import (pyrealsense2/pyzed.sl 없어도 import OK)
  - g1_control.g1_ik.G1_29_ArmIK build (pinocchio + casadi)
  - main.py CLI 해석 (argparse 가 우리 옵션 들고 있는지)

Usage:
    cd /path/to/G1_Teleoperation
    conda activate teleop
    python scripts/verify_offline.py
"""
from __future__ import annotations
import os
import sys
import traceback

# 프로젝트 루트를 path 에 추가
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PASS = 0
FAIL = 0
FAILED_DESCRIPTIONS = []


def check(name: str, fn):
    """fn() 호출. 예외 없이 끝나면 PASS."""
    global PASS, FAIL
    try:
        result = fn()
        if result is False:
            raise AssertionError("check returned False")
        suffix = f"  {result}" if isinstance(result, str) else ""
        print(f"  [PASS] {name}{suffix}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        FAIL += 1
        FAILED_DESCRIPTIONS.append(name)


def section(title: str):
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print(f"verify_offline.py — repo root: {ROOT}\n")

    # ------------------------------------------------------------------
    section("1) Python + 기본 의존성 버전")
    def _versions():
        import numpy, scipy, cv2, pandas
        return (f"py={sys.version_info.major}.{sys.version_info.minor} "
                f"numpy={numpy.__version__} scipy={scipy.__version__} "
                f"cv2={cv2.__version__} pandas={pandas.__version__}")
    check("import core deps", _versions)

    # ------------------------------------------------------------------
    section("2) sharedmemory.shm_schema 의 mapping / TS field")
    def _shm_schema():
        from sharedmemory.shm_schema import (
            HAND_MAPPING, CAMERA_MAPPING, VR_INPUT_MAPPING,
            WAIST_MAPPING, HEAD_MAPPING, TELEOP_CONFIG,
            ROBOT_OBS, ROBOT_ACTION, CAMERA, TELEVISION, QUEST_CONTROLLER,
            LEFT_TOUCH_SENSOR_LAYOUT, RIGHT_TOUCH_SENSOR_LAYOUT, DEPTH_MAP,
        )
        # mapping
        assert HAND_MAPPING == {'inspire': 0, 'dex3': 1}
        assert 'auto' in CAMERA_MAPPING and 'none' in CAMERA_MAPPING
        assert WAIST_MAPPING == {'hmd': 0, 'fixed': 1}
        assert HEAD_MAPPING  == {'dxl': 0, 'off': 1}
        # TS fields
        def names(schema): return [f[0] for f in schema]
        assert 'obs_body_ts'    in names(ROBOT_OBS)
        assert 'obs_hand_ts'    in names(ROBOT_OBS)
        assert 'action_body_ts' in names(ROBOT_ACTION)
        assert 'action_hand_ts' in names(ROBOT_ACTION)
        assert 'camera_zed_ts'       in names(CAMERA)
        assert 'camera_realsense_ts' in names(CAMERA)
        assert 'television_ts'       in names(TELEVISION)
        assert 'controller_ts'       in names(QUEST_CONTROLLER)
        assert 'l_touch_ts' in names(LEFT_TOUCH_SENSOR_LAYOUT)
        assert 'r_touch_ts' in names(RIGHT_TOUCH_SENSOR_LAYOUT)
        assert 'depth_map_ts' in names(DEPTH_MAP)
        # teleop_config 의 waist/head field
        assert 'waist_mode' in names(TELEOP_CONFIG)
        assert 'head_mode'  in names(TELEOP_CONFIG)
        return "all mappings + ts fields present"
    check("shm_schema", _shm_schema)

    # ------------------------------------------------------------------
    section("3) utils.mat_tool — cosine_ease + se3_interp 수치 검증")
    def _mat_tool():
        import numpy as np, math
        from utils.mat_tool import cosine_ease, se3_interp, fast_mat_inv
        assert cosine_ease(0.0) == 0.0
        assert abs(cosine_ease(1.0) - 1.0) < 1e-12
        assert abs(cosine_ease(0.5) - 0.5) < 1e-12

        T0 = np.eye(4)
        T1 = np.eye(4); T1[:3, 3] = [1, 2, 3]
        th = np.pi/2
        T1[:3, :3] = np.array([[np.cos(th), -np.sin(th), 0],
                               [np.sin(th),  np.cos(th), 0],
                               [0, 0, 1]])
        assert np.max(np.abs(se3_interp(T0, T1, 0.0) - T0)) < 1e-9
        assert np.max(np.abs(se3_interp(T0, T1, 1.0) - T1)) < 1e-9
        mid = se3_interp(T0, T1, 0.5)
        assert np.allclose(mid[:3, 3], [0.5, 1.0, 1.5])
        ang = math.atan2(mid[1, 0], mid[0, 0])
        assert abs(math.degrees(ang) - 45.0) < 1e-6
        # fast_mat_inv
        Ti = fast_mat_inv(T1)
        assert np.max(np.abs(T1 @ Ti - np.eye(4))) < 1e-9
        return "cosine_ease + se3_interp + fast_mat_inv OK"
    check("mat_tool helpers", _mat_tool)

    # ------------------------------------------------------------------
    section("4) utils.raw_stream + utils.align")
    def _stream_align():
        import numpy as np
        from utils.raw_stream import RawStreamBuffer
        from utils.align import interp_to_axis, common_time_axis
        b = RawStreamBuffer('t')
        assert b.append(100, {'q': 1.0}) is True
        assert b.append(100, {'q': 1.1}) is False
        assert b.append(0,   {'q': 9.9}) is False
        assert b.append(200, {'q': 2.0}) is True
        ts, p = b.dump()
        assert ts.tolist() == [100, 200]

        ts_src = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
        val    = np.array([0.0, 1.0, 4.0])
        ts_dst = np.array([50_000_000, 150_000_000], dtype=np.int64)
        assert np.allclose(interp_to_axis(ts_src, val, ts_dst, 'linear'),  [0.5, 2.5])
        assert np.allclose(interp_to_axis(ts_src, val, ts_dst, 'zoh'),     [0.0, 1.0])
        ax = common_time_axis([np.array([100,200,300,400], dtype=np.int64),
                               np.array([150,250,350], dtype=np.int64)],
                              rate_hz=1e9 / 50_000_000)
        assert ax.size >= 1
        return "RawStreamBuffer + interp_to_axis + common_time_axis OK"
    check("raw_stream + align", _stream_align)

    # ------------------------------------------------------------------
    section("5) utils.camera_discovery — lazy SDK import")
    def _camera_discovery():
        from utils.camera_discovery import (
            discover_realsense, discover_zed, auto_select, find_by_serial,
        )
        # 실제 device 없어도 빈 list 만 반환하면 OK
        rs = discover_realsense()
        zed = discover_zed()
        assert isinstance(rs, list)
        assert isinstance(zed, list)
        t, s, n = auto_select()
        if t is None:
            verdict = "no device detected (expected when no camera attached)"
        else:
            verdict = f"detected: type={t}, serial={s}, name={n}"
        return verdict
    check("camera_discovery", _camera_discovery)

    # ------------------------------------------------------------------
    section("6) utils.record_collectors + parquet_sink import")
    def _record_collectors():
        from utils.record_collectors import (
            RecordCollectors, align_and_save_episode, DEFAULT_OUTPUT_HZ,
        )
        from utils.parquet_sink import ParquetSink
        from utils.video_sink   import VideoSink
        assert DEFAULT_OUTPUT_HZ == 50.0
        # ParquetSink.add_extra_column 메소드 존재
        assert hasattr(ParquetSink, 'add_extra_column')
        return f"output_hz={DEFAULT_OUTPUT_HZ} + ParquetSink.add_extra_column ✓"
    check("record_collectors + parquet_sink", _record_collectors)

    # ------------------------------------------------------------------
    section("7) g1_control.g1_ik.G1_29_ArmIK build (pinocchio + casadi)")
    def _ik_build():
        # 무거운 build (URDF + pinocchio + casadi opti). 성공하면 IK 가능.
        old_cwd = os.getcwd()
        try:
            os.chdir(ROOT)  # G1_29_ArmIK 가 'g1_control/assets/g1/...' 상대 path 사용
            from g1_control.g1_ik import G1_29_ArmIK
            ik = G1_29_ArmIK(False, False)
            T_L, T_R, T_H = ik.init_pose()
            assert T_L.shape == (4, 4) and T_R.shape == (4, 4) and T_H.shape == (4, 4)
            import numpy as np
            return f"init_pose ✓ (L_ee z={T_L[2,3]:.3f}, R_ee z={T_R[2,3]:.3f})"
        finally:
            os.chdir(old_cwd)
    check("G1_29_ArmIK build + init_pose", _ik_build)

    # ------------------------------------------------------------------
    section("8) workers — controller-mode 핵심 import")
    def _worker_imports():
        # vuer/params-proto argparse hijack 회피용 lazy import 패턴 검증.
        # main.py 가 import 한 직후 sys.argv 가 보존되어야 함.
        import argparse
        sys_argv_backup = sys.argv
        sys.argv = ['verify_offline.py']
        try:
            # worker_vr 는 vuer 끌어옴 — main.py 가 lazy 로 한 것과 동일하게
            from workers.worker_vr import worker_vr  # noqa
            from workers.worker_g1_ik   import worker_g1_ik          # noqa
            from workers.worker_g1_ctrl import worker_g1_ctrl        # noqa
            from workers.worker_hand_ctrl import worker_hand_ctrl    # noqa
            from workers.worker_record import worker_record          # noqa
            return "all workers imported, argparse preserved"
        finally:
            sys.argv = sys_argv_backup
    check("worker imports (vuer argparse hijack avoidance)", _worker_imports)

    # ------------------------------------------------------------------
    section("9) main.py --help 가 신규 CLI 옵션 노출하는지")
    def _main_help():
        import subprocess
        env = dict(os.environ)
        # main.py 가 GUI 끌어옴 — Qt 없는 환경에서도 --help 만큼은 통과해야 함
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
        res = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'main.py'), '--help'],
            capture_output=True, text=True, env=env, timeout=15,
        )
        out = (res.stdout or '') + (res.stderr or '')
        required = ['--hand', '--camera', '--vr-input', '--waist', '--head', '--no-robot', '--zed-mode']
        missing = [opt for opt in required if opt not in out]
        if missing:
            raise AssertionError(f"missing CLI: {missing}")
        return f"all required flags present ({len(required)})"
    check("main.py --help CLI surface", _main_help)

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f" SUMMARY  PASS={PASS}  FAIL={FAIL}")
    print("=" * 70)
    if FAIL:
        print(" Failed checks:")
        for d in FAILED_DESCRIPTIONS:
            print(f"   - {d}")
        sys.exit(1)
    print(" all offline checks passed — codebase + env integrity OK.")


if __name__ == '__main__':
    main()
