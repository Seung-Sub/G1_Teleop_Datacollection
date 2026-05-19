"""Camera device discovery — RealSense (pyrealsense2) + ZED (pyzed.sl).

main.py 의 `--camera auto|zed|realsense|<serial>` 해석 시 사용.

자동 감지 우선순위 (사용자 결정, 2026-05-19):
  1. RealSense — 'Intel RealSense D435I' / D455 / D405 / D435 등 product name 매칭
  2. ZED — 'ZED 2i' / 'ZED Mini' / 'ZED' 등 model 매칭

import 는 *lazy* — pyrealsense2 / pyzed.sl 둘 다 무거운 native binding 이므로
실제 호출 순간에만 import. 미설치 환경에서도 모듈 자체는 import 가능.

Multi-camera 확장 시: utils/cameras.yaml 에 role↔serial 매핑을 두고
main.py 가 그것을 읽어 워커 다중 spawn (단일 ego 단계에선 미사용).
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


# RealSense 우선순위 (제품군 모델명 prefix)
_REALSENSE_PRIORITY = ('D435I', 'D455', 'D405', 'D435', 'D415', 'D457')
# ZED 우선순위
_ZED_PRIORITY = ('ZED 2i', 'ZED Mini', 'ZED 2', 'ZED', 'ZED-M')


def _name_priority(name: str, priority_list) -> int:
    """매칭되는 prefix 의 인덱스를 반환 (낮을수록 우선). 매칭 없으면 큰 값."""
    if not name:
        return 9999
    up = name.upper()
    for i, p in enumerate(priority_list):
        if p.upper() in up:
            return i
    return 9999


def discover_realsense() -> List[Dict[str, str]]:
    """[{'serial', 'name'}] 리스트. RealSense 미설치 / 미연결 시 빈 리스트."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        logger_mp.debug("[CamDiscover] pyrealsense2 not installed")
        return []
    out: List[Dict[str, str]] = []
    try:
        ctx = rs.context()
        for d in ctx.query_devices():
            try:
                sn   = d.get_info(rs.camera_info.serial_number)
                name = d.get_info(rs.camera_info.name)
                out.append({'serial': str(sn), 'name': str(name)})
            except Exception:
                continue
    except Exception as e:
        logger_mp.warning(f"[CamDiscover] realsense enumerate 실패: {e}")
        return []
    # priority 정렬
    out.sort(key=lambda d: _name_priority(d['name'], _REALSENSE_PRIORITY))
    return out


def discover_zed() -> List[Dict[str, str]]:
    """[{'serial', 'name'}] 리스트. ZED 미설치 / 미연결 시 빈 리스트."""
    try:
        import pyzed.sl as sl
    except ImportError:
        logger_mp.debug("[CamDiscover] pyzed.sl not installed")
        return []
    out: List[Dict[str, str]] = []
    try:
        devices = sl.Camera.get_device_list()
        for d in devices:
            try:
                sn    = int(d.serial_number)
                model = d.camera_model
                # sl.MODEL → str 직접 변환이 안 되는 SDK 버전 대비
                model_str = str(model).split('.')[-1] if model is not None else ''
                out.append({'serial': str(sn), 'name': model_str})
            except Exception:
                continue
    except Exception as e:
        logger_mp.warning(f"[CamDiscover] zed enumerate 실패: {e}")
        return []
    out.sort(key=lambda d: _name_priority(d['name'], _ZED_PRIORITY))
    return out


def discover_all() -> Dict[str, List[Dict[str, str]]]:
    return {
        'realsense': discover_realsense(),
        'zed':       discover_zed(),
    }


def auto_select(prefer: str = 'realsense') -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """첫 번째 사용가능한 카메라를 (type, serial, name) 으로 반환.

    prefer='realsense': RealSense 가 있으면 우선, 없으면 ZED.
    prefer='zed':       ZED 우선.
    아무것도 없으면 (None, None, None).
    """
    rs_list  = discover_realsense()
    zed_list = discover_zed()
    if prefer == 'zed':
        if zed_list:
            d = zed_list[0]
            return ('zed', d['serial'], d['name'])
        if rs_list:
            d = rs_list[0]
            return ('realsense', d['serial'], d['name'])
        return (None, None, None)
    # default = realsense first
    if rs_list:
        d = rs_list[0]
        return ('realsense', d['serial'], d['name'])
    if zed_list:
        d = zed_list[0]
        return ('zed', d['serial'], d['name'])
    return (None, None, None)


def find_by_serial(serial: str) -> Optional[Tuple[str, str, str]]:
    """serial 로 양쪽 SDK 모두 검색. 매칭 시 (type, serial, name)."""
    serial = str(serial).strip()
    for d in discover_realsense():
        if d['serial'] == serial:
            return ('realsense', d['serial'], d['name'])
    for d in discover_zed():
        if d['serial'] == serial:
            return ('zed', d['serial'], d['name'])
    return None
