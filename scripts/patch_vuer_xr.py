#!/usr/bin/env python3
"""Vuer 0.0.60 client JS chunk patch — WebXR hand-tracking 기본값 끄기.

배경
----
vuer 0.0.60 의 client JS 가 (Mac 이외 플랫폼에서) WebXR session 의 optionalFeatures
에 'hand-tracking' 을 hardcode 로 항상 포함시킨다. Quest 3 의 경우 hand-tracking
permission 이 활성화되면 hands 가 카메라에 보이는 순간 controller input 이 자동
demote 되어, --vr-input controller 모드로 main.py 를 띄워도 헤드셋 안에서 컨트롤러가
인식되지 않는다 (코드 chain 모두 정상이지만 Quest 측 input source 가 hand 만 송신).

확인된 hardcode (vuer 0.0.60):
  - chunk-{BU6qPyb1, Bf98F3Ua, Dd3xtWba, DmvjxeUa}.js
  - "HX({hand:!0,handTracking:!0..." 등의 hardcoded true
  - "handTracking:r=!KSe()" / "!ZSe()" / "!isAppleVisionPro()" 등의 platform-aware default

본 스크립트는 그 default 들을 false 로 바꾼다.

Usage
-----
    python scripts/patch_vuer_xr.py disable    # hand-tracking 끄기 (controller 모드용)
    python scripts/patch_vuer_xr.py restore    # 원본 복원
    python scripts/patch_vuer_xr.py status     # 현재 패치 상태 확인

Idempotent — 같은 모드 두 번 실행해도 안전. 원본은 *.orig 백업 후 수정.

⚠️ 본 패치를 적용하면 --vr-input hand 모드 (Quest hand-tracking 텔레옵) 가 동작하지
   않는다. 본 워크스페이스는 controller 모드를 표준으로 사용하므로 영향 없음.
   필요 시 'restore' 로 되돌릴 것.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path


CHUNKS_DIR_REL = "lib/python{py_ver}/site-packages/vuer/client_build/assets/chunks"

# 패치 대상 패턴 (find → replace).
#
# WebXR session 의 optionalFeatures 에 'hand-tracking' 이 들어가는 건 bX() 함수의
# default `handTracking:r=!KSe()` (Mac 외 플랫폼에서 r=true). 이것만 false 로 바꾸면
# WebXR session 에서 hand-tracking feature 가 요청되지 않아 Quest 가 controller
# 우선 모드로 동작.
#
# 반대로 `HX({hand:!0,handTracking:!0})` 의 handTracking 값은 vuer 의 내부 XR
# store config 에 영향 — 일부 코드 경로에서 CONTROLLER_MOVE event 전파에 필요할
# 가능성이 있어 건드리지 않는다 (시행착오: 이걸 false 로 바꾸면 controller 가
# 헤드셋에 보이지만 CONTROLLER_MOVE event 가 server 로 전송되지 않는 부작용 확인).
PATCH_RULES = [
    # platform-aware default 들 — !XXX() 가 Mac/Apple 체크 함수. Quest 에선 true 가 됨.
    # 함수명은 chunk 별로 다름 (난독화). 일반화 패턴.
    ("handTracking:r=!KSe()",               "handTracking:r=!1"),
    ("handTracking:r=!ZSe()",               "handTracking:r=!1"),
    ("handTracking:te=!isAppleVisionPro()", "handTracking:te=!1"),
    # WebSocket URL bug fix — HTTPS branch 가 hostname 뒤에 port 누락:
    # 원본: `wss://${window.location.hostname}` (port 없음 → 443 default → 우리 8012 서버 못 만남)
    # 수정: `wss://${window.location.hostname}:${window.location.port}` (현재 페이지 port 그대로)
    # 사용자가 `https://127.0.0.1:8012` 또는 `https://<ip>:8012` 어느 쪽으로 와도 항상 같은 port 로
    # WebSocket 연결 시도 → 우리 서버 hit.
    ("`wss://${window.location.hostname}`",
     "`wss://${window.location.hostname}:${window.location.port}`"),
]


def _find_chunks_dir() -> Path:
    """현재 활성 python 의 site-packages 의 vuer client_build/assets/chunks 디렉토리."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    # sys.prefix/lib/pythonX.Y/site-packages/vuer/client_build/...
    cand = Path(sys.prefix) / "lib" / f"python{py_ver}" / "site-packages" / \
           "vuer" / "client_build" / "assets" / "chunks"
    if cand.is_dir():
        return cand
    # fallback: import vuer
    try:
        import vuer  # type: ignore
        return Path(vuer.__file__).parent / "client_build" / "assets" / "chunks"
    except Exception:
        raise SystemExit(f"vuer chunks dir not found (tried {cand})")


def _target_files(chunks_dir: Path) -> list[Path]:
    """패치 룰 중 하나의 needle (원본) 또는 그 replacement (패치본) 가 들어있는 chunk."""
    out = []
    for f in sorted(chunks_dir.glob("chunk-*.js")):
        try:
            data = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(needle in data or repl in data for needle, repl in PATCH_RULES):
            out.append(f)
    return out


def _status(chunks_dir: Path) -> int:
    targets = _target_files(chunks_dir)
    n_orig_present = 0
    n_patched      = 0
    n_unpatched    = 0
    for f in targets:
        orig = f.with_suffix(f.suffix + ".orig")
        data = f.read_text(encoding="utf-8", errors="ignore")
        has_unpatched = any(needle in data for needle, _ in PATCH_RULES)
        if orig.exists():
            n_orig_present += 1
        if has_unpatched:
            n_unpatched += 1
        else:
            n_patched += 1
    print(f"chunks_dir = {chunks_dir}")
    print(f"target files (containing any rule needle or already patched): {len(targets)}")
    print(f"  with .orig backup     : {n_orig_present}")
    print(f"  currently UNPATCHED   : {n_unpatched}")
    print(f"  currently PATCHED     : {n_patched}")
    if n_unpatched == 0 and n_patched > 0:
        print("→ state: PATCHED (handTracking disabled)")
    elif n_patched == 0 and n_unpatched > 0:
        print("→ state: ORIGINAL")
    else:
        print("→ state: MIXED")
    return 0


def _disable(chunks_dir: Path) -> int:
    targets = _target_files(chunks_dir)
    if not targets:
        # 이미 패치됐을 수 있음 — backups 있는지 본다.
        backups = list(chunks_dir.glob("chunk-*.js.orig"))
        if backups:
            print(f"(no unpatched targets; {len(backups)} backups exist — likely already disabled)")
            return 0
        print("no targets matched any patch rule. vuer version 다를 가능성.")
        return 0
    changed = 0
    for f in targets:
        orig = f.with_suffix(f.suffix + ".orig")
        if not orig.exists():
            shutil.copy2(f, orig)
        data = f.read_text(encoding="utf-8", errors="ignore")
        new = data
        for needle, repl in PATCH_RULES:
            new = new.replace(needle, repl)
        if new != data:
            f.write_text(new, encoding="utf-8")
            changed += 1
            print(f"  patched: {f.name}")
    print(f"done: {changed}/{len(targets)} chunk(s) modified.")
    return 0


def _restore(chunks_dir: Path) -> int:
    backups = list(chunks_dir.glob("chunk-*.js.orig"))
    if not backups:
        print("no .orig backups found — nothing to restore.")
        return 0
    for orig in backups:
        target = orig.with_suffix("")  # strip .orig (e.g., chunk-X.js.orig → chunk-X.js)
        shutil.copy2(orig, target)
        print(f"  restored: {target.name}")
    print(f"done: {len(backups)} chunk(s) restored from .orig.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["disable", "restore", "status"])
    args = ap.parse_args()

    chunks_dir = _find_chunks_dir()
    if args.mode == "status":
        return _status(chunks_dir)
    if args.mode == "disable":
        return _disable(chunks_dir)
    if args.mode == "restore":
        return _restore(chunks_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
