# 설치 가이드 (Ubuntu 22.04 + Python 3.8)

본 문서는 G1_Teleoperation 워크스페이스 + `teleop` conda env 를 새 머신에서
**처음부터** 재현 가능하도록 설치하는 정확한 절차입니다. 2026-05-20 로컬
머신 (Ubuntu 22.04, sandbox) 에서 검증된 시퀀스 그대로.

> **Quest 3 USB 연결** 은 별도 파일 [`QUEST3_SETUP.md`](QUEST3_SETUP.md) 참고.
> **하드웨어 통신 사양** 은 [`HARDWARE.md`](HARDWARE.md) 참고.

---

## 0. 사전 요구사항

- Ubuntu 22.04 (또는 Linux 22.04+)
- conda / miniconda 또는 anaconda 설치됨
- sudo 권한 (시스템 패키지 + udev rule 작성 시)
- 인터넷 (PyPI / conda-forge / github 접근)

---

## 1. conda env 생성 (`teleop`)

```bash
# python 3.8 base
conda create -n teleop python=3.8 -c conda-forge -y
conda activate teleop
python -m pip install --upgrade pip wheel setuptools
```

설치 후 검증:
```bash
python --version    # → Python 3.8.20
which python pip    # → /home/<user>/miniconda3/envs/teleop/bin/...
```

---

## 2. Core scientific stack

```bash
# conda-forge: pinocchio + casadi (IK 의존성, native bindings)
conda install -c conda-forge -y pinocchio casadi
# → pinocchio 3.2.0 / casadi 3.6.5 (2026-05 시점)

# pip: numpy<2 + scipy + opencv-contrib HEADLESS + 데이터 IO
# 주의: opencv-contrib-python-headless 사용 (cv2.aruco 호출 — workers/worker_zed.py).
#       GUI 번들(=비-headless)은 자신만의 Qt5 를 site-packages/cv2/qt/ 로 싣는데,
#       그게 PyQt5 의 Qt 와 충돌해 ("Could not load Qt platform plugin xcb") main.py 가
#       GUI 띄우지 못함. workspace 가 cv2.imshow 미사용이므로 headless 가 정답.
#       setup.py 가 같은 패키지를 ==4.10.0.82 로 pin 해 §6 단계에서 재설치됨.
pip install \
    'numpy<2' \
    scipy \
    'opencv-contrib-python-headless<4.11' \
    pyarrow \
    pandas \
    pyyaml \
    'imageio[ffmpeg]' \
    pyqt5
```

검증:
```bash
python -c "
import numpy, scipy, cv2, pandas, pyarrow, yaml, imageio, PyQt5, pinocchio, casadi
print('all core OK')
"
```

---

## 3. Vuer (WebXR for Quest 3) — Python 3.8 호환 fix 필요

Vuer 의 의존성 `params_proto` 는 최근 버전이 PEP 604 generic (`tuple[X, Y]`)
syntax 를 사용해 Python 3.8 에서 `TypeError: 'type' object is not subscriptable`
로 깨집니다. `from __future__ import annotations` 한 줄로 lazy-eval 시켜 우회.

### 3-1. Vuer + params_proto 설치
```bash
# params_proto 3.x 부터 vuer 0.0.60 이 필요로 하는 `Flag` symbol 이 제거되었으므로
# 2.x 라인 (마지막은 2.13.2) 으로 pin. 2.x 는 Python 3.8 PEP 604 이슈도 없어
# §3-2 monkey-patch 가 불필요.
pip install --no-deps 'vuer==0.0.60' 'params_proto>=2.12,<3.0'
pip install aiohttp aiohttp-cors websockets msgpack dotvar pillow
```

### 3-3. (controller 모드 필수) Vuer WebXR hand-tracking 기본값 OFF

vuer 0.0.60 의 client JS chunk 에 `handTracking:!0` 가 hardcode 되어 있어, Quest 3 의
WebXR session 진입 시 optionalFeatures 에 `'hand-tracking'` 이 항상 포함됨. Quest 의
auto-switch 정책상 hands 가 헤드셋 카메라에 보이면 controller input 이 demote 되어
`--vr-input controller` 모드 운용이 사실상 불가 (헤드셋 안에서 컨트롤러 인식 안 됨).

teleop env 처음 설치 후 한 번 수행:
```bash
python scripts/patch_vuer_xr.py disable   # 4 chunk 의 handTracking 기본값을 false 로 패치, .orig 백업
python scripts/patch_vuer_xr.py status    # 상태 확인 ('PATCHED (handTracking disabled)' 떠야 함)
```

⚠️ 본 패치 적용 시 `--vr-input hand` 모드는 동작하지 않음 (Quest 가 hand-tracking
   permission 없이 session 시작 → HAND_MOVE 이벤트 없음). 본 워크스페이스는 controller
   를 표준 입력으로 사용하므로 무관. 필요시 `python scripts/patch_vuer_xr.py restore`.

### 3-2. (옛 절차) params_proto monkey-patch
§3-1 에서 2.x 로 pin 하면 **불필요**. 만약 3.x 로 두면 Python 3.8 에서
`TypeError: 'type' object is not subscriptable` 가 envvar.py 에서 발생 — 그 경우에만
다음 패치를 적용한다.
```bash
ENVVAR_FILE=$(python -c "
import params_proto, os
print(os.path.join(os.path.dirname(params_proto.__file__), 'envvar.py'))
")
if ! grep -q '^from __future__ import annotations' "$ENVVAR_FILE"; then
    { echo 'from __future__ import annotations'; cat "$ENVVAR_FILE"; } > "$ENVVAR_FILE.tmp"
    mv "$ENVVAR_FILE.tmp" "$ENVVAR_FILE"
fi
```

검증:
```bash
python -c "
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers
print('vuer + Hands + MotionControllers all OK')
"
```

---

## 4. Robot SDKs

### 4-1. Unitree SDK2 Python (G1 + DEX3 DDS)

PyPI 에 없음 — github clone + `pip install -e`.

```bash
WORK=~/   # 또는 원하는 위치
git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git "$WORK/unitree_sdk2_python"
pip install -e "$WORK/unitree_sdk2_python"
```

검증:
```bash
python -c "
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandCmd_, HandState_
print('unitree_sdk2py G1 + DEX3 IDL OK')
"
```

### 4-2. cyclonedds (Unitree DDS backend)
```bash
pip install cyclonedds
```

### 4-3. Inspire SDK (Inspire RH56 손 사용 시)
- inspire_sdkpy 는 Inspire 공식 repo 에서 clone + pip install.
- 미사용 시 (DEX3 만) skip 가능.

### 4-4. Dynamixel SDK — **필수** (운용 모드 무관)
`workers/worker_g1_ctrl.py` 가 `g1_control.g1_head_dynamixel` 을 static import →
`dynamixel_sdk` 가 없으면 worker import 단계에서 실패. `--head off` 운용이라도 SDK 자체는
설치되어 있어야 한다.
```bash
pip install dynamixel_sdk
```

### 4-5. logging_mp (로깅) — API alias 추가

`logging-mp` PyPI 패키지는 표준 logging 식 `getLogger` / `basicConfig` 만 제공하나
본 워크스페이스 코드는 snake_case `get_logger` / `basic_config` 사용. site-packages
에 alias 한 줄씩 추가:

```bash
pip install logging_mp

LMP_INIT=$(python -c "
import logging_mp, os
print(os.path.join(os.path.dirname(logging_mp.__file__), '__init__.py'))
")
grep -q '^get_logger = getLogger' "$LMP_INIT" || echo 'get_logger = getLogger' >> "$LMP_INIT"
grep -q '^basic_config = basicConfig' "$LMP_INIT" || echo 'basic_config = basicConfig' >> "$LMP_INIT"
```

검증:
```bash
python -c "
import logging_mp
assert logging_mp.get_logger is logging_mp.getLogger
assert logging_mp.basic_config is logging_mp.basicConfig
print('logging_mp aliases OK')
"
```

---

## 5. 카메라 SDK (선택)

### 5-1. RealSense (D435i/455/405)
```bash
pip install pyrealsense2
```
검증: 카메라 USB 연결 후
```bash
python -c "
import pyrealsense2 as rs
ctx = rs.context()
for d in ctx.query_devices():
    print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
"
```

### 5-2. ZED (2i / Mini)
1. ZED SDK 설치: https://www.stereolabs.com/developers/release
2. Python API:
```bash
python /usr/local/zed/get_python_api.py
```
검증:
```bash
python -c "
import pyzed.sl as sl
for d in sl.Camera.get_device_list():
    print(d.serial_number, str(d.camera_model))
"
```

> **카메라 둘 다 미사용 시** 이 §5 전체 skip 가능. main.py `--camera none` 사용.

---

## 6. 워크스페이스 install

```bash
cd /path/to/G1_Teleoperation
pip install -e .
```

`setup.py` 에는 `install_requires` 가 등록되어 있어 이 단계에서 다음 패키지들이 추가
설치된다 (수십 분 소요): torch==2.3.0, torchvision==0.18.0, scikit-learn==1.3.2,
nlopt, anytree, trimesh, pytransform3d, h5py, glfw, pyqtgraph, matplotlib,
opencv-contrib-python==4.10.0.82 (§2 와 동일 패키지 — pip 가 같은 자리로 재설치),
`params-proto==2.12.1` 로 다운그레이드됨 (2.12.1 에도 `Flag` 가 있어 vuer 정상 동작).

`aiortc` 는 코드 어디서도 import 안 되는 dead-dep 이라 `setup.py` 에서 제거 (2026-05).

---

## 7. Quest 3 측 사전 준비

OS 와 conda env 와 별개로 Quest 3 헤드셋 안의 Developer Mode + USB Debugging 활성화
+ udev rule 작성이 필요. 상세는 [`QUEST3_SETUP.md`](QUEST3_SETUP.md) 참고.

요약:
```bash
# jammy 이상 — 패키지명이 'adb' (옛 'android-tools-adb' 는 obsolete)
sudo apt-get install -y adb
sudo tee /etc/udev/rules.d/51-android.rules > /dev/null <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG plugdev $USER
# 그룹 변경 반영을 위해 로그아웃 후 재로그인
# udev rule 작성 전에 이미 Quest 가 꽂혀 있었다면, USB 케이블을 한 번 뽑았다 다시 연결해
# 새 add 이벤트로 rule 이 적용되도록 한다. (`udevadm trigger` 만으로는 이미 attached 된
# 디바이스의 노드 권한이 안 바뀌는 경우가 있음.)
```

Quest 3 헤드셋:
- Settings → System → Developer → USB Debugging ON
- USB-C 케이블 연결 → 헤드셋 안 dialog "이 컴퓨터에서 항상 허용" → 확인

---

## 8. Vuer HTTPS 자기서명 cert (한 번만)

Quest 3 안 브라우저는 https 연결만 허용 → 자기서명 cert/key 필요.
```bash
cd /path/to/G1_Teleoperation
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
    -sha256 -days 365 -nodes \
    -subj "/C=KR/ST=Seoul/L=Seoul/O=YourOrg/OU=Robotics/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

`.gitignore` 에 이미 `*.pem` 가 포함되어 있어 commit 되지 않습니다.

---

## 9. 종합 검증

```bash
cd /path/to/G1_Teleoperation
conda activate teleop
QT_QPA_PLATFORM=offscreen python scripts/verify_offline.py
```

기대 출력:
```
SUMMARY  PASS=9  FAIL=0
all offline checks passed — codebase + env integrity OK.
```

이 9 단계 모두 PASS 면 코드베이스 + 환경 무결성 완료. 다음으로:
- Quest 3 검증: [`QUEST3_SETUP.md`](QUEST3_SETUP.md) §3~5
- 실 운용: [`../README.md`](../README.md) §6
- 평가: [`../README.md`](../README.md) §9

---

## 부록 A — 2026-05-21 새 머신 재검증된 정확한 버전

```
python                    3.8.20
numpy                     1.24.4
scipy                     1.10.1
opencv-contrib-python-headless  4.10.0.82  (setup.py pin — cv2.aruco 필요, Qt 번들 X)
pandas                    2.0.3
pyarrow                   17.0.0
PyYAML                    6.0.3
imageio                   2.35.1
imageio-ffmpeg            0.5.1
PyQt5                     5.15.x
pinocchio                 3.2.0  (conda-forge)
casadi                    3.6.5  (conda-forge)
vuer                      0.0.60 (--no-deps)
params_proto              2.12.1 (setup.py 가 ==2.12.1 로 pin)
aiohttp                   3.10.11
aiohttp-cors              0.7.0
websockets                13.1
msgpack                   1.1.1
dotvar                    0.1.1
pillow                    10.4.0
cyclonedds                0.10.2 (unitree_sdk2py dep)
unitree_sdk2py            github HEAD (pip install -e)
logging-mp                0.2.1 + alias
dynamixel_sdk             4.0.5  (필수 — worker_g1_ctrl static import)
pyrealsense2              2.55.x (옵션 — RealSense)
pyzed.sl                  ZED SDK 5.x (옵션 — ZED)
torch                     2.3.0+cu121 (setup.py)
torchvision               0.18.0      (setup.py)
scikit-learn              1.3.2       (setup.py)
nlopt                     2.7.1       (setup.py)
anytree                   2.12.1      (setup.py)
trimesh                   4.12.x      (setup.py)
pytransform3d             3.5.0       (setup.py)
h5py                      3.11.0      (setup.py)
glfw, pyqtgraph 0.13.3, matplotlib 3.7.5 (setup.py)
```

## 부록 B — 자주 만나는 오류

| 증상 | 원인 | 해법 |
|---|---|---|
| `TypeError: 'type' object is not subscriptable` (params_proto) | PEP 604 generic on Python 3.8 | §3-2 monkey-patch |
| `ModuleNotFoundError: No module named 'unitree_sdk2py'` | github 미설치 | §4-1 git clone + pip install -e |
| `module 'logging_mp' has no attribute 'get_logger'` | API 명 불일치 | §4-5 alias 추가 |
| `RuntimeError: Logging system has already been started` | main.py 의 basic_config 가 늦게 호출됨 | main.py:54 의 try/except 가 이미 적용 (코드베이스 패치됨) |
| `from vuer.schemas import MotionControllers` 실패 | vuer 0.0.40 미만 | vuer 0.0.60 이상 설치 |
| `cv2 _ARRAY_API not found` | numpy 2.x ↔ cv2 binary 미스매치 | `numpy<2` 강제 설치 |

## 부록 C — Pro4000 같은 다른 머신으로 이동 시

이 INSTALL.md 절차 그대로 재현. 다음만 추가 확인:
1. `network_interface` (G1 DDS) 가 그 머신의 NIC 명으로 `utils/lan_config.yaml` 에
   반영되어 있는지 (`ip a` 로 NIC 명 확인)
2. ZED 의 경우 외부 stream 송신 PC 가 있다면 `--zed-mode stream` 옵션 가능
   (기본은 `direct` USB)
3. CUDA / GPU 가 필요한 GR00T 평가는 별도 conda env (`gr00t`) 권장 — 본 INSTALL.md
   는 teleop env 만 다룸
