from setuptools import setup, find_packages

# --------------- 사전 안내 ---------------
# 다음 항목들은 OS/드라이버/그래픽스 스택과 밀접하며 conda로 사전 설치를 권장합니다.
# - pinocchio (conda-forge)
# - PyQt5, PyQtWebEngine (conda-forge)
# 위 3가지는 install_requires에 넣지 않고, README에서 "사전 설치"로 명시하세요.

# --------------- 기본 의존성 ---------------
# 사용자가 주신 버전/범위를 최대한 반영했습니다.
# 주의: aiortc 는 코드에서 import 안 함 (workspace 가 vuer.addons.camera_rtc 미사용).
#       Python 3.8 에서 aiortc==1.8.0 → av<12 → cp38 wheel 없어 ffmpeg dev 빌드 필요.
#       2026-05 dead-dep 으로 판단해 제거.
base_requires = [
    "opencv-contrib-python==4.10.0.82",
    "params-proto==2.12.1",
    "pytransform3d==3.5.0",
    "scikit-learn==1.3.2",
    "nlopt>=2.6.1,<2.8.0",
    "trimesh>=4.4.0",
    "anytree>=2.12.0",
    "logging_mp",
    "h5py",
    "imageio",                  # 기본 imageio (ffmpeg 플러그인은 extras로 분리)
    "torch==2.3.0",
    "torchvision==0.18.0",
    "pyrealsense2",
    "glfw",
    "pyqtgraph==0.13.3",
]

# --------------- 선택적(extra) 의존성 ---------------
extras = {
    # 역기구학 (Inverse Kinematics) - conda로 pinocchio 설치 후, pip로 아래 두 개 설치
    "ik": [
        "meshcat",
        "casadi",
    ],
    # VR 입력 (hand tracking + controller, Quest3)
    # Controller 입력(MotionControllers / CONTROLLER_MOVE event) 지원을 위해 0.0.60 이상 권장.
    "vr": [
        "vuer[all]>=0.0.60,<0.1.0",
    ],
    # FFmpeg 플러그인
    "video": [
        "imageio[ffmpeg]",
    ],
    # 데이터 변환(Phase 3): Diffusion Policy zarr export
    "dataset_dp": [
        "zarr>=2.16,<3.0",
        "numcodecs",
    ],
}

# 편의상 전체 설치용 메타 extra
extras["all"] = sorted(set(sum(extras.values(), [])))

setup(
    name="teleop",
    version="0.1.0",
    description="Teleoperation project dependencies",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8,<3.9",
    install_requires=base_requires,
    extras_require=extras,
    include_package_data=True,
)
