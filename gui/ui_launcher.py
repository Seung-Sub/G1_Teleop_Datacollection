# ui_launcher.py
import os, sys

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

import logging
from threading import Thread
from PyQt5 import uic, QtWidgets, QtCore, QtGui
# from PyQt5.QtWebKitWidgets import QWebView

# Meshcat 시각화 기능 제거로 QtWebEngine import 제거됨
# from PyQt5.QtWebEngineWidgets import QWebEngineView
# from PyQt5.QtCore import QUrl
from PyQt5.QtCore    import QCoreApplication, Qt
from PyQt5.QtWidgets import QApplication

import subprocess

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA, ARUCO_MARKERS, WORKSPACE_MASK, RECORD_TASK_LAYOUT, RECORD_EPISODE_LAYOUT, \
    RECORD_MODE_LAYOUT, RIGHT_TOUCH_SENSOR_LAYOUT, LEFT_TOUCH_SENSOR_LAYOUT, WORKER_FREQ,GR00T_TASK_LAYOUT, ROBOT_OBS, ROBOT_ACTION, MASK_CONTROL_LAYOUT


import pyqtgraph as pg
from collections import deque
import time

import numpy as np
import cv2

# GUI 마스크 적용 토글
APPLY_MASK_IN_GUI = True

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# ZED 뷰 상태
ZED_VIEW_LEFT = "left"
ZED_VIEW_REALSENSE = "realsense"


class TeleopUI(QtWidgets.QMainWindow):
    def __init__(self, shared_event, shm_names, shared_lock):
        super().__init__()
        # .ui 파일 로드
 
        # 전달받은 리소스 저장
        self.shared_event = shared_event
        self.shm_name = shm_names
        self.shared_lock = shared_lock

        # 카메라 뷰 상태 초기화 (zed_left 또는 realsense)
        self.zed_current_view = ZED_VIEW_LEFT
        
        # 타이머 관련 변수 초기화 (그래프가 비활성화되어도 필요)
        self.start_time = time.time()
        
        # 시계열 버퍼 초기화
        self.time_data   = deque(maxlen=100)
        self.g1_data     = deque(maxlen=100)
        self.hand_data   = deque(maxlen=100)
        self.vr_data     = deque(maxlen=100)
        self.cam_data    = deque(maxlen=100)
        self.rec_data    = deque(maxlen=100)


        uic.loadUi('gui/teleop_ui.ui', self)
        self._apply_gray_theme()
        # 오른쪽 패널(Loop Hz / Mode)을 완전히 제거해 카메라 영역을 확장
        self._remove_right_panel()
        # 혹시 남아있는 Loop Hz/Mode 위젯을 강제로 숨김
        self._hide_loophz_widgets()
        # 비상 정지 버튼도 화면에서 숨김
        self._hide_emergency_button()
        # 그래프 위젯들을 숨김
        self._hide_graph_widgets()

        self._init_shared_memory()
        self._init_freq_widgets()  # 주파수 텍스트 위젯 초기화 (그래프는 숨김)
        # self._init_hz_plot() # 그래프 숨김으로 인해 주석 처리
        # self._init_webengine_view()  # Meshcat 시각화 기능 제거됨
        # self._init_joint_plot() # 그래프 숨김으로 인해 주석 처리
        self._init_btn_lb()
        self._init_touch_tables()
        self._init_timer()

        # UI 위젯과 함수 연결
        self.G1_connector_button()
        self.Hand_connector_button()
        self.VR_connector_button()
        self.Start_button()
        self.Quit_button()
        self.Emergency_button()
        self.ZED_view_toggle_button()
        # self.Web_reload()  # Meshcat 시각화 기능 제거됨




    def _clear_layout(self, layout: QtWidgets.QLayout):
        """레이아웃의 위젯/하위 레이아웃을 제거."""
        if not layout:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
            if child_layout:
                self._clear_layout(child_layout)

    def _remove_right_panel(self):
        """Loop Hz / Mode 패널을 제거하고 가로 공간을 카메라 영역으로 확장."""
        grid = self.findChild(QtWidgets.QGridLayout, "gridLayout_9")
        if not grid:
            return

        right_item = grid.itemAtPosition(0, 1)
        if right_item:
            target_layout = right_item.layout()
            if target_layout:
                self._clear_layout(target_layout)
            grid.removeItem(right_item)

        # 컬럼 폭을 전부 왼쪽(카메라)로 몰아줌
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)
        grid.setColumnMinimumWidth(1, 0)

    def _hide_emergency_button(self):
        """Emergency 버튼을 숨겨 공간을 비움."""
        btn = self.findChild(QtWidgets.QPushButton, "emergency_button")
        if btn:
            btn.setVisible(False)
            btn.setEnabled(False)

    def _hide_graph_widgets(self):
        """그래프 관련 위젯들을 숨김."""
        graph_widgets = [
            "loop_hz_graph", 
            "Waist_joint_graph", 
            "Left_arm_joint_graph", 
            "Right_arm_joint_graph", 
            "Left_hand_joint_graph", 
            "Right_hand_joint_graph"
        ]
        for name in graph_widgets:
            w = self.findChild(QtWidgets.QWidget, name)
            if w:
                w.setVisible(False)
                # 부모 레이아웃에서도 제거 시도 (공간 확보를 위해)
                # if w.parent():
                #     w.parent().layout().removeWidget(w)

        # 그래프가 포함된 탭 위젯이나 컨테이너가 있다면 그것도 숨길 수 있음
        # 예: tabWidget 등. 현재 구조에서는 개별 그래프 위젯을 숨기는 것으로 처리.

    def _hide_loophz_widgets(self):
        """Loop Hz / Mode 관련 위젯을 강제로 숨김."""
        widget_names = [
            "label_24", "label_19", "label_20", "label_21",
            "label_22", "label_23", "label_25",
            "camera_freq", "vr_freq", "hand_freq", "g1_freq", "record_freq",
            "mode", "start_label", "home_label", "reset_label", "done_label", "replay_label",
        ]
        for name in widget_names:
            w = self.findChild(QtWidgets.QWidget, name)
            if w:
                w.setVisible(False)

        grid = self.findChild(QtWidgets.QGridLayout, "gridLayout_9")
        if grid:
            grid.setColumnStretch(1, 0)
            grid.setColumnMinimumWidth(1, 0)

    def _apply_gray_theme(self):
        """전체 UI를 밝은 흰색 톤으로 설정."""
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QLabel {
                color: #000000;
            }
            QLineEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #f5f5f5;
                color: #000000;
                border: 1px solid #cccccc;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #000000;
                border: 1px solid #999999;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QProgressBar {
                background: #f5f5f5;
                color: #000000;
                border: 1px solid #cccccc;
            }
            QProgressBar::chunk {
                background: #4caf50;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
            }
            QTabBar::tab {
                background: #e0e0e0;
                color: #000000;
                padding: 6px 10px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
            }
        """)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        if key == QtCore.Qt.Key_0:          # 숫자 0
            self.func_start(self.shared_event)
        elif key == QtCore.Qt.Key_1:        # 숫자 1
            self.on_home()
        elif key == QtCore.Qt.Key_2:        # 숫자 2
            self.on_start_record()
        else:
            # 나머지 키는 기본 처리
            super().keyPressEvent(event)
    
    def _init_shared_memory(self):
        
        lock = self.shared_lock
        names = self.shm_name

        self.camera_shm         = SharedMemoryManager(CAMERA, lock["camera_lock"], names["camera_shm"])
        self.aruco_shm          = SharedMemoryManager(ARUCO_MARKERS, lock["aruco_lock"], names["aruco_shm"])
        self.workspace_mask_shm = SharedMemoryManager(WORKSPACE_MASK, lock["workspace_mask_lock"], names["workspace_mask_shm"])
        # television_shm 제거됨 (POSE INFO 기능 제거)
        self.record_task_shm    = SharedMemoryManager(RECORD_TASK_LAYOUT, lock["record_lock"], names["record_task_shm"])
        self.record_episode_shm = SharedMemoryManager(RECORD_EPISODE_LAYOUT, lock["record_lock"], names["record_episode_shm"])
        self.record_mode_shm    = SharedMemoryManager(RECORD_MODE_LAYOUT, lock["record_lock"], names["record_mode_shm"])
        self.right_touch_shm    = SharedMemoryManager(RIGHT_TOUCH_SENSOR_LAYOUT, lock["right_touch_lock"], names["right_touch_shm"])
        self.left_touch_shm     = SharedMemoryManager(LEFT_TOUCH_SENSOR_LAYOUT, lock["left_touch_lock"], names["left_touch_shm"])
        self.freq_shm           = SharedMemoryManager(WORKER_FREQ, lock["freq_lock"], names["freq_shm"])
        self.gr00t_task_shm     = SharedMemoryManager(GR00T_TASK_LAYOUT, lock["gr00t_lock"],names["gr00t_shm"])
        self.robot_obs_shm     = SharedMemoryManager(ROBOT_OBS, lock["robot_obs_lock"],names["robot_obs_shm"])
        self.robot_action_shm     = SharedMemoryManager(ROBOT_ACTION, lock["robot_action_lock"],names["robot_action_shm"])
        self.mask_control_shm   = SharedMemoryManager(MASK_CONTROL_LAYOUT, lock["record_lock"], names["mask_control_shm"])

    def _init_btn_lb(self):
        # 
        self.start_label  = self.findChild(QtWidgets.QLabel, "start_label")
        self.home_label   = self.findChild(QtWidgets.QLabel, "home_label")
        self.reset_label  = self.findChild(QtWidgets.QLabel, "reset_label")
        self.replay_label = self.findChild(QtWidgets.QLabel, "replay_label")
        self.done_label   = self.findChild(QtWidgets.QLabel, "done_label")
 
        # Logging Interface
        self.task_name_le         = self.findChild(QtWidgets.QLineEdit,   "task_name")
        self.record_info          = self.findChild(QtWidgets.QTextEdit,   "record_info")
        self.num_episodes_sb      = self.findChild(QtWidgets.QSpinBox,    "num_episodes")
        self.episode_len_sb       = self.findChild(QtWidgets.QSpinBox,    "episode_len")
        self.set_task_btn         = self.findChild(QtWidgets.QPushButton, "set_task")

        self.start_record_btn     = self.findChild(QtWidgets.QPushButton, "start_record")
        self.reset_record_btn     = self.findChild(QtWidgets.QPushButton, "reset_record")
        self.replay_episode_btn   = self.findChild(QtWidgets.QPushButton, "replay_episode")

        self.replay_episode_num_le = self.findChild(QtWidgets.QLineEdit,  "replay_episode_num")

        # POSE INFO text edit removed

        self.set_task_btn.clicked.connect(self.on_set_task)
        self.start_record_btn.clicked.connect(self.on_start_record)
        self.reset_record_btn.clicked.connect(self.on_reset_record)
        self.replay_episode_btn.clicked.connect(self.on_replay_episode)
        
        self.record_info.setPlainText("설정을 먼저 SET 해주세요.\n")

        # Gr00t Inferenece Interface
        self.gr00t_task_name_le         = self.findChild(QtWidgets.QLineEdit,   "language_instruction")
        self.deploy_btn   = self.findChild(QtWidgets.QPushButton, "deploy_btn")
        self.deploy_btn.clicked.connect(self.on_deploy)

        # 마스크 제어 버튼 추가
        self.mask_control_btn = QtWidgets.QPushButton("마스크 제어: OFF")
        self.mask_control_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                border: 2px solid #ff5252;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
            QPushButton:pressed {
                background-color: #ff3838;
            }
        """)
        self.mask_control_btn.clicked.connect(self.on_toggle_mask_control)
        self.mask_control_enabled = False

        # 기존 버튼들 근처에 추가 (예: deploy 버튼 근처)
        if hasattr(self, 'deploy_btn') and self.deploy_btn:
            parent_layout = self.deploy_btn.parent().layout()
            if parent_layout:
                parent_layout.addWidget(self.mask_control_btn)



        self.home_btn   = self.findChild(QtWidgets.QPushButton, "home_button")
        self.home_btn.clicked.connect(self.on_home)


        # Video Interface (RealSense removed, ZED camera moved to meshcat position)
        self.zed_video_label = self.findChild(QtWidgets.QLabel, "zed_video_label")
        
        # 비디오 라벨 배경색을 흰색으로 설정 (여백 제거)
        if self.zed_video_label:
            self.zed_video_label.setStyleSheet("background-color: #ffffff;")
            self.zed_video_label.setAlignment(QtCore.Qt.AlignCenter)


        self.logging_progress_bar = self.findChild(
            QtWidgets.QProgressBar, "logging_progress"
        )

    def _init_freq_widgets(self):
        """주파수 텍스트 위젯만 초기화 (그래프는 숨김)"""
        self.g1_freq       = self.findChild(QtWidgets.QTextEdit, "g1_freq")
        self.hand_freq     = self.findChild(QtWidgets.QTextEdit, "hand_freq")
        self.vr_freq       = self.findChild(QtWidgets.QTextEdit, "vr_freq")
        self.camera_freq   = self.findChild(QtWidgets.QTextEdit, "camera_freq")
        self.record_freq   = self.findChild(QtWidgets.QTextEdit, "record_freq")
        
        # 그래프 객체들은 None으로 설정 (사용하지 않음)
        self.hz_plot = None
        self.curve_g1 = None
        self.curve_hand = None
        self.curve_vr = None
        self.curve_cam = None
        self.curve_rec = None
        
        # 조인트 플롯 관련 변수도 None으로 초기화
        self.joint_plot_widgets = {}
        self.joint_time = deque(maxlen=100)
        
        self.joint_history = {
            "Waist": {"qpos": [deque(maxlen=100) for _ in range(3)],
                    "action": [deque(maxlen=100) for _ in range(3)]},
            "Left_arm": {"qpos": [deque(maxlen=100) for _ in range(7)],
                        "action": [deque(maxlen=100) for _ in range(7)]},
            "Right_arm": {"qpos": [deque(maxlen=100) for _ in range(7)],
                        "action": [deque(maxlen=100) for _ in range(7)]},
            "Left_hand": {"qpos": [deque(maxlen=100) for _ in range(6)],
                        "action": [deque(maxlen=100) for _ in range(6)]},
            "Right_hand": {"qpos": [deque(maxlen=100) for _ in range(6)],
                        "action": [deque(maxlen=100) for _ in range(6)]},
        }

        self.joint_subcolors = {
            "Waist":      [(255,0,0), (200,0,0), (150,0,0)],
            "Left_arm":   [(0,255,0), (0,200,0), (0,150,0), (0,100,0), (0,50,0), (0,25,0), (0,0,0)],
            "Right_arm":  [(0,0,255), (0,0,200), (0,0,150), (0,0,100), (0,0,50), (0,0,25), (0,0,0)],
            "Left_hand":  [(255,0,255), (200,0,200), (150,0,150), (100,0,100), (50,0,50), (25,0,25)],
            "Right_hand": [(0,255,255), (0,200,200), (0,150,150), (0,100,100), (0,50,50), (0,25,25)],
        }

    def _init_hz_plot(self):
        layout = QtWidgets.QVBoxLayout(self.loop_hz_graph)
        layout.setContentsMargins(0, 0, 0, 0)

        # 2) PlotWidget 생성 후 레이아웃에 추가
        self.hz_plot = pg.PlotWidget()
        layout.addWidget(self.hz_plot)
        self.hz_plot.setBackground('w')
        # 3) 축 범위·그리드·범례 설정
        # self.hz_plot.setYRange(48, 52)
        self.hz_plot.showGrid(x=True, y=True)
        self.hz_plot.addLegend()

        # 2) 모든 곡선을 실선으로 통일하고 각기 다른 색 지정
        red_pen    = pg.mkPen(color=(255, 0, 0),   width=2)  # G1
        green_pen  = pg.mkPen(color=(0, 255, 0),   width=2)  # Hand
        blue_pen   = pg.mkPen(color=(0, 0, 255),   width=2)  # VR
        orange_pen = pg.mkPen(color=(255, 165, 0), width=2)  # Camera
        purple_pen = pg.mkPen(color=(128, 0, 128), width=2)  # Record

        # 3) 곡선 생성 시 스타일과 색 반영
        self.curve_g1   = self.hz_plot.plot(pen=red_pen,    name="G1")
        self.curve_hand = self.hz_plot.plot(pen=green_pen,  name="Hand")
        self.curve_vr   = self.hz_plot.plot(pen=blue_pen,   name="VR")
        self.curve_cam  = self.hz_plot.plot(pen=orange_pen, name="Camera")
        self.curve_rec  = self.hz_plot.plot(pen=purple_pen, name="Record")


        # __init__에 추가
        self.start_time = time.time()

        # 시계열 버퍼 (최근 100개)
        self.time_data   = deque(maxlen=100)
        self.g1_data     = deque(maxlen=100)
        self.hand_data   = deque(maxlen=100)
        self.vr_data     = deque(maxlen=100)
        self.cam_data    = deque(maxlen=100)
        self.rec_data    = deque(maxlen=100)


        self.g1_freq       = self.findChild(QtWidgets.QTextEdit, "g1_freq")
        self.hand_freq     = self.findChild(QtWidgets.QTextEdit, "hand_freq")
        self.vr_freq       = self.findChild(QtWidgets.QTextEdit, "vr_freq")
        self.camera_freq   = self.findChild(QtWidgets.QTextEdit, "camera_freq")
        self.record_freq   = self.findChild(QtWidgets.QTextEdit, "record_freq")

    def _init_webengine_view(self):
        # Meshcat 시각화 기능 제거됨
        pass

    def _init_touch_tables(self):
        """
        SharedMemory에서 받은 키(field name)와 UI에 정의된 QTableWidget 이름을 연결.
        LEFT_TOUCH_SENSOR_LAYOUT, RIGHT_TOUCH_SENSOR_LAYOUT의 필드명이 이 매핑의 키가 됩니다.
        """
        # 각 데이터 키 → 위젯 이름 매핑 사전
        mapping = {
            # 왼손
            "l_fingerone_tip_touch":     "l_l_t",  # 작은손가락 끝
            "l_fingerone_top_touch":     "l_l_n",  # 작은손가락 손톱
            "l_fingerone_palm_touch":    "l_l_p",  # 작은손가락 패드

            "l_fingertwo_tip_touch":     "l_r_t",  # 약지 끝
            "l_fingertwo_top_touch":     "l_r_n",  # 약지 손톱
            "l_fingertwo_palm_touch":    "l_r_p",  # 약지 패드

            "l_fingerthree_tip_touch":   "l_m_t",  # 중지 끝
            "l_fingerthree_top_touch":   "l_m_n",  # 중지 손톱
            "l_fingerthree_palm_touch":  "l_m_p",  # 중지 패드

            "l_fingerfour_tip_touch":    "l_i_t",  # 검지 끝
            "l_fingerfour_top_touch":    "l_i_n",  # 검지 손톱
            "l_fingerfour_palm_touch":   "l_i_p",  # 검지 패드

            "l_fingerfive_tip_touch":    "l_t_t",  # 엄지 끝
            "l_fingerfive_top_touch":    "l_t_n",  # 엄지 손톱
            "l_fingerfive_middle_touch": "l_t_m",  # 엄지 중간 섹션
            "l_fingerfive_palm_touch":   "l_t_p",  # 엄지 패드

            "l_palm_touch":              "l_p",    # 손바닥
            # 오른손
            "r_fingerone_tip_touch":     "r_l_t",
            "r_fingerone_top_touch":     "r_l_n",
            "r_fingerone_palm_touch":    "r_l_p",

            "r_fingertwo_tip_touch":     "r_r_t",
            "r_fingertwo_top_touch":     "r_r_n",
            "r_fingertwo_palm_touch":    "r_r_p",

            "r_fingerthree_tip_touch":   "r_m_t",
            "r_fingerthree_top_touch":   "r_m_n",
            "r_fingerthree_palm_touch":  "r_m_p",

            "r_fingerfour_tip_touch":    "r_i_t",
            "r_fingerfour_top_touch":    "r_i_n",
            "r_fingerfour_palm_touch":   "r_i_p",

            "r_fingerfive_tip_touch":    "r_t_t",
            "r_fingerfive_top_touch":    "r_t_n",
            "r_fingerfive_middle_touch": "r_t_m",
            "r_fingerfive_palm_touch":   "r_t_p",

            "r_palm_touch":              "r_p",
        }

        self.touch_tables = {}
        for field_name, widget_name in mapping.items():
            tbl = self.findChild(QtWidgets.QTableWidget, widget_name)
            if tbl:
                self.touch_tables[field_name] = tbl

    def _init_joint_plot(self):
        self.joint_plot_widgets = {}
        self.joint_time = deque(maxlen=100)

        # 부위별 PlotWidget 추가
        for name in ["Waist_joint_graph", "Left_arm_joint_graph", "Right_arm_joint_graph", 
                    "Left_hand_joint_graph", "Right_hand_joint_graph"]:
            container = self.findChild(QtWidgets.QWidget, name)
            if container:
                layout = QtWidgets.QVBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                plot_widget = pg.PlotWidget()
                layout.addWidget(plot_widget)
                self.joint_plot_widgets[name] = plot_widget

        self.time_data = deque(maxlen=100)

        self.joint_history = {
            "Waist": {"qpos": [deque(maxlen=100) for _ in range(3)],
                    "action": [deque(maxlen=100) for _ in range(3)]},
            "Left_arm": {"qpos": [deque(maxlen=100) for _ in range(7)],
                        "action": [deque(maxlen=100) for _ in range(7)]},
            "Right_arm": {"qpos": [deque(maxlen=100) for _ in range(7)],
                        "action": [deque(maxlen=100) for _ in range(7)]},
            "Left_hand": {"qpos": [deque(maxlen=100) for _ in range(6)],
                        "action": [deque(maxlen=100) for _ in range(6)]},
            "Right_hand": {"qpos": [deque(maxlen=100) for _ in range(6)],
                        "action": [deque(maxlen=100) for _ in range(6)]},
        }

        self.joint_subcolors = {
            "Waist":      [(255,0,0), (200,0,0), (150,0,0)],        # 3개 joint
            "Left_arm":   [(0,255,0), (0,200,0), (0,150,0), (0,100,0), (0,50,0), (0,25,0), (0,0,0)],
            "Right_arm":  [(0,0,255), (0,0,200), (0,0,150), (0,0,100), (0,0,50), (0,0,25), (0,0,0)],
            "Left_hand":  [(255,0,255), (200,0,200), (150,0,150), (100,0,100), (50,0,50), (25,0,25)],
            "Right_hand": [(0,255,255), (0,200,200), (0,150,150), (0,100,100), (0,50,50), (0,25,25)],
        }
        for pw in self.joint_plot_widgets.values():
            pw.addLegend()
            pw.setBackground('w')

    def _init_timer(self):
        # ▶ 3) QTimer 설정 (예: 30fps 정도로 읽어오기)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        fps = 30
        self.timer.start(int(1000 / fps))   # 1000ms / 30 ≈ 33ms 간격

        # POSE INFO removed

        # 4) QTimer 생성
        #    -> 100ms마다 호출되도록 설정 (데이터 수집 주기에 맞춰 변경 가능)
        self.text_timer = QtCore.QTimer(self)
        self.text_timer.setInterval(100)                # 밀리초 단위, 예: 100ms
        self.text_timer.timeout.connect(self.read_from_shm)
        self.text_timer.start()                         # 타이머 시작



    # ────────────────────────────────────────────────────────────
    #   1) Web Reload: meshcat 기능 제거됨
    # ────────────────────────────────────────────────────────────
    def Web_reload(self):
        # Meshcat 시각화 기능 제거됨
        pass

    
    # ────────────────────────────────────────────────────────────
    #   2) G1 / Hand / VR / Start / Emergency 버튼 바인딩
    # ────────────────────────────────────────────────────────────
    def G1_connector_button(self):
        if hasattr(self, 'G1_connector'):
            self.G1_connector.clicked.connect(lambda: self.func_g1(self.shared_event))

    def Hand_connector_button(self):
        if hasattr(self, 'Hand_connector'):
            self.Hand_connector.clicked.connect(lambda: self.func_hand(self.shared_event))

    def VR_connector_button(self):
        if hasattr(self, 'VR_connector'):
            self.VR_connector.clicked.connect(lambda: self.func_vr())

    def Start_button(self):
        if hasattr(self, 'Start_button'):
            self.start_button.clicked.connect(lambda: self.func_start(self.shared_event))

    def Quit_button(self):
        if hasattr(self, 'Quit_button'):
            self.quit_button.clicked.connect(lambda: self.func_quit(self.shared_event))

    def Emergency_button(self):
        if hasattr(self, 'emergency_button'):
            self.emergency_button.clicked.connect(lambda: self.func_emergency(self.shared_event))

    def ZED_view_toggle_button(self):
        if hasattr(self, 'zed_view_toggle_button'):
            self.zed_view_toggle_button.clicked.connect(self.toggle_zed_view)


    def toggle_zed_view(self):
        """카메라 뷰를 zed_left ↔ realsense로 전환"""
        if self.zed_current_view == ZED_VIEW_LEFT:
            self.zed_current_view = ZED_VIEW_REALSENSE
            self.zed_view_toggle_button.setText("RealSense View")
            print("[GUI] 카메라 뷰: RealSense로 전환")
        else:
            self.zed_current_view = ZED_VIEW_LEFT
            self.zed_view_toggle_button.setText("ZED Left View")
            print("[GUI] 카메라 뷰: ZED Left로 전환")




    def func_quit(self, shared_event):
        # 만약 이미 set_start가 set돼 있다면, clear() 해서 워커를 대기 상태로 전환
        shared_event["shutdown"].set()

    def func_start(self, shared_event):
        # 만약 이미 set_start가 set돼 있다면, clear() 해서 워커를 대기 상태로 전환
        if shared_event['set_start'].is_set():
            shared_event['set_start'].clear()

        else:
            shared_event['set_start'].set()

    def func_g1(self,shared_event):
        if not shared_event['set_g1'].is_set():  
            shared_event['set_g1'].set()

    def func_hand(self, shared_event):
        if not shared_event['set_hand'].is_set():  
            shared_event['set_hand'].set()

    def func_vr(self):
        logger_mp.info(f"func_vr")
        try:
            result = subprocess.run(
                ["adb", "reverse", "tcp:8012", "tcp:8012"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger_mp.info("ADB reverse command executed successfully")
            logger_mp.info(result.stdout)
        except FileNotFoundError:
            logger_mp.info("ADB command not found. Ensure that adb is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            logger_mp.info("Failed to execute adb reverse command:")
            logger_mp.info(e.stderr)


    def func_emergency(self,shared_event):
        if not shared_event['emergency'].is_set():  
            logger_mp.info(f"func_emergency")
            shared_event['emergency'].set()

    def on_home(self):  
        record_mode_data = self.record_mode_shm.read_data()
        record_mode_data["home"] = True
        self.record_mode_shm.write_data(**record_mode_data)
        self.record_info.append("Set home")



    def on_set_task(self):
        """
        SET 버튼을 눌렀을 때:
          1) task_name, num_episodes, episode_len을 SHM에 기록
          2) 버튼 플래그 초기화 (start/reset/replay 모두 False)
          3) record_info 창에 “설정 완료” 메시지 출력
        """
        # 1) 입력값 읽어오기
        task_text = self.task_name_le.text().strip()
        num_eps   = self.num_episodes_sb.value()
        ep_len    = self.episode_len_sb.value()

        if task_text == "":
            self.record_info.append("[ERROR] Task Name이 입력되지 않았습니다.")
            return
        logger_mp.info(f"SET: task={task_text}, num_episodes={num_eps}, episode_len={ep_len}")

        # 2) SHM에 값 기록 (write_data 사용 가정)


        try:
            self.record_task_shm.write_data(
                task_name    = task_text,
            )
            self.record_episode_shm.write_data(
                num_episodes = np.int32(num_eps),
                episode_len  = np.int32(ep_len),
                replay_idx   = np.int32(0),
                episode_index   = np.int32(0),
            )
            self.record_mode_shm.write_data(
                start        = np.bool_(False),
                reset        = np.bool_(False),
                replay       = np.bool_(False),
                done         = np.bool_(False),
                deploy       = np.bool_(False) 
            )
            self.record_info.setPlainText(f"[INFO] Task: {task_text}\n"
                                          f"[INFO] num_episodes = {num_eps}, episode_len = {ep_len}(s)\n"
                                          "[INFO] SET 완료.\n")
            logger_mp.info(f"SET: task={task_text}, num_episodes={num_eps}, episode_len={ep_len}")
        except Exception as e:
            self.record_info.append(f"[ERROR] SHM Write 실패: {e}")
            logger_mp.error(f"SHM Write 실패 on_set_task: {e}")

    def on_start_record(self):
        """
        START 버튼을 눌렀을 때:
          - SHM의 'start' 필드를 toggle (False->True, 또는 True->False)  
          - True로 변경되면 consumer(Worker)가 레코드 루프를 시작함
        """
        try:
            self.record_mode_shm.write_data(reset=np.bool_(False),
                                            start=np.bool_(True),
                                            replay=np.bool_(False),
                                            done=np.bool_(False),
                                            deploy= np.bool_(False) )

            record_episode_dict = self.record_episode_shm.read_data()
            episode_index      = record_episode_dict["episode_index"]   


            self.record_info.append(f"{episode_index} 에피소드 녹화 시작")
            logger_mp.info("레코드 START")
            
        except Exception as e:
            self.record_info.append(f"[ERROR] START 처리 실패: {e}")
            logger_mp.error(f"on_start_record Exception: {e}")

    def on_reset_record(self):
        """
        RESET 버튼을 눌렀을 때:
          - SHM의 'reset' 필드를 True로 세팅 → consumer가 중단 플래그로 인식
          - 이후 consumer 측에서 'reset' 처리가 완료되면 다시 False로 값 변경하도록 설계 필요
        """

        try:
            # 단순히 True로 설정
            self.record_mode_shm.write_data(reset=np.bool_(True),
                                            start=np.bool_(False),
                                            replay=np.bool_(False),
                                            done=np.bool_(False),
                                            deploy= np.bool_(False) )
            self.record_info.append("[INFO] RESET 신호 전송 → 레코드 중단 요청")
            logger_mp.info("레코드 RESET 신호 전송")
        except Exception as e:
            self.record_info.append(f"[ERROR] RESET 처리 실패: {e}")
            logger_mp.error(f"on_reset_record Exception: {e}")


    def on_replay_episode(self):
        """
        REPLAY 버튼을 눌렀을 때:
          - 입력창(replay_episode_num)에서 번호 읽어 SHM의 'replay_idx'에 기록
          - SHM의 'replay' 필드를 True로 세팅 → consumer가 재생 동작 수행
          - 이후 consumer 측에서 'replay' 처리가 완료되면 False로 리셋
        """
        text = self.replay_episode_num_le.text().strip()
        if text == "":
            self.record_info.append("[ERROR] REPLAY할 Episode 번호를 입력하세요.")
            return

        try:
            idx = int(text)
        except ValueError:
            self.record_info.append("[ERROR] REPLAY할 Episode 번호는 정수여야 합니다.")
            return

        try:
            self.record_episode_shm.write_data(
                replay_idx = np.int32(idx)
            )
            self.record_mode_shm.write_data(reset=np.bool_(False),
                                            start=np.bool_(False),
                                            replay=np.bool_(True),
                                            done=np.bool_(False),
                                            deploy= np.bool_(False) 

                                            )
            
            self.shared_event['set_start'].set()

            self.record_info.append(f"[INFO] REPLAY 신호 전송 → Episode {idx} 재생 요청")
            logger_mp.info(f"REPLAY 요청: Episode {idx}")
        except Exception as e:
            self.record_info.append(f"[ERROR] REPLAY 처리 실패: {e}")
            logger_mp.error(f"on_replay_episode Exception: {e}")

    def on_deploy(self):
        """
        SET 버튼을 눌렀을 때:
          1) task_name, num_episodes, episode_len을 SHM에 기록
          2) 버튼 플래그 초기화 (start/reset/replay 모두 False)
          3) record_info 창에 “설정 완료” 메시지 출력
        """
        # 1) 입력값 읽어오기
        task_text = self.gr00t_task_name_le.text().strip()


        try:
            self.gr00t_task_shm.write_data(
                task_name    = task_text,
            )
            self.record_mode_shm.write_data(
                start        = np.bool_(False),
                reset        = np.bool_(False),
                replay       = np.bool_(False),
                done         = np.bool_(False),
                deploy       = np.bool_(True) 
            )
        except Exception as e:
            self.record_info.append(f"[ERROR] SHM Write 실패: {e}")
            logger_mp.error(f"SHM Write 실패 on_set_task: {e}")

    def read_from_shm(self):
        """
        QTimer마다 호출되어, TELEVISION shm에 들어 있는 데이터를 읽고
        ui에 한 줄(최신 값)로 덮어쓴다.
        """
        # (A) SharedMemoryManager는 내부적으로 lock/unlock을 처리하므로
        #     바로 read_data()를 호출해도 무방한 경우가 많습니다.
        #     만약 별도 lock/unlock이 필요하다면 SharedMemoryManager 문서를 참고하세요.

        # POSE INFO 기능 제거됨

        try:
            left_touch_data  = self.left_touch_shm.read_data()
            right_touch_data = self.right_touch_shm.read_data()
            
        except Exception:
            left_touch_data  = {}
            right_touch_data = {}

        # 내부 helper: 0–4095 범위를 0–255로 정규화 후 QColor 생성
        def value_to_color(val: int):
            if val < 0:
                val = 0
            if val > 4095:
                val = 4095
            intensity = int((val / 4095.0) * 255)
            r = 255
            g = 255 - intensity
            b = 255 - intensity
            return QtGui.QColor(r, g, b)

        # Left Touch 갱신
        transpose_fields = {"l_palm_touch", "r_palm_touch"}


        for field_name, matrix in left_touch_data.items():

            # # (1) 해당 필드면 transpose
            # if field_name in transpose_fields:
            #     matrix = matrix.T

            table = self.touch_tables.get(field_name)   # ✅ 이 방식으로 바꿔야 함
            if not table:
                continue

            rows, cols = matrix.shape
            # 테이블이 정확히 같은 rowCount/columnCount를 가지고 있어야 함
            table.setRowCount(rows)               # ✅ 여기 추가
            table.setColumnCount(cols)           # ✅ 여기 추가
            for i in range(rows):
                for j in range(cols):
                    val = int(matrix[i, j])
                    color = value_to_color(val)

                    item = table.item(i, j)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        table.setItem(i, j, item)
                    item.setBackground(QtGui.QBrush(color))
            table.viewport().update()            # ✅ 루프 끝나고 여기에 추가
            
        # Right Touch 갱신
        for field_name, matrix in right_touch_data.items():


            # # (1) 해당 필드면 transpose
            # if field_name in transpose_fields:
            #     matrix = matrix.T

            table = self.touch_tables.get(field_name)   # ✅ 이 방식으로 바꿔야 함

            if not table:
                continue

            rows, cols = matrix.shape
            table.setRowCount(rows)               # ✅ 여기 추가
            table.setColumnCount(cols)           # ✅ 여기 추가
            for i in range(rows):
                for j in range(cols):
                    val = int(matrix[i, j])
                    color = value_to_color(val)

                    item = table.item(i, j)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        table.setItem(i, j, item)
                    item.setBackground(QtGui.QBrush(color))
            table.viewport().update()            # ✅ 루프 끝나고 여기에 추가

        try:
            mode_data = self.record_mode_shm.read_data()
            done_flag = mode_data["done"].item()  # np.bool_ → Python bool

            if done_flag:
                # 1) UI에 "녹화 완료" 메시지 추가
                self.record_info.append("[INFO] 녹화 완료")

                # 2) 'done' 플래그를 False로 클리어            

                self.record_mode_shm.write_data(reset=np.bool_(False),
                                                start=np.bool_(False),
                                                replay=np.bool_(False),
                                                done=np.bool_(False))
        except Exception as e:
            # 만약 SHM 연결이 아직 안 됐거나 읽기/쓰기에 문제가 있으면 무시하고 넘어갑니다.
            pass


        try:
            freq_data = self.freq_shm.read_data()
            self.g1_freq.setPlainText(f"{freq_data['g1_freq']:.2f} Hz")
            self.hand_freq.setPlainText(f"{freq_data['hand_freq']:.2f} Hz")
            self.vr_freq.setPlainText(f"{freq_data['vr_freq']:.2f} Hz")
            self.camera_freq.setPlainText(f"{freq_data['camera_freq']:.2f} Hz")
            self.record_freq.setPlainText(f"{freq_data['record_freq']:.2f} Hz")
        except Exception as e:
            logger_mp.error(f"주파수 SHM 읽기 실패: {e}")
        
        # 1) 시간축 추가 (상대 시간)
        now = time.time() - self.start_time
        self.time_data.append(now)
        # 2) 각 주파수 버퍼에 추가
        self.g1_data.append( freq_data['g1_freq'] )
        self.hand_data.append( freq_data['hand_freq'] )
        self.vr_data.append(   freq_data['vr_freq'] )
        self.cam_data.append(  freq_data['camera_freq'] )
        self.rec_data.append(  freq_data['record_freq'] )

        # 3) Curve에 데이터 그리기 (그래프가 활성화된 경우에만)
        if self.curve_g1:
            self.curve_g1  .setData(self.time_data, self.g1_data)
        if self.curve_hand:
            self.curve_hand.setData(self.time_data, self.hand_data)
        if self.curve_vr:
            self.curve_vr  .setData(self.time_data, self.vr_data)
        if self.curve_cam:
            self.curve_cam .setData(self.time_data, self.cam_data)
        if self.curve_rec:
            self.curve_rec .setData(self.time_data, self.rec_data)


        try:
            mode_data = self.record_mode_shm.read_data()
            for key in ("start", "home", "reset", "replay", "done"):
                flag = bool(mode_data.get(key, False))
                label = getattr(self, f"{key}_label", None)
                if label:
                    col = "green" if flag else "red"
                    label.setStyleSheet(f"background-color: {col};")
        except Exception as e:
            logger_mp.error(f"Status label update failed: {e}")
        


        if self.shared_event['set_start'].is_set():
            self.start_button.setText("PAUSE")       # 버튼 글씨 변경
        else :
            self.start_button.setText("START")       # 버튼 글씨 변경

        try:
            record_episode_dict = self.record_episode_shm.read_data()
            progress = record_episode_dict.get("logging_progress", 0)
            # logging_progress 가 이미 0~100 정수라면 그대로 사용
            # 안전하게 범위 제한
            progress = max(0, min(int(progress), 100))
        except Exception:
            progress = 0

        if self.logging_progress_bar:
            # 메인-스레드에서 UI 업데이트 (Qt 신호/슬롯 필요 없음)
            self.logging_progress_bar.setValue(progress)

        self.joint_buf = {
            "Waist":     {"qpos": [], "action": []},
            "Left_arm":  {"qpos": [], "action": []},
            "Right_arm": {"qpos": [], "action": []},
            "Left_hand": {"qpos": [], "action": []},
            "Right_hand":{"qpos": [], "action": []},
        }

        # robot_data 읽기
        try:
            robot_obs = self.robot_obs_shm.read_data()
            robot_action = self.robot_action_shm.read_data()

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

        except Exception as e:
            logger_mp.error(f"ROBOT_OBS, ACTION shm 읽기 실패: {e}")
            return

        # 시간 추가
        self.joint_time.append(now)

        # 관절 인덱스 슬라이싱 정의
        q_slices = {
            "Waist": slice(0, 3),
            "Left_arm": slice(5, 12),
            "Right_arm": slice(12, 19),
        }
        h_slices = {
            "Left_hand": slice(0, 6),
            "Right_hand": slice(6, 12),
        }

        # 2) joint_history 에 qpos/action 값 쌓기
        for group, sl in q_slices.items():
            hist = self.joint_history[group]
            q = qpos[sl]
            a = action[sl]
            for i, (qi, ai) in enumerate(zip(q, a)):
                hist["qpos"][i].append(qi)
                hist["action"][i].append(ai)

        for group, sl in h_slices.items():
            hist = self.joint_history[group]
            hq = obs_hand[sl]
            ha = action_hand[sl]
            for i, (hqi, hai) in enumerate(zip(hq, ha)):
                hist["qpos"][i].append(hqi)
                hist["action"][i].append(hai)

        for group, sl in q_slices.items():
            subcols = self.joint_subcolors[group]
            hist = self.joint_history[group]
            # 그래프 위젯이 없으면 건너뜀
            if f"{group}_joint_graph" not in self.joint_plot_widgets:
                continue
            pw = self.joint_plot_widgets[f"{group}_joint_graph"]
            pw.clear()
            for i in range(sl.stop - sl.start):
                color = subcols[i]
                t_list = list(self.joint_time)[-len(hist["qpos"][i]):]
                # qpos: 실선
                pw.plot(
                    t_list,
                    list(hist["qpos"][i]),
                    pen=pg.mkPen(color=color, width=2),
                    name=f"{group}_qpos_{i}"
                )
                # action: 점선, 같은 색
                pw.plot(
                    t_list,
                    list(hist["action"][i]),
                    pen=pg.mkPen(color=color, style=QtCore.Qt.DashLine),
                    name=f"{group}_act_{i}"
                )

        # ── hand_qpos/hand_action 그리기 ──
        for group, sl in h_slices.items():
            subcols = self.joint_subcolors[group]
            hist = self.joint_history[group]
            # 그래프 위젯이 없으면 건너뜀
            if f"{group}_joint_graph" not in self.joint_plot_widgets:
                continue
            pw = self.joint_plot_widgets[f"{group}_joint_graph"]
            pw.clear()
            for i in range(sl.stop - sl.start):
                color = subcols[i]
                t_list = list(self.joint_time)[-len(hist["qpos"][i]):]
                pw.plot(
                    t_list,
                    list(hist["qpos"][i]),
                    pen=pg.mkPen(color=color, width=2),
                    name=f"{group}_hq_{i}"
                )
                pw.plot(
                    t_list,
                    list(hist["action"][i]),
                    pen=pg.mkPen(color=color, style=QtCore.Qt.DashLine),
                    name=f"{group}_ha_{i}"
                )


    def update_frame(self):
        """
        Shared Memory에서 최신 컬러 이미지를 읽어와서 video_label과 zed_video_label에 표시
        """
        try:
            data_dict = self.camera_shm.read_data()
        except Exception as e:
            # 읽기 실패 시 간단히 리턴 (로그를 남겨도 좋음)
            return

        # 카메라 이미지 가져오기 (선택된 뷰에 따라)
        if self.zed_current_view == ZED_VIEW_LEFT:
            zed_img = data_dict.get("camera_left", None)
            if zed_img is None:
                # 이전 프레임이 있다면 재사용
                if hasattr(self, 'prev_left_img') and self.prev_left_img is not None:
                    zed_img = self.prev_left_img.copy()
            else:
                # 현재 프레임을 저장해두기
                self.prev_left_img = zed_img.copy()
        else:  # ZED_VIEW_REALSENSE
            zed_img = data_dict.get("realsense", None)
            if zed_img is None:
                # 이전 프레임이 있다면 재사용
                if hasattr(self, 'prev_realsense_img') and self.prev_realsense_img is not None:
                    zed_img = self.prev_realsense_img.copy()
            else:
                # 현재 프레임을 저장해두기
                self.prev_realsense_img = zed_img.copy()

        # 작업 공간 마스크 적용 및 테두리/마커 표시 (APPLY_MASK_IN_GUI가 True이고 ZED Left일 때만)
        if zed_img is not None and APPLY_MASK_IN_GUI and self.zed_current_view == ZED_VIEW_LEFT:
                try:
                    mask_data = self.workspace_mask_shm.read_data()
                    mask_left_flat = mask_data.get("mask_left_flat", None)
                    mask_right_flat = mask_data.get("mask_right_flat", None)


                    # ZED Left일 때만 마스크 contour와 marker corners 선택
                    mask_contour_flat = mask_data.get("mask_contour_left", None)
                    marker_corners_flat = mask_data.get("marker_corners_left", None)

                    # 실제 마스크 적용 (작업 공간만 표시)
                    mask_flat = mask_left_flat
                    if mask_flat is not None:
                        mask = mask_flat.reshape(480, 640).astype(np.uint8)
                        if mask.shape[:2] == zed_img.shape[:2]:
                            zed_img = cv2.bitwise_and(zed_img, zed_img, mask=mask)

                    # 마스크 테두리와 마커 꼭지점 오버레이 표시
                    if mask_contour_flat is not None and marker_corners_flat is not None:
                        # 데이터 형태 복원
                        mask_contour = mask_contour_flat.reshape(4, 2).astype(np.int32)
                        marker_corners = marker_corners_flat.reshape(4, 2).astype(np.int32)

                        # 마스크 테두리 그리기 (초록색 선)
                        contour_points = mask_contour.reshape(-1, 1, 2)
                        cv2.polylines(zed_img, [contour_points], True, (0, 255, 0), 2)

                        # 마커 꼭지점 그리기 (빨강색 점)
                        for corner in marker_corners:
                            if corner[0] > 0 and corner[1] > 0:  # 유효한 좌표만
                                cv2.circle(zed_img, tuple(corner), 6, (0, 0, 255), -1)

                        # print(f"[GUI] 마스크 적용 및 테두리/마커 표시됨")
                    else:
                        print(f"[GUI] 마스크 테두리/마커 데이터 없음")
                except Exception as e:
                    print(f"[GUI] 마스크 적용/표시 실패: {e}")
                    pass
        if zed_img is not None:
            # NumPy BGR 배열을 QImage로 변환
            h, w, ch = zed_img.shape  # height, width, channels(=3)
            bytes_per_line = ch * w
            qimg = QtGui.QImage(zed_img.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888)

            # QPixmap으로 변환
            pixmap = QtGui.QPixmap.fromImage(qimg)

            # label 크기에 맞춰 스케일링 (여백 없이 전체 채우기)
            scaled_pixmap = pixmap.scaled(
                self.zed_video_label.width(),
                self.zed_video_label.height(),
                QtCore.Qt.KeepAspectRatio,  # 비율 유지
                QtCore.Qt.SmoothTransformation  # 고품질 스케일링
            )
            self.zed_video_label.setPixmap(scaled_pixmap)

    def on_toggle_mask_control(self):
        """
        마스크 제어 토글 버튼 핸들러
        """
        self.mask_control_enabled = not self.mask_control_enabled

        try:
            # 공유 메모리에 상태 업데이트
            self.mask_control_shm.write_data(
                mask_control_enabled=np.bool_(self.mask_control_enabled),
                generate_new_mask=np.bool_(False)  # 토글 시에는 새 마스크 생성하지 않음
            )

            # 버튼 텍스트와 스타일 업데이트
            if self.mask_control_enabled:
                self.mask_control_btn.setText("마스크 제어: ON")
                self.mask_control_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4caf50;
                        color: white;
                        border: 2px solid #388e3c;
                        border-radius: 5px;
                        padding: 8px;
                        font-size: 12px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #388e3c;
                    }
                    QPushButton:pressed {
                        background-color: #2e7d32;
                    }
                """)
                self.record_info.append("[INFO] 마스크 제어 활성화됨")
            else:
                self.mask_control_btn.setText("마스크 제어: OFF")
                self.mask_control_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #ff6b6b;
                        color: white;
                        border: 2px solid #ff5252;
                        border-radius: 5px;
                        padding: 8px;
                        font-size: 12px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #ff5252;
                    }
                    QPushButton:pressed {
                        background-color: #ff3838;
                    }
                """)
                self.record_info.append("[INFO] 마스크 제어 비활성화됨 - 실시간 마스크 생성으로 복귀")

            logger_mp.info(f"[GUI] 마스크 제어 상태 변경: {self.mask_control_enabled}")

        except Exception as e:
            self.record_info.append(f"[ERROR] 마스크 제어 상태 변경 실패: {e}")
            logger_mp.error(f"[GUI] 마스크 제어 상태 변경 실패: {e}")

    def on_generate_new_mask(self):
        """
        새 마스크 생성 버튼 핸들러
        """
        try:
            # 공유 메모리에 새 마스크 생성 요청
            self.mask_control_shm.write_data(
                mask_control_enabled=np.bool_(self.mask_control_enabled),
                generate_new_mask=np.bool_(True)
            )

            self.record_info.append("[INFO] 새 마스크 생성 요청 전송됨")
            logger_mp.info("[GUI] 새 마스크 생성 요청")

            # 잠시 후 generate_new_mask 플래그를 False로 리셋 (worker가 처리했음을 표시하기 위해)
            QtCore.QTimer.singleShot(100, self._reset_generate_mask_flag)

        except Exception as e:
            self.record_info.append(f"[ERROR] 새 마스크 생성 요청 실패: {e}")
            logger_mp.error(f"[GUI] 새 마스크 생성 요청 실패: {e}")

    def _reset_generate_mask_flag(self):
        """
        generate_new_mask 플래그를 False로 리셋
        """
        try:
            self.mask_control_shm.write_data(
                mask_control_enabled=np.bool_(self.mask_control_enabled),
                generate_new_mask=np.bool_(False)
            )
        except Exception as e:
            logger_mp.error(f"[GUI] generate_new_mask 플래그 리셋 실패: {e}")

    def closeEvent(self, event):
        """
        윈도우가 닫힐 때 SharedMemoryManager 닫기
        """
        try:
            self.camera_shm.worker_close()
            self.workspace_mask_shm.worker_close()
            # television_shm.worker_close() 제거됨
            self.record_task_shm.worker_close()
            self.record_episode_shm.worker_close()
            self.record_mode_shm.worker_close()
            self.right_touch_shm.worker_close()
            self.left_touch_shm .worker_close()
            self.freq_shm .worker_close()
            self.gr00t_task_shm.worker_close()
            self.mask_control_shm.worker_close()
            # self.webView 관련 cleanup 제거됨 (meshcat 기능 제거)
        except:
            pass
        event.accept()

    def load_with_port(self):
        # Meshcat 시각화 기능 제거됨
        pass

def run_ui(shared_event, shm_name, shared_lock):

    # Qt WebEngine 초기화를 위한 속성 설정 (OpenGL 컨텍스트 공유)
    # Meshcat 제거로 필요 없지만, 호환성을 위해 유지
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

    # Qt 애플리케이션 생성
    app = QtWidgets.QApplication(sys.argv)

    # TeleopUI 인스턴스 생성 및 표시
    window = TeleopUI(shared_event,shm_name, shared_lock)
    window.show()

    # stop_evt_ui가 set되면 애플리케이션 종료
    def watch_stop():
        shared_event['shutdown'].wait()
        app.quit()

    Thread(target=watch_stop, daemon=True).start()
    app.exec_()