import os
import time
import numpy as np

from utils.state import State, EventsSnapshot, next_state
from utils.rate import Rate

from multiprocessing import shared_memory, Array, Lock
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import TELEVISION, WORKER_FREQ, RECORD_MODE_LAYOUT, RECORD_EPISODE_LAYOUT, RECORD_TASK_LAYOUT, RECORD_TASK_LAYOUT, ROBOT_ACTION, ROBOT_OBS, KISTAR_HAND_ACTION, CURRENT_MODE_LAYOUT, MODE_MAPPING_INV
import traceback
import pandas as pd

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# 디버깅 시각화를 위한 추가 임포트
import matplotlib
# 헤드리스 환경이 아니라면 Qt5Agg 백엔드 사용
try:
    import tkinter
    matplotlib.use('TkAgg')  # Tkinter 백엔드 사용 (더 안정적)
except ImportError:
    matplotlib.use('Agg')  # GUI 없는 환경에서는 Agg 사용

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
from queue import Queue
import signal

# 디버깅 시각화 활성화 플래그
DEBUG_VIZ_ENABLED = False  # True로 설정하면 3D 시각화 활성화

# KISTAR Hand IK 함수들 import (utils/geometry_solver_np에서)
from utils.geometry_solver_np.hand_fk_solver_v2_np import Transform_Wrist2Hand_V2_np
from utils.geometry_solver_np.hand_ik_solver_v2_np import IK_Finger_V2_np, IK_Thumb_V2_np
from utils.geometry_solver_np.hand_ik_solver_np import transform_goalpoint_wrist2world_np

class HandDebugVisualizer:
    """오른손 디버깅용 실시간 3D 시각화 클래스"""
    
    def __init__(self, update_interval=0.1):
        self.fig = None
        self.ax = None
        self.data_queue = Queue()
        self.update_interval = update_interval
        self.running = False
        self.viz_thread = None
        self.setup_complete = False
        
        # 손가락 이름과 색상 매핑
        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        self.finger_colors = ['red', 'blue', 'green', 'orange', 'purple']
        
    def start_visualization(self):
        """시각화 쓰레드 시작"""
        if not self.running:
            self.running = True
            self.viz_thread = threading.Thread(target=self._visualization_loop, daemon=True)
            self.viz_thread.start()
            logger_mp.info("[DEBUG VIZ] 실시간 3D 시각화 시작")
    
    def stop_visualization(self):
        """시각화 쓰레드 종료"""
        self.running = False
        if self.viz_thread and self.viz_thread.is_alive():
            self.viz_thread.join(timeout=2.0)
        if self.fig:
            try:
                plt.close(self.fig)
            except:
                pass
        logger_mp.info("[DEBUG VIZ] 3D 시각화 종료")
    
    def update_hand_data(self, right_hand_tips, right_distal, right_proximal, right_wrist_pos):
        """
        손 데이터 업데이트
        Args:
            right_hand_tips: (5, 3) - 손가락 끝점 좌표
            right_distal: (5, 3) - distal joint 좌표  
            right_proximal: (5, 3) - proximal joint 좌표
            right_wrist_pos: (3,) - 손목 위치
        """
        if self.running and self.data_queue.qsize() < 10:  # 큐 크기 제한
            try:
                self.data_queue.put({
                    'tips': right_hand_tips.copy(),
                    'distal': right_distal.copy(), 
                    'proximal': right_proximal.copy(),
                    'wrist': right_wrist_pos.copy(),
                    'timestamp': time.time()
                }, block=False)
            except:
                pass  # 큐가 가득 차면 무시
    
    def _setup_matplotlib(self):
        """matplotlib 설정 (서브스레드에서 안전하게)"""
        try:
            # 신호 처리 비활성화
            import matplotlib
            matplotlib.rcParams['toolbar'] = 'None'  # 툴바 제거
            
            # Figure 생성
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # 인터랙티브 모드 설정
            plt.ion()
            self.fig.show()
            
            self.setup_complete = True
            logger_mp.info("[DEBUG VIZ] Matplotlib 설정 완료")
            return True
            
        except Exception as e:
            logger_mp.error(f"[DEBUG VIZ] Matplotlib 설정 실패: {e}")
            return False
    
    def _visualization_loop(self):
        """메인 시각화 루프"""
        try:
            # matplotlib 설정
            if not self._setup_matplotlib():
                return
                
            last_update = time.time()
            
            while self.running:
                try:
                    current_time = time.time()
                    
                    # 업데이트 간격 체크
                    if current_time - last_update >= self.update_interval:
                        # 최신 데이터 가져오기
                        latest_data = None
                        while not self.data_queue.empty():
                            try:
                                latest_data = self.data_queue.get_nowait()
                            except:
                                break
                        
                        if latest_data is not None:
                            self._plot_hand_data(latest_data)
                            last_update = current_time
                    
                    time.sleep(0.01)  # CPU 부하 줄이기
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger_mp.error(f"[DEBUG VIZ] Error in visualization loop: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger_mp.error(f"[DEBUG VIZ] Fatal error in visualization: {e}")
        finally:
            try:
                plt.ioff()
                if self.fig:
                    plt.close(self.fig)
            except:
                pass
    
    def _plot_hand_data(self, data):
        """손 데이터를 3D로 플롯"""
        try:
            if not self.setup_complete or not self.fig:
                return
                
            self.ax.clear()
            
            tips = data['tips']
            distal = data['distal'] 
            proximal = data['proximal']
            wrist_pos = data['wrist']
            
            # 데이터 디버그 출력 (처음 몇 번만)
            if hasattr(self, 'debug_count'):
                self.debug_count += 1
            else:
                self.debug_count = 1
                
            if self.debug_count <= 3:  # 처음 3번만 출력
                logger_mp.info(f"[DEBUG VIZ #{self.debug_count}] Coordinate analysis:")
                logger_mp.info(f"  - wrist_pos: {wrist_pos}")
                logger_mp.info(f"  - tips[0] (Thumb): {tips[0]} (wrist frame)")
                logger_mp.info(f"  - Converting to world coordinates...")
            
            # 손목 위치 플롯 (월드 좌표계)
            self.ax.scatter(wrist_pos[0], wrist_pos[1], wrist_pos[2], 
                           c='black', s=150, marker='s', label='Wrist', alpha=0.8)
            
            # 손가락 좌표를 손목 좌표계에서 월드 좌표계로 변환
            # right_wrist_mat에서 회전 행렬 추출해야 하지만, 여기서는 간단히 위치만 변환
            # 실제로는 right_wrist_mat의 회전도 적용해야 하지만, 우선 위치 변환만 적용
            finger_visible_count = 0
            
            for i in range(5):
                color = self.finger_colors[i]
                name = self.finger_names[i]
                
                # 손목 좌표계의 손가락 좌표를 월드 좌표계로 변환
                # tips[i], distal[i], proximal[i]는 손목 좌표계 기준
                # 월드 좌표 = 손목_월드위치 + 손가락_손목좌표계위치
                tip_world = wrist_pos + tips[i]
                distal_world = wrist_pos + distal[i]
                proximal_world = wrist_pos + proximal[i]
                
                # 데이터가 유효한지 확인
                tip_valid = np.any(np.abs(tips[i]) > 1e-6)
                distal_valid = np.any(np.abs(distal[i]) > 1e-6)
                proximal_valid = np.any(np.abs(proximal[i]) > 1e-6)
                
                if tip_valid or distal_valid or proximal_valid:
                    finger_visible_count += 1
                
                # Proximal 관절 (월드 좌표)
                if proximal_valid:
                    self.ax.scatter(proximal_world[0], proximal_world[1], proximal_world[2], 
                                   c=color, s=80, marker='o', alpha=0.8)
                
                # Distal 관절 (월드 좌표)
                if distal_valid:
                    self.ax.scatter(distal_world[0], distal_world[1], distal_world[2], 
                                   c=color, s=100, marker='^', alpha=0.9)
                
                # Tip (월드 좌표)
                if tip_valid:
                    self.ax.scatter(tip_world[0], tip_world[1], tip_world[2], 
                                   c=color, s=120, marker='*', label=f'{name} Tip')
                
                # 관절들을 선으로 연결 (손목부터 시작)
                valid_points = [wrist_pos]  # 손목부터 시작
                
                if proximal_valid:
                    valid_points.append(proximal_world)
                if distal_valid:
                    valid_points.append(distal_world)
                if tip_valid:
                    valid_points.append(tip_world)
                
                if len(valid_points) >= 2:
                    points = np.array(valid_points)
                    self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                                c=color, linewidth=3, alpha=0.7)
                
                # 첫 번째 프레임에서 월드 좌표 확인
                if self.debug_count <= 3 and i == 0:  # 엄지손가락만
                    logger_mp.info(f"  - Thumb world coords:")
                    logger_mp.info(f"    Proximal: {proximal_world}")
                    logger_mp.info(f"    Distal: {distal_world}")
                    logger_mp.info(f"    Tip: {tip_world}")
            
            # 디버그 정보를 제목에 추가
            debug_info = f"Visible fingers: {finger_visible_count}/5"
            
            # 축 설정
            self.ax.set_xlabel('X (m)', fontsize=12)  # 월드 좌표계 미터 단위
            self.ax.set_ylabel('Y (m)', fontsize=12) 
            self.ax.set_zlabel('Z (m)', fontsize=12)
            
            # 범례
            handles, labels = self.ax.get_legend_handles_labels()
            if len(handles) > 0:
                self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            self.ax.set_title(f'Right Hand Real-time Debug (World Coordinates)\n{debug_info} | Time: {data["timestamp"]:.2f}', fontsize=14)
            
            # 축 범위 계산 (손목 포함, 전체 손 기준)
            valid_points_all = [wrist_pos]  # 손목 포함
            
            for i in range(5):
                if np.any(np.abs(tips[i]) > 1e-6):
                    valid_points_all.append(wrist_pos + tips[i])
                if np.any(np.abs(distal[i]) > 1e-6):
                    valid_points_all.append(wrist_pos + distal[i])
                if np.any(np.abs(proximal[i]) > 1e-6):
                    valid_points_all.append(wrist_pos + proximal[i])
            
            if len(valid_points_all) > 1:
                all_points = np.vstack(valid_points_all)
                center = np.mean(all_points, axis=0)
                distances = np.linalg.norm(all_points - center, axis=1)
                max_range = max(np.max(distances) * 1.3, 0.1)  # 최소 10cm 범위
                
                self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
                self.ax.set_ylim(center[1] - max_range, center[1] + max_range) 
                self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
            else:
                # 손목만 있는 경우
                range_val = 0.2  # 20cm 범위
                self.ax.set_xlim(wrist_pos[0] - range_val, wrist_pos[0] + range_val)
                self.ax.set_ylim(wrist_pos[1] - range_val, wrist_pos[1] + range_val) 
                self.ax.set_zlim(wrist_pos[2] - range_val, wrist_pos[2] + range_val)
            
            # 그리드 추가
            self.ax.grid(True, alpha=0.3)
            
            # 시점 조정
            self.ax.view_init(elev=20, azim=45)
            
            # 화면 업데이트
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logger_mp.error(f"[DEBUG VIZ] Error plotting: {e}")
            import traceback
            logger_mp.error(f"[DEBUG VIZ] Traceback: {traceback.format_exc()}")


def debug_plot_hand_once(right_hand_tips, right_distal, right_proximal, right_wrist_pos, 
                        save_path=None, show=False):
    """
    한 번만 손 데이터를 3D로 플롯하는 함수 (간단한 디버깅용)
    """
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        finger_colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # 손목 위치 플롯
        ax.scatter(right_wrist_pos[0], right_wrist_pos[1], right_wrist_pos[2], 
                   c='black', s=100, marker='s', label='Wrist')
        
        # 각 손가락별로 플롯
        for i in range(5):
            color = finger_colors[i]
            name = finger_names[i]
            
            # 각 관절 위치 플롯
            ax.scatter(right_proximal[i, 0], right_proximal[i, 1], right_proximal[i, 2], 
                       c=color, s=60, marker='o', alpha=0.7)
            ax.scatter(right_distal[i, 0], right_distal[i, 1], right_distal[i, 2], 
                       c=color, s=80, marker='^', alpha=0.8)
            ax.scatter(right_hand_tips[i, 0], right_hand_tips[i, 1], right_hand_tips[i, 2], 
                       c=color, s=100, marker='*', label=f'{name} Tip')
            
            # 관절들을 선으로 연결
            points = np.array([right_wrist_pos, right_proximal[i], right_distal[i], right_hand_tips[i]])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                    c=color, linewidth=2, alpha=0.6)
        
        # 축 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') 
        ax.legend()
        ax.set_title('Right Hand Debug Visualization')
        
        # 축 범위 자동 조정
        all_points = np.vstack([right_hand_tips, right_distal, right_proximal, right_wrist_pos.reshape(1, -1)])
        center = np.mean(all_points, axis=0)
        max_range = np.max(np.linalg.norm(all_points - center, axis=1)) * 1.2
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger_mp.info(f"[DEBUG VIZ] 이미지 저장됨: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger_mp.error(f"[DEBUG VIZ] Error in debug plot: {e}")


def quat_to_rotmat(quat):
    """쿼터니언을 회전 행렬로 변환"""
    w, x, y, z = quat
    R = np.empty((3, 3), dtype=np.float32)
    R[0, 0] = 1 - 2 * (y**2 + z**2)
    R[0, 1] = 2 * (x * y - z * w)
    R[0, 2] = 2 * (x * z + y * w)
    R[1, 0] = 2 * (x * y + z * w)
    R[1, 1] = 1 - 2 * (x**2 + z**2)
    R[1, 2] = 2 * (y * z - x * w)
    R[2, 0] = 2 * (x * z - y * w)
    R[2, 1] = 2 * (y * z + x * w)
    R[2, 2] = 1 - 2 * (x**2 + y**2)
    return R

def _events_snapshot(shared_event, hand_initialized: bool) -> EventsSnapshot:
    return EventsSnapshot(
        shutdown = shared_event['shutdown'].is_set(),
        emergency = shared_event['emergency'].is_set(),
        g1_ready = True,  # kistar_only 모드에서는 G1 없이도 진행
        hand_ready = hand_initialized,
        start = shared_event['set_start'].is_set(),
        select_pressed = False
    )

def get_current_mode(current_mode_shm):
    """공유 메모리에서 현재 모드 정보를 읽어옴 (정수 → 문자열 변환)"""
    try:
        mode_data = current_mode_shm.read_data()
        mode_int = int(mode_data["mode"].item())
        current_mode = MODE_MAPPING_INV.get(mode_int, 'teleop')
        return current_mode
    except Exception as e:
        logger_mp.warning(f"[VR Hand IK] 모드 정보 읽기 실패: {e}, 기본값 'teleop' 사용")
        return "teleop"

def find_perpendicular_vectors(tip_points, distal_points, proximal_points):
    """
    (벡터화 사용) 모든 손가락에 대해 수직 벡터를 효율적으로 계산합니다.
    """

    vec_tip_distal = distal_points - tip_points
    vec_tip_proximal = proximal_points - tip_points

    normal_vectors = np.cross(vec_tip_distal, vec_tip_proximal)

    target_vectors = np.cross(normal_vectors, vec_tip_distal)

    norms = np.linalg.norm(target_vectors, axis=1)

    zero_norms_mask = norms < 1e-9
    norms[zero_norms_mask] = 1.0

    unit_vectors = target_vectors / norms[:, np.newaxis]

    #norm이 0이었던 위치의 벡터들을 [0, 0, 0]으로 설정
    unit_vectors[zero_norms_mask] = [0.0, 0.0, 0.0]
    
    return unit_vectors

def solve_kistar_hand_ik(right_hand_fingers, right_wrist_mat, norm_vector):
    """
    VR 데이터를 사용해서 KISTAR Hand의 16개 관절 각도를 계산
    
    Args:
        right_hand_fingers: (5, 3) - 오른손 5개 손가락 끝점 좌표
        right_wrist_mat: (4, 4) - 오른손 손목 변환 행렬
    
    Returns:
        np.ndarray: (16,) - KISTAR Hand 관절 각도
    """
    try:
        # 손목 위치와 회전 추출
        wrist_pos = right_wrist_mat[:3, 3]  # (3,)
        wrist_rot = right_wrist_mat[:3, :3]  # (3, 3)
        
        # 회전 행렬을 쿼터니언으로 변환 (w, x, y, z)
        from utils.geometry_solver_np.geometry_utils_np import rotmat_to_quat, quat_mul
        wrist_quat = rotmat_to_quat(wrist_rot)  # (4,) [w, x, y, z]        
        

        # 손목에서 손으로 변환 (회전 적용된 쿼터니언 사용)
        hand_quat, hand_pos = Transform_Wrist2Hand_V2_np(wrist_quat, wrist_pos)
        
        
        # 손가락 끝점들을 손목 좌표계로 변환
        finger_points = []
        for i in range(5):  # 5개 손가락
            finger_point = right_hand_fingers[i]  # (3,)
            # 3개 요소만 사용 (x, y, z)
            if len(finger_point) > 3:
                finger_point = finger_point[:3]
            
            # 손목 좌표계로 변환
            finger_point_world = transform_goalpoint_wrist2world_np(
                finger_point, wrist_quat, wrist_pos, 1.0
            )
            finger_points.append(finger_point_world)
        
        # 각 손가락별로 IK 계산
        joint_angles = np.zeros(16)
        
        # 엄지 (4개 관절)
        thumb_dof = IK_Thumb_V2_np(finger_points[0], hand_quat, hand_pos, norm_vector[0, :])
        joint_angles[0:4] = thumb_dof
        
        # 검지 (4개 관절)
        index_dof = IK_Finger_V2_np("Index", finger_points[1], hand_quat, hand_pos)
        joint_angles[4:8] = index_dof
        
        # 중지 (4개 관절)
        middle_dof = IK_Finger_V2_np("Middle", finger_points[2], hand_quat, hand_pos)
        joint_angles[8:12] = middle_dof
        
        # 약지 (4개 관절)
        ring_dof = IK_Finger_V2_np("Ring", finger_points[3], hand_quat, hand_pos)
        joint_angles[12:16] = ring_dof
        
        # print('엄지', np.degrees(thumb_dof))    
        # print('검지',np.degrees(index_dof))
        # print('중지',np.degrees(middle_dof))
        # print('약지',np.degrees(ring_dof))

        return joint_angles
        
    except Exception as e:
        logger_mp.exception("[KISTAR IK] Error solving IK: %s", e)
        # 에러 시 기본값 반환
        return np.zeros(16)

def worker_vr_hand_ik(shared_event, shm_name, shared_lock):
    
    #Set SharedMemory
    television_shm = SharedMemoryManager(TELEVISION, shared_lock["television_lock"], shm_name["television_shm"])
    freq_shm = SharedMemoryManager(WORKER_FREQ, shared_lock["freq_lock"], shm_name["freq_shm"])
    record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, shared_lock["record_lock"], shm_name["record_mode_shm"])
    record_episode_shm = SharedMemoryManager(RECORD_EPISODE_LAYOUT, shared_lock["record_lock"], shm_name["record_episode_shm"])
    record_task_shm = SharedMemoryManager(RECORD_TASK_LAYOUT, shared_lock["record_lock"], shm_name["record_task_shm"])
    current_mode_shm = SharedMemoryManager(CURRENT_MODE_LAYOUT, shared_lock["record_lock"], shm_name["current_mode_shm"])
    robot_action_shm = SharedMemoryManager(ROBOT_ACTION, shared_lock["robot_action_lock"], shm_name["robot_action_shm"])
    robot_obs_shm = SharedMemoryManager(ROBOT_OBS, shared_lock["robot_obs_lock"], shm_name["robot_obs_shm"])
    kistar_hand_action_shm = SharedMemoryManager(KISTAR_HAND_ACTION, shared_lock["kistar_hand_action_lock"], shm_name["kistar_hand_action_shm"])   

    rate = Rate(50.0)

    # G1/Hand 컨트롤러를 아직 생성하지 않은 상태를 추적하는 플래그
    hand_initialized = False

    # VR 데이터 배열들
    left_hand_array = None
    right_hand_array = None
    dual_hand_data_lock = None
    dual_hand_state_array = None
    dual_hand_action_array = None

    # 디버깅 시각화 객체
    debug_visualizer = None

    replay_demo_init = False
    replay_actions = None   # action 배열 (N×31)
    replay_hand_data   = None   # 핸드 관련 action 배열 (N×12)

    replay_length = 0       # 전체 프레임 수
    replay_frame_idx = 0    # 현재 재생 중인 프레임 인덱스

    REPLAY_HZ = 20.0
    _replay_period = 1.0 / REPLAY_HZ
    _replay_next_t = None           # 다음 프레임을 전진시킬 절대시각
    _replay_start_wall = None       # (선택) 타임스탬프 재생용 시작 벽시각
    _replay_rel_ts = None           # (선택) 데이터셋 상대 타임스탬프 배열
    _last_written_idx = -1          # (선택) 진행률/로그 중복 방지

    state = State.WAIT_CONNECT
    logger_mp.info("[Hand_Ctrl] FSM start: WAIT_CONNECT")

    try:
        while True:
            # 상태 전이
            state_next = next_state(state, _events_snapshot(shared_event, hand_initialized))
            if state_next != state:
                logger_mp.info(f"[Hand_Ctrl] {state.name} -> {state_next.name}")
            state = state_next

            if state is State.EXIT:
                break

            if state is State.WAIT_CONNECT:
                # set_hand가 올라오면 IK 초기화 1회 시도 (teleop 모드처럼 수동 초기화)
                if shared_event['set_hand'].is_set() and not hand_initialized:
                    try:
                        # VR 데이터 처리를 위한 배열 생성
                        left_hand_array = Array('d', 5 * 3, lock=True)  # 각 손가락 tip의 (x, y, z) 위치 값
                        right_hand_array = Array('d', 5 * 3, lock=True)  # 각 손가락 tip의 (x, y, z) 위치 값

                        dual_hand_data_lock = Lock()
                        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
                        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
    
                        hand_initialized = True
                        # teleop 모드와 동일: set_hand는 UI 버튼으로만 설정되도록 함 (자동 설정 제거)
                        logger_mp.info("[Hand_Ctrl] VR data arrays initialized.")
                        
                        # 디버깅 시각화 초기화 (실시간 모드)
                        if DEBUG_VIZ_ENABLED and not debug_visualizer:
                            debug_visualizer = HandDebugVisualizer(update_interval=0.1)  # 10Hz 업데이트
                            debug_visualizer.start_visualization()
                            logger_mp.info("[Hand_Ctrl] Real-time debug visualization initialized.")
                            
                    except Exception as e:
                        logger_mp.exception("[Hand_Ctrl] Array init failed: %s", e)
                        shared_event['emergency'].set()
                time.sleep(0.01)

            elif state is State.WAIT_START:
                # START 대기
                time.sleep(0.01)

            elif state is State.RUN:

                mode = record_mode_shm.read_data()
                home, replay, deploy = mode["home"], mode["replay"], mode["deploy"]

                freq_shm.write_data(hand_freq=rate.tick_hz())

                # VR 데이터 읽기 및 KISTAR Hand IK 계산
                try:
                    data = television_shm.read_data()
    
                    right_hand       = data["right_hand"]
                    right_wrist_mat  = data["right_wrist_mat"]  # 손목 pose 추가
                    # 추가: 오른손 distal/proximal 포인트 (각 5x3)
                    try:
                        right_distal     = data["right_distal"]
                        right_proximal   = data["right_proximal"]
                    except KeyError:
                        right_distal     = np.zeros((5, 3), dtype=np.float64)
                        right_proximal   = np.zeros((5, 3), dtype=np.float64)
                    
                    #print(right_distal, right_distal.shape)
                    norm_vector = find_perpendicular_vectors(right_hand, right_distal, right_proximal)

                    # 디버깅 시각화 업데이트
                    if DEBUG_VIZ_ENABLED and debug_visualizer:
                        wrist_pos = right_wrist_mat[:3, 3]
                        debug_visualizer.update_hand_data(right_hand, right_distal, right_proximal, wrist_pos)
                    
                    # 일회성 디버깅 (필요시 주석 해제)
                    # if DEBUG_VIZ_ENABLED and time.time() % 10 < 0.1:  # 10초마다 한 번
                    #     wrist_pos = right_wrist_mat[:3, 3]
                    #     debug_plot_hand_once(right_hand, right_distal, right_proximal, wrist_pos, 
                    #                          save_path=f"debug_hand_{int(time.time())}.png", show=False)
                    
                    right_hand_array[:] = right_hand.flatten()

                    # KISTAR Hand IK 계산 (오른손만)
                    kistar_joint_angles = solve_kistar_hand_ik(right_hand, right_wrist_mat, norm_vector)
                    
                    # 현재 모드 확인 (Gr00t 배포 모드에서는 VR IK 결과를 덮어쓰지 않음)
                    current_mode = get_current_mode(current_mode_shm)
                            
                    if current_mode in ['teleop','kistar_teleop','kistar_inspire_teleop']:
                        # KISTAR Hand 관절 각도를 shared memory에 저장
                        kistar_hand_action_shm.write_data(
                            hand_action=kistar_joint_angles
                        )

                except Exception as e:
                    logger_mp.exception("[VRHandIK] Error reading VR data: %s", e)

                if shared_event['set_start'].is_set() and not replay and not deploy:
                    # 현재 모드 확인
                    current_mode = get_current_mode(current_mode_shm)

                    if current_mode in ['kistar_teleop', 'kistar_inspire_teleop']:
                        # KISTAR 모드: kistar_hand_joints_shm의 데이터를 사용하므로 dual_hand는 0으로 설정
                        with dual_hand_data_lock:
                            dual_hand_state_array[:] = np.zeros(12)
                            dual_hand_action_array[:] = np.zeros(12)

                        # [BUG FIX] 이 워커는 robot_obs_shm을 오염시키면 안 됨. g1_ctrl이 쓰는 값을 덮어쓰게 됨.
                        # robot_obs_shm.write_data(
                        #     obs_hand=dual_hand_state_array,
                        # )
                        # [BUG FIX] 이 워커는 robot_action_shm을 오염시키면 안 됨. g1_ik가 쓰는 값을 덮어쓰게 됨.
                        # robot_action_shm.write_data(
                        #     action_hand=dual_hand_action_array,
                        # )
                        logger_mp.debug("[VR] KISTAR mode - using kistar_hand_action_shm for control")
                    else:
                        # 기존 teleop 모드: dual_hand_action_array 사용
                        # [BUG FIX] 이 워커는 robot_obs_shm을 오염시키면 안 됨.
                        # robot_obs_shm.write_data(
                        #     obs_hand=dual_hand_state_array,
                        # )
                        # [BUG FIX] 이 워커는 robot_action_shm을 오염시키면 안 됨.
                        # robot_action_shm.write_data(
                        #     action_hand=dual_hand_action_array,
                        # )
                        pass # Inspire Hand 모드일 경우 여기에 로직 추가 필요

                if replay and not replay_demo_init :
                    replay_idx = int(record_episode_shm.read_data()["replay_idx"].item())
                    task_name = record_task_shm.read_data()["task_name"].item().strip()

                    # 현재 모드 확인
                    current_mode = get_current_mode(current_mode_shm)

                    # 2) chunk 계산 (예: CHUNK_SIZE가 1000이라면)
                    CHUNK_SIZE = 1000
                    chunk_id   = replay_idx // CHUNK_SIZE

                    parquet_path   = os.path.join(
                        "record", task_name,
                        "data", f"chunk-{chunk_id:03d}",
                        f"episode_{replay_idx:06d}.parquet"
                    )
                    logger_mp.info(f"[REPLAY INIT] Loading {parquet_path}")

                    # 4) DataFrame 로드
                    df = pd.read_parquet(parquet_path, engine="pyarrow")
                    # 5) NumPy 배열로 변환
                    #    - observation/state도 필요하면 같은 식으로 df["observation.state"]
                    replay_actions   = np.stack(df["action"].to_numpy()).astype(np.float64)
                    replay_length    = replay_actions.shape[0]
                    # 모드별로 핸드 데이터 추출
                    if current_mode in ['kistar_teleop', 'kistar_inspire_teleop']:
                        # KISTAR 모드: 뒤 16개 추출 (19:35)
                        replay_hand_data = replay_actions[:, 19:]   # (N,16) - KISTAR hand 16개 자유도
                        logger_mp.info(f"[REPLAY INIT] KISTAR mode - hand_data shape = {replay_hand_data.shape}")
                    else:
                        # 기존 teleop 모드: 뒤 12개 추출 (19:31)
                        replay_hand_data = replay_actions[:, 19:31]  # (N,12) - 기존 양손 6+6개 자유도
                        logger_mp.info(f"[REPLAY INIT] Teleop mode - hand_data shape = {replay_hand_data.shape}")

                    replay_demo_init = True
                    replay_frame_idx = 0
                    logger_mp.info(f"[REPLAY INIT] Loaded {replay_length} frames from {current_mode} mode")
                    if "timestamp" in df.columns:
                        ts = df["timestamp"].to_numpy().astype(np.float64)
                        ts = ts - ts[0]                 # 0초 기준 상대시간
                        _replay_rel_ts = ts
                        _replay_start_wall = time.perf_counter()
                    else:
                        _replay_rel_ts = None
                        _replay_start_wall = None

                    # 고정 20Hz 재생을 위한 다음 예약 시각 (첫 프레임은 즉시 송출)
                    _replay_next_t = time.perf_counter()
                    _last_written_idx = -1

                if replay and replay_demo_init:
                    now = time.perf_counter()

                    # --- 프레임 인덱스 결정: 타임스탬프 기반 또는 고정 20Hz ---
                    if _replay_rel_ts is not None:
                        # 기록된 타임스탬프 기반 (가변 간격)
                        elapsed = now - _replay_start_wall
                        target_idx = int(np.searchsorted(_replay_rel_ts, elapsed, side="right")) - 1
                        target_idx = max(0, min(target_idx, replay_length - 1))
                        replay_frame_idx = target_idx
                    else:
                        # 고정 20Hz: 예약 시각이 지났을 때만 한 프레임 전진
                        if now >= _replay_next_t:
                            if replay_frame_idx < replay_length - 1:
                                replay_frame_idx += 1
                            # 지연 누적분 보정(드리프트 억제)
                            while _replay_next_t <= now:
                                _replay_next_t += _replay_period
                        # 아직 시간이 안 됐으면 같은 프레임 유지

                    idx = replay_frame_idx
                    hand_vec = replay_hand_data[idx]

                    if current_mode in ['kistar_teleop', 'kistar_inspire_teleop']:
                        # KISTAR 모드: kistar_hand_joints_shm에 16개 데이터를 저장
                        # dual_hand_action_array는 사용하지 않음
                        kistar_hand_action_shm.write_data(
                            hand_action=hand_vec[:16]  # KISTAR hand 16개 관절 데이터
                        )
                        logger_mp.info(f"[REPLAY] KISTAR mode - frame {idx}: stored 16 joints to kistar_hand_action_shm")

                        # dual_hand_action_array는 0으로 설정 (사용하지 않음)
                        with dual_hand_data_lock:
                            dual_hand_state_array[:] = np.zeros(12)
                            dual_hand_action_array[:] = np.zeros(12)

                    else:
                        # 기존 teleop 모드: 왼손 6개 + 오른손 6개 = 12개
                        left_q_target  = hand_vec[:6]
                        right_q_target = hand_vec[6:12]
                        logger_mp.info(f"[REPLAY] Teleop mode - frame {idx}: left={left_q_target[:3]}, right={right_q_target[:3]}")

                        # 3) 배열에 저장
                        with dual_hand_data_lock:
                            dual_hand_state_array[:] = hand_vec
                            dual_hand_action_array[:] = hand_vec

                    # 5) 다음 프레임으로
                    replay_frame_idx += 1
                    if replay_frame_idx >= replay_length - 1:
                        pass
                
                if deploy : 
                    robot_dict = robot_action_shm.read_data()
                    hand_action = robot_dict["action_hand"]
                    hand_action = np.clip(hand_action, a_min=None, a_max=1.0)

                    left_q_target  = hand_action[:6]
                    right_q_target = hand_action[6:]
                    
                    # 배열에 저장
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = hand_action
                        dual_hand_action_array[:] = hand_action
            
                rate.sleep()

            elif state is State.PAUSE:
                time.sleep(0.01)

    finally:
        logger_mp.info("[Hand_Ctrl] 종료 신호 수신. 종료합니다.")
        
        # 디버깅 시각화 종료
        if debug_visualizer:
            debug_visualizer.stop_visualization()
            
        television_shm.worker_close()
        freq_shm.worker_close()
        record_mode_shm.worker_close()
        record_episode_shm.worker_close()
        record_task_shm.worker_close()
        robot_obs_shm.worker_close()
        robot_action_shm.worker_close()
        kistar_hand_action_shm.worker_close()