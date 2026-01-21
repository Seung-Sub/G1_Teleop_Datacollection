# workers/worker_zmq_kistar_sender.py
import time
import zmq
import io
import numpy as np
from multiprocessing.synchronize import Lock
from multiprocessing.shared_memory import SharedMemory
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import ROBOT_OBS, ROBOT_ACTION, KISTAR_HAND_ACTION, RECORD_MODE_LAYOUT

class ZMQConfig:
    # NUC IP 설정
    NUC_IP = "192.168.6.10"  # NUC IP 주소
    
    # 제어 명령 전송 (A → NUC)
    CONTROL_PUB_IP = "192.168.6.11"  # 본체 IP
    CONTROL_PUB_PORT = 6665
    
    # 타임아웃 설정
    TIMEOUT = 1000  # ms

class ZMQKistarSender:
    def __init__(self, events, shm_names, locks):
        self.events = events
        self.shm_names = shm_names
        self.locks = locks
        
        # ZMQ 컨텍스트
        self.context = None
        self.pub_socket = None
        
        # 제어 명령 카운터
        self.step_cnt = 0
       
        self.kistar_hand_action_shm = SharedMemoryManager(KISTAR_HAND_ACTION, locks["kistar_hand_action_lock"], shm_names["kistar_hand_action_shm"])
        self.record_mode_shm = SharedMemoryManager(RECORD_MODE_LAYOUT, locks["record_lock"], shm_names["record_mode_shm"])
        
    def on_start(self):
        """ZMQ 소켓 초기화"""
        print("[ZMQ KISTAR Sender] 초기화 중...")
        
        self.context = zmq.Context()
        
        # Publisher: 제어 명령 전송 (A → NUC)
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{ZMQConfig.CONTROL_PUB_IP}:{ZMQConfig.CONTROL_PUB_PORT}")
        time.sleep(0.1)  # 바인딩 대기
        
        print(f"[ZMQ KISTAR Sender] {ZMQConfig.CONTROL_PUB_IP}:{ZMQConfig.CONTROL_PUB_PORT}에 바인딩 완료")
        
    def on_finish(self):
        """리소스 정리"""
        print("[ZMQ KISTAR Sender] 종료 중...")
        
        if self.pub_socket:
            self.pub_socket.close()
        if self.context:
            self.context.term()
            
    def send_control_command(self, joint_angles):
        """NUC로 제어 명령 전송"""
        try:
            # NUC 형식으로 데이터 구성 (16자유도 hand_joint_targets만)
            control_data = {
                'hand_joint_targets': joint_angles  # 16개 관절 각도
            }
            
            # NPZ 형식으로 압축하여 전송
            with io.BytesIO() as buf:
                np.savez_compressed(buf, **control_data)
                self.pub_socket.send(buf.getvalue())
                
            self.step_cnt += 1
                
        except Exception as e:
            print(f"[ZMQ KISTAR Sender] 전송 오류: {e}")
            
    def task(self):
        """메인 작업 루프"""
        try:
            # KISTAR Hand 전용 shared memory에서 관절 각도 읽기
            kistar_data = self.kistar_hand_action_shm.read_data()
            joint_angles = kistar_data["hand_action"]  # 16개 관절 각도

            # 모드 확인 (start 또는 replay 중 하나라도 활성화되어 있으면 데이터 전송)
            mode_data = self.record_mode_shm.read_data()
            is_active = self.events['set_start'].is_set() or bool(mode_data["replay"])

            if is_active:
                
                # start 또는 replay가 활성화되었을 때: KISTAR Hand shared memory에서 읽은 값 사용
                pass  # joint_angles는 이미 kistar_data에서 읽은 값
            else:
                default_thumb_0=20/180*np.pi
                default_thumb_1=-90/180*np.pi
                default_thumb_2=30/180*np.pi
                # 아무 모드도 활성화되지 않았을 때: 초기 자세 (손가락을 펼친 상태)
                # KISTAR Hand는 오른손 16자유도만 사용
                joint_angles = np.array([
                    # 엄지 4개 관절
                    default_thumb_0, default_thumb_1, default_thumb_2, 0.0,
                    # 검지 3개 관절
                    0.0, 0.0, 0.0,
                    # 중지 3개 관절
                    0.0, 0.0, 0.0,
                    # 약지 3개 관절
                    0.0, 0.0, 0.0,
                    # 소지 3개 관절
                    0.0, 0.0, 0.0,
                ], dtype=np.float32)
            
            # print('########################################################')
            # print(joint_angles)
            # print('########################################################')
            self.send_control_command(joint_angles)
                        
        except Exception as e:
            print(f"[ZMQ KISTAR Sender] 작업 오류: {e}")
            
        time.sleep(0.02)  # 50Hz

def worker_zmq_kistar_sender(events, shm_names, locks):
    """ZMQ KISTAR 송신 워커 실행 함수"""
    sender = ZMQKistarSender(events, shm_names, locks)
    
    try:
        sender.on_start()
        
        while not events['shutdown'].is_set():
            sender.task()
            
    except KeyboardInterrupt:
        print("[ZMQ KISTAR Sender] 키보드 인터럽트로 종료")
    except Exception as e:
        print(f"[ZMQ KISTAR Sender] 예외 발생: {e}")
    finally:
        sender.on_finish()
