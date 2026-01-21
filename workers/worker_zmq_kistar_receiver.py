# workers/worker_zmq_kistar_receiver.py
import time
import zmq
import io
import numpy as np
from multiprocessing.synchronize import Lock
from multiprocessing.shared_memory import SharedMemory
from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import KISTAR_HAND_RECEIVED

class ZMQConfig:
    # NUC IP 설정
    NUC_IP = "192.168.6.10"  # ASUS NUC IP 주소

    # ASUS NUC로부터 데이터 수신 (NUC → A)
    NUC_PUB_IP = NUC_IP
    NUC_PUB_PORT = 6666  # ASUS NUC에서 데이터를 발행하는 포트

    # 타임아웃 설정
    TIMEOUT = 1000  # ms

class ZMQKistarReceiver:
    def __init__(self, events, shm_names, locks):
        self.events = events
        self.shm_names = shm_names
        self.locks = locks

        # ZMQ 컨텍스트
        self.context = None
        self.sub_socket = None

        # 수신된 데이터 카운터
        self.recv_cnt = 0

        # ASUS NUC로부터 받은 데이터를 저장할 공유 메모리
        self.kistar_hand_received_shm = SharedMemoryManager(
            KISTAR_HAND_RECEIVED,
            locks["kistar_hand_received_lock"],
            shm_names["kistar_hand_received_shm"]
        )

        # 마지막으로 받은 데이터 저장 (연결 끊김 시 사용)
        self.last_hand_q_pos = np.zeros(16, dtype=np.float32)
        self.last_play_cnt = 0

    def on_start(self):
        """ZMQ 소켓 초기화"""
        print("[ZMQ KISTAR Receiver] 초기화 중...")

        self.context = zmq.Context()

        # Subscriber: ASUS NUC로부터 데이터 수신 (NUC → A)
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{ZMQConfig.NUC_PUB_IP}:{ZMQConfig.NUC_PUB_PORT}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')  # 모든 메시지 구독
        self.sub_socket.setsockopt(zmq.RCVTIMEO, ZMQConfig.TIMEOUT)  # 타임아웃 설정

        print(f"[ZMQ KISTAR Receiver] {ZMQConfig.NUC_PUB_IP}:{ZMQConfig.NUC_PUB_PORT}에 연결됨")

    def on_finish(self):
        """리소스 정리"""
        print("[ZMQ KISTAR Receiver] 종료 중...")

        if self.sub_socket:
            self.sub_socket.close()
        if self.context:
            self.context.term()

    def receive_data(self):
        """ASUS NUC로부터 데이터 수신"""
        try:
            # ASUS NUC에서 보낸 압축된 NPZ 데이터 수신
            compressed_data = self.sub_socket.recv()

            # NPZ 데이터 압축 해제
            with io.BytesIO(compressed_data) as buf:
                data_dict = np.load(buf)

                # 데이터 추출
                hand_q_pos = data_dict['hand_q_pos']  # 16차원 배열

                # print("hand q pos type: ",hand_q_pos.dtype)

                play_cnt = data_dict['play_cnt']       # 플레이 카운트

                # print("play_cnt: ",play_cnt,"   /   data received: ",hand_q_pos)
                
                # print("play_cnt: ",play_cnt)

                # 데이터 유효성 검증
                if len(hand_q_pos) != 16:
                    print(f"[ZMQ KISTAR Receiver] 잘못된 hand_q_pos 크기: {len(hand_q_pos)} (16이어야 함)")
                    return False

                # 마지막 데이터 저장
                self.last_hand_q_pos = hand_q_pos.copy()
                self.last_play_cnt = int(play_cnt)

                #print(self.last_hand_q_pos)
                # 공유 메모리에 데이터 저장
                self.kistar_hand_received_shm.write_data(
                    hand_q_pos=self.last_hand_q_pos,
                    play_cnt=np.int32(self.last_play_cnt)
                )

                self.recv_cnt += 1

                # # 디버그 출력 (10번에 1번)
                # if self.recv_cnt % 10 == 0:
                #     print(f"[ZMQ KISTAR Receiver] 수신 #{self.recv_cnt}: "
                #           f"hand_q_pos[:4]={hand_q_pos[:4]}, play_cnt={play_cnt}")

                return True

        except zmq.Again:
            # 타임아웃 발생 (ASUS NUC 연결 끊김)
            # print("[ZMQ KISTAR Receiver] 타임아웃 - ASUS NUC 연결 확인 필요")

            # 마지막으로 받은 데이터로 공유 메모리 업데이트 (연결 복구 시까지)
            self.kistar_hand_received_shm.write_data(
                hand_q_pos=self.last_hand_q_pos,
                play_cnt=np.int32(self.last_play_cnt)
            )
            return False

        except Exception as e:
            print(f"[ZMQ KISTAR Receiver] 수신 오류: {e}")
            return False

    def task(self):
        """메인 작업 루프"""
        try:
            # ASUS NUC로부터 데이터 수신
            success = self.receive_data()

            if not success:
                # 수신 실패 시 마지막 데이터 유지
                pass

        except Exception as e:
            print(f"[ZMQ KISTAR Receiver] 작업 오류: {e}")

        time.sleep(0.005)  # 200Hz

def worker_zmq_kistar_receiver(events, shm_names, locks):
    """ZMQ KISTAR 수신 워커 실행 함수"""
    receiver = ZMQKistarReceiver(events, shm_names, locks)

    try:
        receiver.on_start()

        while not events['shutdown'].is_set():
            receiver.task()

    except KeyboardInterrupt:
        print("[ZMQ KISTAR Receiver] 키보드 인터럽트로 종료")
    except Exception as e:
        print(f"[ZMQ KISTAR Receiver] 예외 발생: {e}")
    finally:
        receiver.on_finish()
