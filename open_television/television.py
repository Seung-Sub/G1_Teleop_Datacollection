import time
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands
from multiprocessing import Array, Process, shared_memory, Lock
import numpy as np
import asyncio
import cv2
import threading

from multiprocessing import context
Value = context._default_context.Value

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA


class TeleVision:
    def __init__(self, binocular, img_shape, img_shm_name, cert_file="./cert.pem", key_file="./key.pem", ngrok=False):
        self.binocular = binocular
        self.img_height = img_shape[0]
        if binocular:
            self.img_width  = img_shape[1] // 2
        else:
            self.img_width  = img_shape[1]

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.vuer = Vuer(host='0.0.0.0', port=8012, queries=dict(grid=False), queue_len=3)

        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)

        # existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        # self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)
        self.lock = Lock()

        self.camera_image = SharedMemoryManager(CAMERA, self.lock, "camera_shm")

        if binocular:
            self.vuer.spawn(start=False)(self.main_image_binocular)
        else:
            self.vuer.spawn(start=False)(self.main_image_monocular)

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        # self.left_landmarks_shared = Array('d', 75, lock=True)
        # self.right_landmarks_shared = Array('d', 75, lock=True)
        

        self.left_landmarks_shared = Array('d', 5 * 3, lock=True)  # 각 손가락 tip의 (x, y, z) 위치 값
        self.right_landmarks_shared = Array('d', 5 * 3, lock=True)  # 각 손가락 tip의 (x, y, z) 위치 값

        # 추가: 오른손 distal/proximal (각각 5점 * 3)
        self.right_distal_landmarks_shared = Array('d', 5 * 3, lock=True)
        self.right_proximal_landmarks_shared = Array('d', 5 * 3, lock=True)


        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        # self.process = Process(target=self.vuer_run)
        # self.process.daemon = True
        # self.process.start()

        self.thread = threading.Thread(target=self.vuer_run, daemon=True, name="VUER_THREAD")
        self.thread.start()

    def vuer_run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.vuer.run()

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        # try:
        #     left_data = event.value.get("left", [0] * 400)  # 데이터가 없으면 0으로 채움
        #     if isinstance(left_data, list) and len(left_data) >= 400:
        #         # Wrist 변환 행렬 업데이트
        #         self.left_hand_shared[:] = left_data[:16]
        #     else:
        #           # 데이터가 유효하지 않은 경우 0으로 채움
        #         self.left_hand_shared[:] = [0] * 16
                
        #     right_data = event.value.get("right", [0] * 400)  # 데이터가 없으면 0으로 채움
        #     if isinstance(right_data, list) and len(right_data) >= 400:
        #         # Wrist 변환 행렬 업데이트
        #         self.right_hand_shared[:] = right_data[:16]
        #     else:
        #         # 데이터가 유효하지 않은 경우 0으로 채움
        #         self.right_hand_shared[:] = [0] * 16
                
        #     self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
        #     self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        # except: 
        #     pass
    
        try:                
            tip_indices = [4, 9, 14, 19, 24]  # 각 손가락 tip의 인덱스
            distal_indices = [3, 8, 13, 18, 23]
            proximal_indices = [2, 7, 12, 17, 22]

            # 왼손 데이터 처리
            left_data = event.value.get("left", [0] * 400)  # 데이터가 없으면 0으로 채움
            if isinstance(left_data, list) and len(left_data) >= 400:
                # Wrist 변환 행렬 업데이트
                self.left_hand_shared[:] = left_data[:16]
                
                scale_factor = 1.1 
                
                # 손가락 tip 위치 추출
                left_data = np.array(left_data).reshape(25, 16)
                left_positions = left_data[np.ix_(tip_indices, [12, 13, 14])].flatten()  # 각 tip의 (x, y, z) 위치
                left_distal_positions = left_data[np.ix_(distal_indices, [12, 13, 14])].flatten()  # 각 tip의 (x, y, z) 위치
                left_proximal_positions = left_data[np.ix_(proximal_indices, [12, 13, 14])].flatten()  # 각 tip의 (x, y, z) 위치

            else:
                # 데이터가 유효하지 않은 경우 0으로 채움
                self.left_hand_shared[:] = [0] * 16
                left_positions = [0] * 15  # 5개의 tip 위치 (x, y, z)

            # 왼손 landmarks 위치 데이터 저장
            self.left_landmarks_shared[:] = left_positions
            # print("Left landmarks:", self.left_landmarks_shared[:])

            # 오른손 데이터 처리
            right_data = event.value.get("right", [0] * 400)  # 데이터가 없으면 0으로 채움
            if isinstance(right_data, list) and len(right_data) >= 400:
                # Wrist 변환 행렬 업데이트
                self.right_hand_shared[:] = right_data[:16]
                
                # 손가락 tip 위치 추출
                right_data = np.array(right_data).reshape(25, 16)
                right_positions = right_data[np.ix_(tip_indices, [12, 13, 14])].flatten()  # 각 tip의 (x, y, z) 위치
                right_distal_positions = right_data[np.ix_(distal_indices, [12, 13, 14])].flatten()  # 각 손가락 distal (x, y, z)
                right_proximal_positions = right_data[np.ix_(proximal_indices, [12, 13, 14])].flatten()  # 각 손가락 proximal (x, y, z)
                
            else:
                # 데이터가 유효하지 않은 경우 0으로 채움
                self.right_hand_shared[:] = [0] * 16
                right_positions = [0] * 15  # 5개의 tip 위치 (x, y, z)

            # 오른손 landmarks 위치 데이터 저장
            self.right_landmarks_shared[:] = right_positions
            self.right_distal_landmarks_shared[:] = right_distal_positions if 'right_distal_positions' in locals() else [0] * 15
            self.right_proximal_landmarks_shared[:] = right_proximal_positions if 'right_proximal_positions' in locals() else [0] * 15
            # print("Right landmarks:", self.right_landmarks_shared[:])

        except Exception as e:
            print(f"Error processing hand data: {e}")
            pass

    async def main_image_binocular(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        while True:
            try:
                data_dict = self.camera_image.read_data()
            except Exception as e:
                # 읽기 실패 시 간단히 리턴 (로그를 남겨도 좋음)
                return

            left_raw = data_dict.get("camera_left", None)
            left_display_image = cv2.cvtColor(left_raw, cv2.COLOR_BGR2RGB)            
            
            right_raw = data_dict.get("camera_right", None)
            right_display_image = cv2.cvtColor(right_raw, cv2.COLOR_BGR2RGB)
            aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        left_display_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=50,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        right_display_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=50,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        while True:
            # display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)


            try:
                data_dict = self.camera_image.read_data()
            except Exception as e:
                # 읽기 실패 시 간단히 리턴 (로그를 남겨도 좋음)
                return

            raw = data_dict.get("camera_color", None)
            display_image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

            session.upsert(
                [
                    ImageBackground(
                        display_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=50,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def left_landmarks(self):
        # return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.left_landmarks_shared[:]).reshape(5, 3)

    @property
    def right_landmarks(self):
        # return np.array(self.right_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.right_landmarks_shared[:]).reshape(5, 3)

    @property
    def right_distal_landmarks(self):
        return np.array(self.right_distal_landmarks_shared[:]).reshape(5, 3)

    @property
    def right_proximal_landmarks(self):
        return np.array(self.right_proximal_landmarks_shared[:]).reshape(5, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)
    
if __name__ == '__main__':
    import os 
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    import threading
    from image_server.image_client import ImageClient

    # image
    img_shape = (480, 640 * 2, 3)
    img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)
    img_client = ImageClient(tv_img_shape = img_shape, tv_img_shm_name = img_shm.name)
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()

    # television
    tv = TeleVision(True, img_shape, img_shm.name)
    print("vuer unit test program running...")
    print("you can press ^C to interrupt program.")
    while True:
        time.sleep(0.03)