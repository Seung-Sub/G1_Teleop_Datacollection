import time
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands
try:
    # vuer >= 0.0.40 ships MotionControllers. Older versions may not.
    from vuer.schemas import MotionControllers
    _HAS_MOTION_CONTROLLERS = True
except Exception:
    MotionControllers = None
    _HAS_MOTION_CONTROLLERS = False
from multiprocessing import Array, Process, shared_memory, Lock
import numpy as np
import asyncio
import cv2
import threading

from multiprocessing import context
Value = context._default_context.Value

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import CAMERA_VIEW


class TeleVision:
    def __init__(self, binocular, img_shape, img_shm_name,
                 cert_file="./cert.pem", key_file="./key.pem",
                 ngrok=False, vr_input="hand", camera_shm_key="rs_ego_shm"):
        """
        vr_input:
            "hand"       - Quest3 hand tracking (Vuer HAND_MOVE 이벤트)
            "controller" - Quest3 controller (Vuer CONTROLLER_MOVE 이벤트)
            HMD를 머리에 쓰든 목에 걸든 동일한 채널을 사용. controller 모드는
            wrist target으로 controller pose를, button/trigger/squeeze 를 추가
            shared array에 적재한다.
        """
        self.binocular = binocular
        self.vr_input = vr_input
        self.img_height = img_shape[0]
        if binocular:
            self.img_width  = img_shape[1] // 2
        else:
            self.img_width  = img_shape[1]

        # Vuer 가 cert/key 받으면 HTTPS, 아니면 HTTP. WebXR (Quest3 안 브라우저) 는
        # HTTPS 강제이므로 cert/key 가 실제 존재할 때만 전달. 둘 다 없으면 HTTP 로 시작
        # (로컬 개발 환경 / 테스트 용도).
        import os as _os
        _cert_kwargs = {}
        if cert_file and key_file and _os.path.exists(cert_file) and _os.path.exists(key_file):
            _cert_kwargs = {"cert": cert_file, "key": key_file}
        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3, **_cert_kwargs)
        else:
            self.vuer = Vuer(host='0.0.0.0', port=8012, queries=dict(grid=False), queue_len=3, **_cert_kwargs)

        # 양쪽 핸들러 모두 등록 — Vuer는 활성 입력 종류에 맞는 이벤트만 발화한다.
        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if _HAS_MOTION_CONTROLLERS:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        # existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        # self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)
        self.lock = Lock()

        # Part5: 표시 경로 = 데이터 경로 (CAMERA_VIEW). default 는 ego 시점 (헤드셋
        # 사용자 view). 멀티 카메라 setup 에서 wrist 카메라를 Vuer 화면에 띄울 필요
        # 는 없으니 ego 고정. 외부에서 camera_shm_key 로 override 가능.
        self.camera_image = SharedMemoryManager(CAMERA_VIEW, self.lock, camera_shm_key)

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

        # ---- Quest3 controller shared state (vr_input="controller" 시 사용) ----
        # 4x4 SE(3) — Vuer는 column-major 16-float, on_controller_move 에서 그대로 저장
        self.left_ctrl_shared  = Array('d', 16, lock=True)
        self.right_ctrl_shared = Array('d', 16, lock=True)
        # state floats: [trigger, squeeze, thumb_x, thumb_y, a_or_x, b_or_y, thumb_click]
        self.left_ctrl_state_shared  = Array('d', 7, lock=True)
        self.right_ctrl_state_shared = Array('d', 7, lock=True)
        # 한 번이라도 controller 이벤트가 들어왔는지
        self.controller_connected = Value('i', 0, lock=True)

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

    async def on_controller_move(self, event, session, fps=60):
        """
        Quest3 controller 이벤트 처리.
        Vuer CONTROLLER_MOVE payload (vuer >= 0.0.40 기준):
          event.value = {
            "left":  [16 floats column-major SE(3)],
            "right": [16 floats column-major SE(3)],
            "leftState":  {"trigger":bool, "triggerValue":float, "squeeze":bool,
                           "squeezeValue":float, "thumbstick":bool,
                           "thumbstickValue":[x,y], "aButton":bool, "bButton":bool},
            "rightState": {... 동일 ...}
          }
        ※ aButton/bButton: 우측 컨트롤러는 A/B, 좌측은 X/Y에 해당.
        """
        try:
            v = event.value or {}

            left_pose  = v.get("left")
            right_pose = v.get("right")
            if isinstance(left_pose, list) and len(left_pose) >= 16:
                self.left_ctrl_shared[:] = left_pose[:16]
            if isinstance(right_pose, list) and len(right_pose) >= 16:
                self.right_ctrl_shared[:] = right_pose[:16]

            def _state_to_array(state):
                if not isinstance(state, dict):
                    return [0.0] * 7
                ts = state.get("thumbstickValue") or [0.0, 0.0]
                tx = float(ts[0]) if len(ts) > 0 else 0.0
                ty = float(ts[1]) if len(ts) > 1 else 0.0
                return [
                    float(state.get("triggerValue", 0.0)),
                    float(state.get("squeezeValue", 0.0)),
                    tx, ty,
                    1.0 if state.get("aButton", False) else 0.0,
                    1.0 if state.get("bButton", False) else 0.0,
                    1.0 if state.get("thumbstick", False) else 0.0,
                ]

            left_state  = _state_to_array(v.get("leftState"))
            right_state = _state_to_array(v.get("rightState"))
            self.left_ctrl_state_shared[:]  = left_state
            self.right_ctrl_state_shared[:] = right_state

            # 한 번이라도 들어왔으면 connected=True 로 고정
            self.controller_connected.value = 1
        except Exception as e:
            print(f"Error processing controller data: {e}")

    async def main_image_binocular(self, session, fps=60):
        if self.vr_input == "controller" and _HAS_MOTION_CONTROLLERS:
            session.upsert @ MotionControllers(fps=fps, stream=True, key="ctrls",
                                               left=True, right=True)
        else:
            session.upsert @ Hands(fps=fps, stream=True, key="hands",
                                   showLeft=False, showRight=False)
        while True:
            try:
                data_dict = self.camera_image.read_data()
            except Exception as e:
                # 읽기 실패 시 간단히 리턴 (로그를 남겨도 좋음)
                return

            # Part5: CAMERA_VIEW schema (frame_left/frame_right/is_stereo).
            # 빈 프레임 (전부 0) 이면 표시 skip — 워커 미동작 / 카메라 미연결 가드.
            left_raw  = data_dict.get("frame_left",  None)
            right_raw = data_dict.get("frame_right", None)
            if left_raw is None or not left_raw.any():
                await asyncio.sleep(0.016)
                continue
            # is_stereo=0 (RealSense mono) 일 때 frame_right 가 zero — 양안 모두
            # frame_left 사용 (헤드셋에서 한쪽 눈만 보이지 않게).
            is_stereo = int(data_dict.get("is_stereo", 0)) if "is_stereo" in data_dict else 0
            if not is_stereo or right_raw is None or not right_raw.any():
                right_raw = left_raw
            left_display_image  = cv2.cvtColor(left_raw,  cv2.COLOR_BGR2RGB)
            right_display_image = cv2.cvtColor(right_raw, cv2.COLOR_BGR2RGB)
            aspect_ratio = self.img_width / self.img_height
            try:
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
            except (AssertionError, KeyError) as e:
                # VR(WebXR) 세션이 끊기면 vuer 가 "Websocket session is missing" 을 던진다.
                # teleop 종료/연결 끊김 시 정상 현상이므로 조용히 루프 종료 (스택트레이스 방지).
                print(f"[TeleVision] VR session closed during upsert — image stream stop ({type(e).__name__}).")
                return
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        # NOTE: 현재 worker_vr 는 항상 binocular=True 로 TeleVisionWrapper 를 생성하므로
        # 이 분기는 실행되지 않는다. 향후 single-view 모드를 enable 할 때를 위해
        # CAMERA schema 에 존재하는 'realsense' 키로 정렬해 둠 ('camera_color' 는
        # schema 에 없는 키였음).
        if self.vr_input == "controller" and _HAS_MOTION_CONTROLLERS:
            session.upsert @ MotionControllers(fps=fps, stream=True, key="ctrls",
                                               left=True, right=True)
        else:
            session.upsert @ Hands(fps=fps, stream=True, key="hands",
                                   showLeft=False, showRight=False)
        while True:
            try:
                data_dict = self.camera_image.read_data()
            except Exception as e:
                return

            # Part5: CAMERA_VIEW schema 의 frame_left 사용 (mono 표시).
            raw = data_dict.get("frame_left", None)
            if raw is None or not raw.any():
                # frame 이 아직 안 들어왔거나 전부 0 → 다음 cycle 까지 대기.
                await asyncio.sleep(0.016)
                continue
            display_image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

            try:
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
            except (AssertionError, KeyError) as e:
                print(f"[TeleVision] VR session closed during upsert (mono) — stream stop ({type(e).__name__}).")
                return
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

    # ---- Quest3 controller properties (vr_input="controller") ----
    @property
    def left_ctrl_pose(self):
        return np.array(self.left_ctrl_shared[:]).reshape(4, 4, order="F")

    @property
    def right_ctrl_pose(self):
        return np.array(self.right_ctrl_shared[:]).reshape(4, 4, order="F")

    @property
    def left_ctrl_state(self):
        # [trigger, squeeze, thumb_x, thumb_y, a, b, thumb_click]
        return np.array(self.left_ctrl_state_shared[:], dtype=np.float64)

    @property
    def right_ctrl_state(self):
        return np.array(self.right_ctrl_state_shared[:], dtype=np.float64)

    @property
    def is_controller_connected(self):
        return bool(self.controller_connected.value)
    
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