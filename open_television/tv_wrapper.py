import numpy as np
from open_television.television import TeleVision
from open_television.constants import *
from utils.mat_tool import mat_update, fast_mat_inv

"""
(basis) OpenXR Convention : y up, z back, x right. 
(basis) Robot  Convention : z up, y left, x front.  
p.s. Vuer's all raw data follows OpenXR Convention, WORLD coordinate.

under (basis) Robot Convention, wrist's initial pose convention:

    # (Left Wrist) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (Right Wrist) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.
  
    # (Left Wrist URDF) Unitree Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from palm toward back of the hand.
        - the z-axis pointing from pinky toward index.

    # (Right Wrist URDF) Unitree Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from back of the hand toward palm. 
        - the z-axis pointing from pinky toward index.

under (basis) Robot Convention, hand's initial pose convention:

    # (Left Hand) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (Right Hand) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.

    # (Left Hand URDF) Unitree Convention:   
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from pinky toward index.

    # (Right Hand URDF) Unitree Convention: 
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from index toward pinky. 

    p.s. From website: https://registry.khronos.org/OpenXR/specs/1.1/man/html/openxr.html.
         You can find **(Left/Right Wrist) XR/AppleVisionPro Convention** related information like this below:
           "The wrist joint is located at the pivot point of the wrist, which is location invariant when twisting the hand without moving the forearm. 
            The backward (+Z) direction is parallel to the line from wrist joint to middle finger metacarpal joint, and points away from the finger tips. 
            The up (+Y) direction points out towards back of the hand and perpendicular to the skin at wrist. 
            The X direction is perpendicular to the Y and Z directions and follows the right hand rule."
         Note: The above context is of course under **(basis) OpenXR Convention**.

    p.s. **(Wrist/Hand URDF) Unitree Convention** information come from URDF files.
"""

class TeleVisionWrapper:
    def __init__(self, binocular, img_shape, img_shm_name, vr_input="hand"):
        self.tv = TeleVision(binocular, img_shape, img_shm_name, vr_input=vr_input)
        self.vr_input = vr_input

    def get_data(self):

        # --------------------------------wrist-------------------------------------

        # TeleVision obtains a basis coordinate that is OpenXR Convention
        head_vuer_mat, head_flag = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())
        left_wrist_vuer_mat, left_wrist_flag  = mat_update(const_left_wrist_vuer_mat, self.tv.left_hand.copy())
        right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.tv.right_hand.copy())

        # Change basis convention: VuerMat ((basis) OpenXR Convention) to WristMat ((basis) Robot Convention)
        # p.s. WristMat = T_{robot}_{openxr} * VuerMat * T_{robot}_{openxr}^T
        # Reason for right multiply fast_mat_inv(T_robot_openxr):
        #   This is similarity transformation: B = PAP^{-1}, that is B ~ A
        #   For example:
        #   - For a pose data T_r under the Robot Convention, left-multiplying WristMat means:
        #   - WristMat * T_r  ==>  T_{robot}_{openxr} * VuerMat * T_{openxr}_{robot} * T_r
        #   - First, transform to the OpenXR Convention (The function of T_{openxr}_{robot})
        #   - then, apply the rotation VuerMat in the OpenXR Convention (The function of VuerMat)
        #   - finally, transform back to the Robot Convention (The function of T_{robot}_{openxr})
        #   This results in the same rotation effect under the Robot Convention as in the OpenXR Convention.
        head_mat = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
        left_wrist_mat  = T_robot_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)
        right_wrist_mat = T_robot_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)

        # Change wrist convention: WristMat ((Left Wrist) XR/AppleVisionPro Convention) to UnitreeWristMat((Left Wrist URDF) Unitree Convention)
        # Reason for right multiply (T_to_unitree_left_wrist) : Rotate 90 degrees counterclockwise about its own x-axis.
        # Reason for right multiply (T_to_unitree_right_wrist): Rotate 90 degrees clockwise about its own x-axis.
        unitree_left_wrist = left_wrist_mat @ (T_to_unitree_left_wrist if left_wrist_flag else np.eye(4))
        unitree_right_wrist = right_wrist_mat @ (T_to_unitree_right_wrist if right_wrist_flag else np.eye(4))

        # Transfer from WORLD to HEAD coordinate (translation only).
        unitree_left_wrist[0:3, 3]  = unitree_left_wrist[0:3, 3] #- head_mat[0:3, 3]
        unitree_right_wrist[0:3, 3] = unitree_right_wrist[0:3, 3] #- head_mat[0:3, 3]

        # --------------------------------hand-------------------------------------
        left_hand_vuer_mat  = np.concatenate([self.tv.left_landmarks.copy().T, np.ones((1, self.tv.left_landmarks.shape[0]))])
        right_hand_vuer_mat = np.concatenate([self.tv.right_landmarks.copy().T, np.ones((1, self.tv.right_landmarks.shape[0]))])

        left_hand_mat  = grd_yup2grd_zup @ left_hand_vuer_mat
        right_hand_mat = grd_yup2grd_zup @ right_hand_vuer_mat


        left_hand_mat_wb  = fast_mat_inv(left_wrist_mat) @ left_hand_mat
        right_hand_mat_wb = fast_mat_inv(right_wrist_mat) @ right_hand_mat
             
        unitree_left_hand  = (hand2inspire.T @ left_hand_mat_wb)[0:3, :].T
        unitree_right_hand = (hand2inspire.T @ right_hand_mat_wb)[0:3, :].T
        # --------------------------------offset-------------------------------------

        # head_rmat = head_mat[:3, :3]
        head_rmat = head_mat
        # The origin of the coordinate for IK Solve is the WAIST joint motor. You can use teleop/robot_control/robot_arm_ik.py Unit_Test to check it.
        # The origin of the coordinate of unitree_left_wrist is HEAD. So it is necessary to translate the origin of unitree_left_wrist from HEAD to WAIST.
        # unitree_left_wrist[0, 3] +=0.15
        # unitree_right_wrist[0,3] +=0.15
        # unitree_left_wrist[2, 3] +=0.45
        # unitree_right_wrist[2,3] +=0.45
        
        scale = 1.1 

        return head_rmat, unitree_left_wrist, unitree_right_wrist, unitree_left_hand, unitree_right_hand

    def get_data_with_segments(self):
        """
        기존 get_data() 반환값에 오른손 distal/proximal 포인트(각 5x3)를 추가로 포함하여 반환.
        반환: (head_rmat, unitree_left_wrist, unitree_right_wrist,
              unitree_left_hand, unitree_right_hand,
              right_distal_points, right_proximal_points)
        """
        head_rmat, unitree_left_wrist, unitree_right_wrist, unitree_left_hand, unitree_right_hand = self.get_data()

        # 원본(오픈XR→로봇 변환 전)의 오른손 포인트를 가져와 동일 변환 적용
        # tv.right_distal_landmarks / tv.right_proximal_landmarks: (5,3)
        right_distal_vuer = np.concatenate([self.tv.right_distal_landmarks.copy().T, np.ones((1, self.tv.right_distal_landmarks.shape[0]))])
        right_proximal_vuer = np.concatenate([self.tv.right_proximal_landmarks.copy().T, np.ones((1, self.tv.right_proximal_landmarks.shape[0]))])

        right_distal_mono = grd_yup2grd_zup @ right_distal_vuer
        right_proximal_mono = grd_yup2grd_zup @ right_proximal_vuer

        # 손목 기준 좌표계로 변환 후 Unitree Hand 기준으로 보정
        _, _, right_wrist_mat, _, _ = self.get_data()
        right_distal_wb = fast_mat_inv(right_wrist_mat) @ right_distal_mono
        right_proximal_wb = fast_mat_inv(right_wrist_mat) @ right_proximal_mono

        right_distal_points = (hand2inspire.T @ right_distal_wb)[0:3, :].T
        right_proximal_points = (hand2inspire.T @ right_proximal_wb)[0:3, :].T

        scale = np.array([[1.1, 1.1, 1.06, 1.12, 1.06]]).T 

        unitree_left_hand = scale * unitree_left_hand
        unitree_right_hand = scale * unitree_right_hand

        right_distal_points = scale * right_distal_points 
        right_proximal_points = scale * right_proximal_points

        return (head_rmat, unitree_left_wrist, unitree_right_wrist,
                unitree_left_hand, unitree_right_hand,
                right_distal_points, right_proximal_points)

    # ------------------------------------------------------------------
    # Quest3 Controller 모드 데이터 — clutch 등 다운스트림 처리 *전* 변환만 수행
    # ------------------------------------------------------------------
    def get_controller_data(self):
        """
        Vuer raw controller pose 를 Robot Convention(Z-up, x-front) 기저로
        similarity transform 한 head/left/right SE(3) 와 입력 상태(state) 를
        반환한다.

        - hand-tracking 과 다른 점: T_to_unitree_left/right_wrist 곱셈을 수행
          하지 않는다. xr_teleoperate 에 따르면 Vuer 가 컨트롤러 pose 를 이미
          Unitree URDF 와 같은 축 규약으로 보내준다.
        - clutch (grip 동안만 추종) 와 head→waist, head_pose 기반 상대 좌표
          처리는 본 함수에서 수행하지 않고 worker_g1_ik 에서 처리한다.

        반환:
            head_mat        (4,4) Robot Convention SE(3)
            left_ctrl_mat   (4,4)
            right_ctrl_mat  (4,4)
            left_state      (7,) [trigger, squeeze, tx, ty, a, b, thumb_click]
            right_state     (7,)
            connected       bool
        """
        head_vuer_mat, _ = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())
        left_vuer_mat, left_flag   = mat_update(const_left_wrist_vuer_mat,
                                                self.tv.left_ctrl_pose.copy())
        right_vuer_mat, right_flag = mat_update(const_right_wrist_vuer_mat,
                                                self.tv.right_ctrl_pose.copy())

        # 좌표 기저 변경 (OpenXR -> Robot)
        head_mat       = T_robot_openxr @ head_vuer_mat  @ fast_mat_inv(T_robot_openxr)
        left_ctrl_mat  = T_robot_openxr @ left_vuer_mat  @ fast_mat_inv(T_robot_openxr)
        right_ctrl_mat = T_robot_openxr @ right_vuer_mat @ fast_mat_inv(T_robot_openxr)
        # NOTE: T_to_unitree_left/right_wrist 는 곱하지 않는다.

        left_state  = self.tv.left_ctrl_state
        right_state = self.tv.right_ctrl_state
        connected   = self.tv.is_controller_connected

        # Phase K9 (P2-6) TODO: 향후 wrist pose (4x4 SE(3)) 를 parquet 의 state/action
        # 벡터에 저장하기 시작하면, utils/record_collectors.align_and_save_episode 에서
        # interp_to_axis 로 4x4 를 원소별 linear 보간하면 회전행렬의 직교성이 깨진다.
        # 그 경우 translation([:3,3]) 은 linear, rotation([:3,:3]) 은 quaternion SLERP
        # (utils.mat_tool._quat_slerp / se3_interp 활용 가능) 으로 분리해야 한다.
        # 현재는 wrist pose 가 SHM 에는 있지만 parquet 에 저장되지 않으므로 영향 없음.
        return head_mat, left_ctrl_mat, right_ctrl_mat, left_state, right_state, connected