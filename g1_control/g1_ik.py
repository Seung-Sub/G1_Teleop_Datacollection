import casadi                                                                       
# import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin    
# from pinocchio.visualize import MeshcatVisualizer   

import os
import sys

# parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(parent2_dir)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.weighted_moving_filter import WeightedMovingFilter

# from .g1_head_dynamixel import Dynamixel_Controller

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

TICKS_PER_REV    = 4096            # 0 ~ 4095 → 360°
TICKS_PER_RAD    = TICKS_PER_REV / (2 * np.pi)   # 라디안 당 틱 수
ZERO_OFFSET_TICK = TICKS_PER_REV // 2             # 180° 위치 = 2048


class G1_29_ArmIK:
    def __init__(self, AMO = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.AMO = AMO
        self.Visualization = Visualization

        if not self.AMO:
            self.robot = pin.RobotWrapper.BuildFromURDF('g1_control/assets/g1/g1_body29_inspire_zed.urdf', 'g1_control/assets/g1/')
        else:
            self.robot = pin.RobotWrapper.BuildFromURDF('g1_control/assets/g1/g1_body29_inspire_zed.urdf', 'g1_control/assets/g1/',pin.JointModelFreeFlyer())

        self.mixed_jointsToLockIDs = [
                                        "left_hip_pitch_joint" ,
                                        "left_hip_roll_joint" ,
                                        "left_hip_yaw_joint" ,
                                        "left_knee_joint" ,
                                        "left_ankle_pitch_joint" ,
                                        "left_ankle_roll_joint" ,
                                        "right_hip_pitch_joint" ,
                                        "right_hip_roll_joint" ,
                                        "right_hip_yaw_joint" ,
                                        "right_knee_joint" ,
                                        "right_ankle_pitch_joint" ,
                                        "right_ankle_roll_joint" ,


                                        "L_index_proximal_joint",
                                        "L_index_intermediate_joint",
                                        "L_middle_proximal_joint",
                                        "L_middle_intermediate_joint",
                                        "L_pinky_proximal_joint",
                                        "L_pinky_intermediate_joint",
                                        "L_ring_proximal_joint",
                                        "L_ring_intermediate_joint",
                                        "L_thumb_proximal_yaw_joint",
                                        "L_thumb_proximal_pitch_joint",
                                        "L_thumb_intermediate_joint",
                                        "L_thumb_distal_joint",


                                        "R_index_proximal_joint",
                                        "R_index_intermediate_joint",
                                        "R_middle_proximal_joint",
                                        "R_middle_intermediate_joint",
                                        "R_pinky_proximal_joint",
                                        "R_pinky_intermediate_joint",
                                        "R_ring_proximal_joint",
                                        "R_ring_intermediate_joint",
                                        "R_thumb_proximal_yaw_joint",
                                        "R_thumb_proximal_pitch_joint",
                                        "R_thumb_intermediate_joint",
                                        "R_thumb_distal_joint",
                                    ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

#--------------------------------------------------------------------#

        self.waist_joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        # 모델 안에서의 위치(0-based configuration 인덱스) 구해두기
        # self.waist_idx = [
        #     self.reduced_robot.model.getJointId(jn) - 1   # root_joint(0) 보정
        #     for jn in self.waist_joint_names
        # ]
        self.waist_idx = []
        for jn in self.waist_joint_names:
            jid = self.reduced_robot.model.getJointId(jn)
            j = self.reduced_robot.model.joints[jid]
            # 허리가 1-DoF라면 아래 range는 한 칸짜리
            self.waist_idx += list(range(j.idx_q, j.idx_q + j.nq))

#--------------------------------------------------------------------#


#--------------------------------sol_q-------------------------------#
# 1: root_joint                     0:7
# 2: waist_yaw_joint
# 3: waist_roll_joint
# 4: waist_pitch_joint              7:10
# 5: camera_yaw_joint
# 6: camera_pitch_joint             10:12
# 7: left_shoulder_pitch_joint      
# 8: left_shoulder_roll_joint
# 9: left_shoulder_yaw_joint
# 10: left_elbow_joint
# 11: left_wrist_roll_joint
# 12: left_wrist_pitch_joint
# 13: left_wrist_yaw_joint
# 14: right_shoulder_pitch_joint
# 15: right_shoulder_roll_joint
# 16: right_shoulder_yaw_joint
# 17: right_elbow_joint
# 18: right_wrist_roll_joint
# 19: right_wrist_pitch_joint
# 20: right_wrist_yaw_joint         12:26
#--------------------------------------------------------------------#



        self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        
        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addFrame(
            pin.Frame('head_target', 
                      self.reduced_robot.model.getJointId('camera_pitch_joint'),
                      pin.SE3(np.eye(3), 
                              np.array([0.04, 0, 0.0]).T), 
                        pin.FrameType.OP_FRAME)
        )

        self.reduced_robot.data = self.reduced_robot.model.createData()


        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     logger_mp.info(f"Frame ID: {frame_id}, Name: {frame.name}")
        # for idx, name in enumerate(self.reduced_robot.model.names):
        #     logger_mp.info(f"{idx}: {name}")
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        self.cTf_head = casadi.SX.sym("tf_head", 4, 4)

        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        self.head_id = self.reduced_robot.model.getFrameId("head_target")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r, self.cTf_head],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3],
                    self.cdata.oMf[self.head_id].translation - self.cTf_head[:3, 3]

                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r, self.cTf_head],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.head_id].rotation @ self.cTf_head[:3,:3].T)
                )
            ],
        )
        

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.param_tf_head = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r,self.param_tf_head))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r, self.param_tf_head))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)


        self.q_waist_ref = np.zeros(len(self.waist_idx))   # “움직이지 않았으면” 하는 기준
        self.waist_weight = 2.0                            # 클수록 허리 움직임 억제
        self.waist_cost = casadi.sumsqr(
            self.var_q[self.waist_idx] - self.q_waist_ref
        )

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost \
                        #    + 0.1 * self.smooth_cost )
                           + 0.1 * self.smooth_cost    + self.waist_weight * self.waist_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# logger_mp.info or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.reduced_robot.model.nq)
        
        
        self.vis = None


        # if self.Visualization:
        #     # Initialize the Meshcat visualizer for visualization
        #     self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        #     self.vis.initViewer(open=True) 
        #     self.vis.loadViewerModel("pinocchio") 
        #     self.vis.displayFrames(True, frame_ids=[107, 108], axis_length = 0.15, axis_width = 5)
        #     self.vis.display(pin.neutral(self.reduced_robot.model))

        #     # Enable the display of end effector target frames with short axis lengths and greater width.
        #     frame_viz_names = ['L_ee', 'R_ee', 'head_target']
        #     FRAME_AXIS_POSITIONS = (
        #         np.array([[0, 0, 0], [1, 0, 0],
        #                   [0, 0, 0], [0, 1, 0],
        #                   [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        #     )
        #     FRAME_AXIS_COLORS = (
        #         np.array([[1, 0, 0], [1, 0.6, 0],
        #                   [0, 1, 0], [0.6, 1, 0],
        #                   [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        #     )
        #     axis_length = 0.1
        #     axis_width = 20
        #     for frame_viz_name in frame_viz_names:
        #         self.vis.viewer[frame_viz_name].set_object(
        #             mg.LineSegments(
        #                 mg.PointsGeometry(
        #                     position=axis_length * FRAME_AXIS_POSITIONS,
        #                     color=FRAME_AXIS_COLORS,
        #                 ),
        #                 mg.LineBasicMaterial(
        #                     linewidth=axis_width,
        #                     vertexColors=True,
        #                 ),
        #             )
        #         )

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_left_pose, human_right_pose, human_head_pose, human_arm_length=0.60, robot_arm_length=0.75, human_upper_length=0.75, robot_upper_length=0.60):
        scale_factor = robot_arm_length / human_arm_length
        upper_factor = robot_upper_length / human_upper_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_head_pose = human_head_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        robot_head_pose[:3, 3] *= upper_factor
        return robot_left_pose, robot_right_pose, robot_head_pose

    def _extract_best_q(self):
        # solve/solve_limited 후에는 value()가, 완전 실패 시엔 debug.value()가 안전
        try:
            return np.array(self.opti.value(self.var_q)).reshape(-1)
        except Exception:
            return np.array(self.opti.debug.value(self.var_q)).reshape(-1)


    def solve_ik(self, left_wrist, right_wrist, head_pose, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # self.Visualization = False
        # if self.Visualization:
        #     self.vis.viewer['L_ee'].set_transform(left_wrist)   # for visualization
        #     self.vis.viewer['R_ee'].set_transform(right_wrist)  # for visualization
        #     self.vis.viewer['head_target'].set_transform(head_pose)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.param_tf_head, head_pose)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        # try:
            # sol = self.opti.solve()
            # # sol = self.opti.solve_limited()

        solved = False
        try:
            self.opti.solve()   # 먼저 정석 시도
            solved = True
        except Exception as e:
            logger_mp.info(f"solve() failed: {e}. Falling back to solve_limited().")
            try:
                self.opti.solve_limited()  # 마지막 iterate라도 확보
            except Exception as e2:
                logger_mp.info(f"solve_limited() also failed: {e2}. Using debug iterate.")


        sol_q = self._extract_best_q()
        # sol_q = self.opti.value(self.var_q)
        self.smooth_filter.add_data(sol_q)
        sol_q = self.smooth_filter.filtered_data

        # if self.Visualization:
        #     self.vis.display(sol_q)  # for visualization
        
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, sol_q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        # 3) Frame ID 취득
        pelvis_id = self.reduced_robot.model.getFrameId('pelvis')        # Frame ID 2
        torso_id  = self.reduced_robot.model.getFrameId('torso_link')    # Frame ID 36
        root_id  = self.reduced_robot.model.getFrameId('root_joint')    # Frame ID 36

        # 4) 세계좌표계에서의 SE3 변환 획득
        pelvis_se3 = self.reduced_robot.data.oMf[pelvis_id]
        torso_se3  = self.reduced_robot.data.oMf[torso_id]
        root_se3  = self.reduced_robot.data.oMf[root_id]

        # 5) 동차변환 행렬(4×4)과 분리된 위치·회전 정보
        pelvis_T  = pelvis_se3.homogeneous        # 4×4 numpy array
        pelvis_t  = pelvis_se3.translation        # (3,) 위치 벡터
        pelvis_R  = pelvis_se3.rotation           # (3×3) 회전 행렬

        torso_T   = torso_se3.homogeneous
        torso_t   = torso_se3.translation
        torso_R   = torso_se3.rotation

        pelvis_height = float(pelvis_t[2])  # numpy scalar -> python float
        q_xyzw = pin.Quaternion(torso_R)  
        torso_quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
                                dtype=np.float64)  # [w, x, y, z]

        if current_lr_arm_motor_dq is not None:
            v = current_lr_arm_motor_dq * 0.0
        else:
            v = (sol_q - self.init_data) * 0.0

        self.init_data = sol_q
        v=np.zeros(self.reduced_robot.model.nv)


        sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))
        

        # stats = self.opti.stats() if solved else {}
        # logger_mp.info(f"IK status: solved={solved}, stats={stats.get('return_status','n/a')}")

        return sol_q, sol_tauff, pelvis_height, torso_quat_wxyz

        
    def get_dual_wrist_pose(self, current_lr_arm_motor_q):
        
        """
            현재 로봇 상태에서 오른쪽 손목 끝단(R_ee)의 pose(SE3)를 구합니다.
            :param current_lr_arm_motor_q: 현재 reduced robot의 관절 위치 (np.array)
            :return: pin.SE3 형식의 오른쪽 손목 pose
        """
        assert current_lr_arm_motor_q.shape[0] == self.reduced_robot.model.nq, \
            "Input joint configuration shape mismatch."

        # Forward kinematics 계산
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, current_lr_arm_motor_q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        right_arm_id = self.reduced_robot.model.getFrameId('R_ee')        # Frame ID 2
        right_arm_se3  = self.reduced_robot.data.oMf[right_arm_id]
        right_arm_T = right_arm_se3.homogeneous  

        left_arm_id = self.reduced_robot.model.getFrameId('L_ee')        # Frame ID 2
        left_arm_se3  = self.reduced_robot.data.oMf[left_arm_id]
        left_arm_T = left_arm_se3.homogeneous         
        
        
        return right_arm_T, left_arm_T
    


    def init_pose(self):
        q_zero = np.zeros(self.reduced_robot.model.nq)


        q_zero = np.array([0,0,0,0,0,0,0.3491,0,-0.1745,0,0,0,0,-0.3491,0,-0.1745,0,0,0])

        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_zero)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        T_L_ee      = self.reduced_robot.data.oMf[self.L_hand_id].homogeneous
        T_R_ee      = self.reduced_robot.data.oMf[self.R_hand_id].homogeneous
        T_head_tgt  = self.reduced_robot.data.oMf[self.head_id].homogeneous


        return T_L_ee, T_R_ee,T_head_tgt
    
    def get_torso_quat(self, current_lr_arm_motor_q: np.ndarray, as_wxyz: bool = False):
        """
        현재 reduced 로봇 상태에서 torso_link의 orientation을 world 기준 쿼터니언으로 반환.

        Args:
            current_lr_arm_motor_q (np.ndarray): (nq,) 형태의 관절각 벡터 (reduced_robot 기준)
            as_wxyz (bool): True면 [w, x, y, z] numpy 배열을 반환, False면 pin.Quaternion 반환

        Returns:
            pin.Quaternion 또는 np.ndarray([w, x, y, z])

        Raises:
            AssertionError: 입력 크기가 모델 nq와 다를 때
        """
        assert current_lr_arm_motor_q.shape[0] == self.reduced_robot.model.nq, \
            f"q size mismatch: expected {self.reduced_robot.model.nq}, got {current_lr_arm_motor_q.shape[0]}"

        # FK & frame 업데이트
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, current_lr_arm_motor_q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        # torso 링크의 회전행렬 -> 쿼터니언
        torso_id = self.reduced_robot.model.getFrameId('torso_link')
        torso_R  = self.reduced_robot.data.oMf[torso_id].rotation
        q_pin    = pin.Quaternion(torso_R)  # Pinocchio는 내부 계수 순서가 [x, y, z, w]

        if as_wxyz:
            coeffs_xyzw = q_pin.coeffs()  # [x, y, z, w]
            q_wxyz = np.array([coeffs_xyzw[3], coeffs_xyzw[0], coeffs_xyzw[1], coeffs_xyzw[2]])
            return q_wxyz
        else:
            return q_pin


    def pose_quaternion_to_homogeneous_np(self,position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        position: (3,) 배열, [x, y, z]
        quaternion: (4,) 배열, [w, x, y, z] 순서 가정
        반환: (4,4) 동차변환 행렬
        """
        w, x, y, z = quaternion

        # 회전 행렬 계산
        R = np.zeros((3, 3), dtype=position.dtype)
        R[0, 0] = 1 - 2*(y*y + z*z)
        R[0, 1] = 2*(x*y - z*w)
        R[0, 2] = 2*(x*z + y*w)
        R[1, 0] = 2*(x*y + z*w)
        R[1, 1] = 1 - 2*(x*x + z*z)
        R[1, 2] = 2*(y*z - x*w)
        R[2, 0] = 2*(x*z - y*w)
        R[2, 1] = 2*(y*z + x*w)
        R[2, 2] = 1 - 2*(x*x + y*y)

        # 동차변환 행렬 조립
        T = np.eye(4, dtype=position.dtype)
        T[:3, :3] = R
        T[:3, 3]   = position
        return T
    
    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions q1 and q2.
        Args:
            q1: First quaternion as [w, x, y, z]
            q2: Second quaternion as [w, x, y, z]
        Returns:
            Resulting quaternion as [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])


# ---- add below this line -----------------------------------------------------

def _pretty_T(name, T):
    import numpy as np
    import pinocchio as pin
    np.set_printoptions(precision=5, suppress=True, linewidth=200)
    print(f"\n[{name}] 4x4 homogeneous:")
    print(T)
    pos = T[:3, 3]
    # pinocchio Quaternion.coeffs() returns [x, y, z, w]
    q_xyzw = pin.Quaternion(T[:3, :3]).coeffs()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    print(f"{name} position [m]: {pos}")
    print(f"{name} quaternion [w, x, y, z]: {q_wxyz}")

if __name__ == "__main__":
    import os
    import numpy as np

    # (선택) 이 파일 기준으로 프로젝트 루트로 CWD 이동
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"[info] cwd -> {project_root}")

    # 인스턴스 만들고 init_pose 호출
    ik = G1_29_ArmIK(AMO=False)   # free-flyer 필요하면 True
    T_L_ee, T_R_ee, T_head_tgt = ik.init_pose()

    # 보기 좋게 출력
    _pretty_T("L_ee", T_L_ee)
    _pretty_T("R_ee", T_R_ee)
    _pretty_T("head_target", T_head_tgt)

    # 필요 시 넘파이 배열로 저장 예시
    # np.savez("init_pose.npz", T_L_ee=T_L_ee, T_R_ee=T_R_ee, T_head_tgt=T_head_tgt)
    print("\n[done] init_pose retrieved.")



