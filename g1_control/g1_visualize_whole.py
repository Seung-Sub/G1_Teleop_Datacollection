import casadi                                                                       
# import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
from pinocchio import casadi as cpin                
from pinocchio.robot_wrapper import RobotWrapper    
# from pinocchio.visualize import MeshcatVisualizer 
import os
import sys  

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)


class G1_Visualization:
    def __init__(self):#, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.robot = pin.RobotWrapper.BuildFromURDF('g1_control/assets/g1/g1_body29_inspire_zed.urdf', 'g1_control/assets/g1/',pin.JointModelFreeFlyer())

        self.mixed_jointsToLockIDs = [
                                        
                                    ]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        
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


        # Creating Casadi models and data for symbolic computing

        self.vis = None

        # for idx, name in enumerate(self.reduced_robot.model.names):
        #     logger_mp.info(f"{idx}: {name}")
      
        # Initialize the Meshcat visualizer for visualization
        # self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        logger_mp.info("open !!")
        # self.vis.initViewer(open=False) 
        # self.vis.loadViewerModel("pinocchio") 
        # self.vis.displayFrames(True, frame_ids=[107, 108], axis_length = 0.15, axis_width = 5)
        # self.vis.display(pin.neutral(self.reduced_robot.model))
        self._q = pin.neutral(self.reduced_robot.model).copy()  # 부분 갱신용 상태 캐시




    def _add_axis(self, name: str):
        """MeshcatViewer 에 `name` 노드로 3축을 만든다(없을 때만)."""

        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                        [0, 0, 0], [0, 1, 0],
                        [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                        [0, 1, 0], [0.6, 1, 0],
                        [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 20

        # self.vis.viewer[name].set_object(
        #     mg.LineSegments(
        #         mg.PointsGeometry(
        #             position=axis_length * FRAME_AXIS_POSITIONS,
        #             color=FRAME_AXIS_COLORS,
        #         ),
        #         mg.LineBasicMaterial(
        #             linewidth=axis_width,
        #             vertexColors=True,
        #         ),
        #     )
        # )

        # self._axes.add(name)

    def visual_target_frame(self, target_name : str , pose):
        self._add_axis(target_name)          # 자동 생성

        # self.vis.viewer[target_name].set_transform(pose)  # for visualization

    def _joint_slice(self, joint_name: str) -> slice:
        m = self.reduced_robot.model
        jid = m.getJointId(joint_name)
        if jid == 0:
            raise ValueError(f"Unknown joint name: {joint_name}")
        start = getattr(m.joints[jid], "idx_q", None)
        if start is None:
            start = m.idx_qs[jid]
        nqj = m.joints[jid].nq
        return slice(start, start + nqj)

    # 여러 조인트 슬라이스에 values를 순서대로 채워넣기
    def _apply_by_joint_names(self, joint_names, values):
        values = np.asarray(values, dtype=float).ravel()
        expected = 0
        slices = []
        for jn in joint_names:
            sl = self._joint_slice(jn)
            slices.append(sl)
            expected += (sl.stop - sl.start)
        if len(values) != expected:
            raise ValueError(f"Expected {expected} values for {joint_names}, got {len(values)}")

        q = self._q.copy()
        idx = 0
        for sl in slices:
            w = sl.stop - sl.start
            q[sl] = values[idx:idx+w]
            idx += w
        self._q = q



    def visual_waist_q(self, waist=None, right=None, display=True):
        if (waist is None) :
            return
        if waist is not None:
            self._apply_by_joint_names(
                ["waist_yaw_joint","waist_roll_joint","waist_pitch_joint"],
                waist
            )
        # if display:
        #     self.vis.display(self._q)

    # 다리: 각각 6개 값 [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
    # left, right는 선택적으로 전달(둘 중 하나만 전달해도 됨)
    def visual_leg_q(self, left=None, right=None, display=True):
        if (left is None) and (right is None):
            return
        if left is not None:
            self._apply_by_joint_names(
                ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
                 "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint"],
                left
            )
        if right is not None:
            self._apply_by_joint_names(
                ["right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
                 "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"],
                right
            )
        # if display:
        #     self.vis.display(self._q)

    # 팔: 각각 7개 값 [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
    def visual_arm_q(self, left=None, right=None, display=True):
        if (left is None) and (right is None):
            return
        if left is not None:
            self._apply_by_joint_names(
                ["left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
                 "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"],
                left
            )
        if right is not None:
            self._apply_by_joint_names(
                ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                 "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"],
                right
            )
        # if display:
        #     self.vis.display(self._q)

    # 손: 각각 12개 값
    # [index_proximal, index_intermediate,
    #  middle_proximal, middle_intermediate,
    #  pinky_proximal,  pinky_intermediate,
    #  ring_proximal,   ring_intermediate,
    #  thumb_prox_yaw,  thumb_prox_pitch,
    #  thumb_intermediate, thumb_distal]
    def visual_hand_q(self, left=None, right=None, display=True):
        if (left is None) and (right is None):
            return
        if left is not None:
            self._apply_by_joint_names(
                ["L_index_proximal_joint","L_index_intermediate_joint",
                 "L_middle_proximal_joint","L_middle_intermediate_joint",
                 "L_pinky_proximal_joint","L_pinky_intermediate_joint",
                 "L_ring_proximal_joint","L_ring_intermediate_joint",
                 "L_thumb_proximal_yaw_joint","L_thumb_proximal_pitch_joint",
                 "L_thumb_intermediate_joint","L_thumb_distal_joint"],
                left
            )
        if right is not None:
            self._apply_by_joint_names(
                ["R_index_proximal_joint","R_index_intermediate_joint",
                 "R_middle_proximal_joint","R_middle_intermediate_joint",
                 "R_pinky_proximal_joint","R_pinky_intermediate_joint",
                 "R_ring_proximal_joint","R_ring_intermediate_joint",
                 "R_thumb_proximal_yaw_joint","R_thumb_proximal_pitch_joint",
                 "R_thumb_intermediate_joint","R_thumb_distal_joint"],
                right
            )
        # if display:
        #     self.vis.display(self._q)

    # 머리: 2개 값 [camera_yaw, camera_pitch]
    def visual_head_q(self, head, display=True):
        self._apply_by_joint_names(["camera_yaw_joint","camera_pitch_joint"], head)
        # if display:
        #     self.vis.display(self._q)

    def visual_base_q(self, base, normalize_quaternion: bool = True, display: bool = True):
        vals = np.asarray(base, dtype=float).ravel()
        if len(vals) != 7:
            raise ValueError(f"base expects 7 values [x,y,z,qx,qy,qz,qw], got {len(vals)}")

        if normalize_quaternion:
            # 쿼터니언 정규화 (0 division 방지)
            q = vals[3:7].copy()
            n = np.linalg.norm(q)
            if n > 0:
                q /= n
            else:
                q = np.array([0., 0., 0., 1.], dtype=float)
            vals = np.concatenate([vals[:3], q])

        # root_freeflyer 조인트 이름은 사용하시는 URDF에서 'root_joint' (질문 기준)
        self._apply_by_joint_names(["root_joint"], vals)

        # if display:
        #     self.vis.display(self._q)

    def visualize_traj(self, vis, traj, name="traj_path",
                    axis_length=0.01, axis_width=4, line_width=2.0):
        """
        vis   : MeshcatVisualizer 객체 (visual.vis)
        traj  : list[np.ndarray] – 4×4 SE(3) 행렬들
        name  : Meshcat 노드 이름
        """
        # ① 기존 노드 삭제
        try:
            vis.viewer[name].delete()
        except KeyError:
            pass   
        # ② (N,3) 위치 배열
        positions = np.array([T[:3, 3] for T in traj], dtype=np.float32)

        # ③ 파란색 → 빨간색 그라디언트
        n = len(positions)
        colors = np.stack(
            [np.linspace(0, 1, n),     # R
            np.zeros(n),              # G
            np.linspace(1, 0, n)],    # B
            axis=1, dtype=np.float32)

        # ④ LineSegments용 점·색 배열
        seg_pos = np.empty((2*(n-1), 3), dtype=np.float32)
        seg_col = np.empty((2*(n-1), 3), dtype=np.float32)
        seg_pos[0::2], seg_pos[1::2] = positions[:-1], positions[1:]
        seg_col[0::2], seg_col[1::2] = colors[:-1], colors[1:]

        vis.viewer[name].set_object(
            mg.LineSegments(
                mg.PointsGeometry(position=seg_pos.T, color=seg_col.T),
                mg.LineBasicMaterial(linewidth=line_width, vertexColors=True)
            )
        )

        # ⑤ 10프레임마다 좌표축 표시(선택)
        for i, T in enumerate(traj[::10]):
            frame_name = f"{name}_frame_{i}"
            vis.viewer[frame_name].set_transform(T)
            vis.viewer[frame_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * np.array(
                            [[0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1]],
                            dtype=np.float32).T,
                        color=np.array(
                            [[1,0,0],[1,0.6,0],[0,1,0],[0.6,1,0],[0,0,1],[0,0.6,1]],
                            dtype=np.float32).T
                    ),
                    mg.LineBasicMaterial(linewidth=axis_width, vertexColors=True)
                )
            )



import time, math

def wave(t, base, amp, freq, phase=0.0):
    """시간 t에서 base를 중심으로 amp 진폭, freq(Hz) 주파수, phase 위상의 사인파."""
    return base + amp * math.sin(2 * math.pi * freq * t + phase)

if __name__ == "__main__":
    visual = G1_Visualization()

    user_input = input("enter 's' to start:\n")
    if user_input.lower() == 's':
        try:
            t0 = time.time()
            while True:
                t = time.time() - t0

                # 다리(각 6개) – 좌우 반대 위상으로 보행 느낌
                left_leg = [
                    wave(t, -0.10, 0.10, 0.25, 0.0),              # hip_roll
                    wave(t,  0.00, 0.10, 0.20, math.pi/2),        # hip_yaw
                    wave(t,  0.00, 0.30, 0.25, math.pi),          # hip_pitch
                    wave(t,  0.80, 0.30, 0.25, math.pi),          # knee_pitch
                    wave(t, -0.20, 0.15, 0.25, 0.0),              # ankle_pitch
                    wave(t,  0.00, 0.10, 0.35, math.pi/2),        # ankle_roll
                ]
                right_leg = [
                    wave(t, -0.10, 0.10, 0.25, math.pi),          # hip_roll (mirror)
                    wave(t,  0.00, 0.10, 0.20, math.pi/2+math.pi), # hip_yaw
                    wave(t,  0.00, 0.30, 0.25, 0.0),              # hip_pitch
                    wave(t,  0.80, 0.30, 0.25, 0.0),              # knee_pitch
                    wave(t, -0.20, 0.15, 0.25, math.pi),          # ankle_pitch
                    wave(t,  0.00, 0.10, 0.35, math.pi/2+math.pi),# ankle_roll
                ]
                visual.visual_leg_q(left=left_leg, right=right_leg)

                # 팔(각 7개) – 서로 다른 주파수/위상으로 자연스러운 스윙
                left_arm = [
                    wave(t,  0.30, 0.20, 0.33, 0.0),
                    wave(t, -0.10, 0.20, 0.33, math.pi/2),
                    wave(t,  0.20, 0.20, 0.33, math.pi),
                    wave(t, -0.80, 0.30, 0.33, math.pi),
                    wave(t,  0.20, 0.20, 0.50, 0.0),
                    wave(t, -0.10, 0.20, 0.50, math.pi/2),
                    wave(t,  0.10, 0.20, 0.50, math.pi),
                ]
                right_arm = [
                    wave(t, -0.20, 0.20, 0.33, math.pi),
                    wave(t,  0.15, 0.20, 0.33, math.pi/2+math.pi),
                    wave(t,  0.00, 0.20, 0.33, 0.0),
                    wave(t, -1.10, 0.30, 0.33, 0.0),
                    wave(t, -0.10, 0.20, 0.50, math.pi),
                    wave(t,  0.05, 0.20, 0.50, math.pi/2+math.pi),
                    wave(t, -0.05, 0.20, 0.50, 0.0),
                ]
                visual.visual_arm_q(left=left_arm, right=right_arm)

                # 손(각 12개) – 손가락 파형; 좌/우 반대 위상으로 쥐었다 폈다
                def finger_block(base_phase):
                    # 0.0~0.6 범위로 흔들림
                    return [wave(t, 0.30, 0.30, 0.80, base_phase + i * 0.4) for i in range(12)]
                right_hand = finger_block(0.0)
                left_hand  = finger_block(math.pi)
                visual.visual_hand_q(right=right_hand, left=left_hand)

                # 머리(2개: yaw, pitch)
                head = [
                    wave(t,  0.00, 0.40, 0.10, 0.0),              # yaw
                    wave(t, -0.05, 0.20, 0.15, math.pi/2),        # pitch
                ]
                visual.visual_head_q(head)

                # 베이스(x,y,z,qx,qy,qz,qw) – 필요시만 움직이세요(여긴 정지)
                visual.visual_base_q([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

                # 너무 빠른 업데이트 방지
                time.sleep(0.02)  # ≈50 Hz

        except KeyboardInterrupt:
            logger_mp.info("Interrupted by user.")
        finally:
            logger_mp.info("종료합니다.")
