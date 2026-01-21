import numpy as np
from .geometry_utils_np import quat_to_rotmat, transform_vector, cos2ndLaw_angle, cos2ndLaw_length


def transform_goalpoint_wrist2world_np(
    point_goal: np.ndarray,   # (3,) goal in wrist frame
    wrist_quat: np.ndarray,   # (4,) wrist orientation (w, x, y, z)
    wrist_pos: np.ndarray,    # (3,) wrist position in world
    ratio: float = 1.0,
) -> np.ndarray:
    """
    Scale and transform a point from wrist frame to world frame.

    Returns:
        point_world : (3,)
    """
    point_wrist = point_goal.astype(np.float32) * ratio          # (3,)
    R = quat_to_rotmat(wrist_quat)                      # (3, 3)

    point_world = R @ point_wrist + wrist_pos.astype(np.float32)  # (3,)
    return point_world


def IK_Finger_np(
    finger_type: str,
    contact_point: np.ndarray,  # shape (3,) : target contact point in world frame
    hand_quat: np.ndarray,      # shape (4,) : [w, x, y, z]
    hand_pos : np.ndarray       # shape (3,) : hand center in world frame
) -> np.ndarray:
    """
    Computes inverse kinematics (IK) for a specified finger ("Index", "Middle", or "Ring") for a single sample.
    Returns an array of shape (4,) containing the four finger joint angles in radians.
    """

    # 1) Finger link lengths (constant)
    link_0 = 0.0095
    link_1 = 0.0401
    link_2 = 0.0298
    link_3 = 0.0250

    # 2) Finger base offsets and deviations (in degrees)
    finger_bases_np = {
        "Index" : np.array([-0.03234003159, -0.0099, 0.00318800373], dtype=np.float64),
        "Middle": np.array([ 0.0,           -0.0099, 0.0046       ], dtype=np.float64),
        "Ring"  : np.array([ 0.03234003159, -0.0099, 0.00318800373], dtype=np.float64)
    }
    finger_deviations_np = {
        "Index" : -5.0,
        "Middle":  0.0,
        "Ring"  :  5.0
    }

    # 3) Select base and deviation based on finger type
    Finger_base = finger_bases_np[finger_type]
    finger_deviation_deg = finger_deviations_np[finger_type]

    # 4) Compute hand rotation matrix and local axes
    R_hand = quat_to_rotmat(hand_quat)  # 3x3
    hand_X_direction = R_hand[:, 0]
    hand_Y_direction = R_hand[:, 1]
    hand_Z_direction = R_hand[:, 2]

    # 5) Compute finger base position in world frame
    finger_base_pose = (hand_pos
                        + Finger_base[0]*hand_X_direction
                        + Finger_base[1]*hand_Y_direction
                        + Finger_base[2]*hand_Z_direction)

    # Deviation angle (radians)
    dev_rad = np.radians(finger_deviation_deg)
    sin_dev = np.sin(dev_rad)
    cos_dev = np.cos(dev_rad)

    # 6) Define finger's local z-axis with deviation
    finger_z_axis = sin_dev * hand_X_direction + cos_dev * hand_Z_direction
    finger_z_axis /= (np.linalg.norm(finger_z_axis) + 1e-8)
    finger_y_axis = hand_Y_direction
    finger_x_axis = np.cross(finger_y_axis, finger_z_axis)

    # 7) Transform the target position into finger's frame (minus link_0 offset in local Z)
    target_posi = contact_point - finger_base_pose
    target_posi_fc = transform_vector(target_posi, finger_x_axis, finger_y_axis, finger_z_axis)

    # Subtract link_0 along local Z up (0,0,1 in finger frame).
    upright_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    target_posi_fc -= link_0 * upright_local

    # 8) Prepare a container for the final 4 joint angles (degrees internally, then convert to rad)
    joint_angles_deg = np.zeros(4, dtype=np.float64)

    # joint_0: rotation around local Z, in degrees
    # The original code: joint_0 = atan2(-x, y)
    # (Careful with indexing: x->0, y->1, z->2 in target_posi_fc)
    angle_0 = np.arctan2(-target_posi_fc[0], target_posi_fc[1]) * (180.0 / np.pi)
    joint_angles_deg[0] = angle_0

    # 9) Clamp final finger tip distance for stage logic
    max_length = link_1 + link_2 + link_3
    min_length = np.sqrt(link_2**2 + (link_1 - link_3)**2)

    target_distance = np.linalg.norm(target_posi_fc)
    Tip_distance = np.clip(target_distance, min_length, max_length)

    # joint_3: a bending angle that linearly goes from 90 deg (when very short) down to 0 deg (when very long)
    # "finger tip" approach from the original code
    if target_distance > max_length:
        # Over-extended => 0 deg
        joint_angles_deg[3] = 0.0
    elif target_distance < min_length:
        # Too close => 90 deg
        joint_angles_deg[3] = 90.0
    else:
        # Interpolate between 90 deg (min_length) and 0 deg (max_length)
        ratio = (max_length - target_distance) / (max_length - min_length)
        joint_angles_deg[3] = 90.0 * ratio

    # 10) Solve for link_x (virtual side) with the updated joint_3
    joint_3_rad = np.radians(180.0 - joint_angles_deg[3])  # original code used (180 - joint3)
    link_x = cos2ndLaw_length(link_2, link_3, joint_3_rad)

    # theta_intm = cos^-1(...) for angle between link_1 and link_x with tip distance
    theta_intm_rad = cos2ndLaw_angle(link_1, link_x, Tip_distance)
    # theta_sub = angle between link_2, link_x, link_3
    theta_sub_rad = cos2ndLaw_angle(link_2, link_x, link_3)

    # joint_2 = 180 - (theta_intm + theta_sub) in degrees
    joint_angles_deg[2] = 180.0 - np.degrees(theta_intm_rad + theta_sub_rad)

    # 11) joint_1: a combination of the vertical angle and geometry
    theta_goal_dir_rad = np.arctan2(target_posi_fc[2],
                                    np.sqrt(target_posi_fc[0]**2 + target_posi_fc[1]**2))
    theta_goal_sub_rad = cos2ndLaw_angle(link_1, Tip_distance, link_x)

    # joint_1 = 90 - (theta_goal_dir + theta_goal_sub), in degrees
    joint_angles_deg[1] = 90.0 - np.degrees(theta_goal_dir_rad + theta_goal_sub_rad)

    # 12) Convert everything to radians for final output
    joint_angles_rad = np.radians(joint_angles_deg)
    return joint_angles_rad


def IK_Thumb_np(
    contact_point: np.ndarray,  # shape (3,) : target contact position in world frame
    hand_quat: np.ndarray,      # shape (4,) : [w, x, y, z] quaternion of the hand
    hand_pos: np.ndarray        # shape (3,) : hand center position in world frame
) -> np.ndarray:
    """
    Single-sample version of the thumb inverse kinematics using NumPy.
    
    Returns thumb joint angles as shape (4,) in radians.
    The logic follows the same steps and naming as the original PyTorch batch code.
    """
    # Finger link lengths
    link_1 = 0.0466
    link_2 = 0.0401
    link_3 = 0.0270

    # Finger deviation in radians
    finger_deviation_deg = -5.0
    finger_deviation = np.radians(finger_deviation_deg)

    # Finger base offset relative to the hand center
    Finger_base = np.array([-0.01044496692, -0.02255, -0.05777929249], dtype=np.float64)

    # Convert hand quaternion to rotation matrix
    hand_rot_mat = quat_to_rotmat(hand_quat)  # (3,3)

    # Extract the hand frame directions
    hand_X_direction = hand_rot_mat[:, 0]  # (3,)
    hand_Y_direction = hand_rot_mat[:, 1]  # (3,)
    hand_Z_direction = hand_rot_mat[:, 2]  # (3,)

    # Compute the finger base position in world frame
    finger_base_pose = (hand_pos
                        + Finger_base[0] * hand_X_direction
                        + Finger_base[1] * hand_Y_direction
                        + Finger_base[2] * hand_Z_direction)

    # Define the finger z-axis with the specified deviation around X
    finger_z_axis = (np.sin(finger_deviation) * hand_X_direction
                     + np.cos(finger_deviation) * hand_Z_direction)
    finger_z_axis /= (np.linalg.norm(finger_z_axis) + 1e-8)

    finger_y_axis = hand_Y_direction
    finger_x_axis = np.cross(finger_y_axis, finger_z_axis)  # X = Y x Z

    # Vector from finger base to target contact
    target_posi = contact_point - finger_base_pose

    # Recompute x-axis from the updated finger axes
    x_axis_b = np.cross(finger_y_axis, finger_z_axis)
    x_axis_b /= (np.linalg.norm(x_axis_b) + 1e-8)

    # Transform this vector into the local finger coordinate frame
    target_posi_fc = transform_vector(target_posi, x_axis_b, finger_y_axis, finger_z_axis)
    # Clamp the z-axis component to be non-negative
    target_posi_fc[2] = max(target_posi_fc[2], 0.0)

    # The heading is from the hand center (some logic in original code).
    # For single environment, it was basically the same as 'hand_pos' direction, but let's keep it consistent:
    target_heading = hand_pos.copy()
    norm_heading = np.linalg.norm(target_heading) + 1e-8
    target_heading_fc = transform_vector(target_heading / norm_heading, x_axis_b, finger_y_axis, finger_z_axis)

    # --- Stage 1 ---
    eps = 1e-8
    heading_z = target_heading_fc[2]
    if abs(heading_z) < eps:
        heading_z = eps

    # Project onto base plane
    on_base_plane_x = target_posi_fc[0] - (target_posi_fc[2] / heading_z) * target_heading_fc[0]
    on_base_plane_y = target_posi_fc[1] - (target_posi_fc[2] / heading_z) * target_heading_fc[1]

    # theta_0_s1
    theta_0_s1 = np.arctan2(on_base_plane_y, -on_base_plane_x)

    # j1_x_ndir, j1_z_dir, j1_y_dir (local directions)
    j1_x_ndir = np.array([
        -np.cos(theta_0_s1),
         np.sin(theta_0_s1),
         0.0
    ])
    j1_z_dir = np.array([0.0, 0.0, 1.0])
    j1_y_dir = np.array([
         np.sin(theta_0_s1),
         np.cos(theta_0_s1),
         0.0
    ])

    # heading-based angle
    dot_y = np.dot(target_heading_fc, j1_y_dir)
    dot_z = np.dot(target_heading_fc, j1_z_dir)
    theta_1_s1 = np.arctan2(dot_y, dot_z)

    # Adjust range
    if theta_1_s1 > 0.5 * np.pi:
        theta_1_s1 -= np.pi
    elif theta_1_s1 < -0.5 * np.pi:
        theta_1_s1 += np.pi

    # link_x
    link_x_vect = target_posi_fc - link_1 * j1_x_ndir
    link_x = np.linalg.norm(link_x_vect)

    # theta_3_s1
    theta_3_s1 = np.pi - cos2ndLaw_angle(link_2, link_3, link_x)

    # theta_2_s1
    dot_x = np.dot(link_x_vect, j1_x_ndir)
    safe_val = dot_x / (link_x + eps)
    safe_val = np.clip(safe_val, -1.0, 1.0)
    theta_2_intm_s1 = np.arccos(safe_val)
    theta_2_sub_s1 = cos2ndLaw_angle(link_x, link_2, link_3)
    theta_2_s1 = theta_2_intm_s1 - theta_2_sub_s1

    # Stage 2 logic: check length constraints
    max_length = link_2 + link_3
    min_length = np.sqrt(link_2**2 + link_3**2)

    link_x_s2 = link_x
    theta_0_s2 = theta_0_s1
    theta_1_s2 = theta_1_s1
    theta_2_s2 = theta_2_s1
    theta_3_s2 = theta_3_s1

    if link_x_s2 > max_length:
        # If too long, bend fully?
        theta_3_s2 = 0.0
        link_x_s2 = max_length
    elif link_x_s2 < min_length:
        # If too short, set to some limit
        theta_3_s2 = np.radians(90.0)  # forced angle
        link_x_s2 = min_length

    # Additional geometry to refine angles
    # This is the same approach as the batch code but done step by step.
    Target_base_plane = np.array([target_posi_fc[0], target_posi_fc[1], 0.0])
    joint_0_to_tip_base_length = np.linalg.norm(Target_base_plane)
    max_possible_length = (link_1 + link_2 + link_3)
    if joint_0_to_tip_base_length > max_possible_length:
        joint_0_to_tip_base_length = max_possible_length

    tmp1 = (joint_0_to_tip_base_length - link_1)**2
    tmp2 = link_x_s2**2 - target_posi_fc[2]**2
    joint_1_to_tip_plane_length_sq = max(tmp1, tmp2)
    joint_1_to_tip_plane_length = np.sqrt(max(joint_1_to_tip_plane_length_sq, 0.0))

    angle_ = cos2ndLaw_angle(joint_0_to_tip_base_length, link_1, joint_1_to_tip_plane_length)
    joint_1_y_dir_length = joint_0_to_tip_base_length * np.sin(angle_)

    # Direction check (2D cross-like check)
    norm_tp = np.linalg.norm(target_posi_fc) + eps
    target_direction = target_posi_fc / norm_tp
    direction_check = np.array([target_direction[1], -target_direction[0], 0.0])
    check_val = np.dot(target_heading_fc, direction_check)
    positive = (check_val > 0.0)

    denom = np.sqrt(max(link_x_s2**2 - joint_1_to_tip_plane_length_sq, 0.0)) + eps
    tmp_theta_1_s2 = np.arctan2(joint_1_y_dir_length, denom)
    if positive:
        # same sign
        new_theta_1_s2 = tmp_theta_1_s2
    else:
        # flip sign
        new_theta_1_s2 = -tmp_theta_1_s2
    
    # If we changed length constraints in stage2, update angles
    # (the batch version used masks; single version we do if-check)
    if not (min_length <= link_x <= max_length):
        theta_1_s2 = new_theta_1_s2

    theta_0_s2_intm = np.arctan2(target_posi_fc[1], -target_posi_fc[0])
    theta_0_s2_sub = cos2ndLaw_angle(link_1, joint_0_to_tip_base_length, joint_1_to_tip_plane_length)
    if positive:
        tmp_theta_0_s2 = theta_0_s2_intm - theta_0_s2_sub
    else:
        tmp_theta_0_s2 = theta_0_s2_intm + theta_0_s2_sub

    if not (min_length <= link_x <= max_length):
        theta_0_s2 = tmp_theta_0_s2

    # Recompute local directions
    j1p_x = -link_1 * np.cos(theta_0_s2)
    j1p_y =  link_1 * np.sin(theta_0_s2)
    j1p = np.array([j1p_x, j1p_y, 0.0])
    j1p_norm = np.linalg.norm(j1p) + eps
    joint_1_dir = j1p / j1p_norm

    joint_1_to_tip = target_posi_fc - j1p
    j1t_norm = np.linalg.norm(joint_1_to_tip) + eps
    joint_1_to_tip_dir = joint_1_to_tip / j1t_norm

    dot_j1 = np.dot(joint_1_dir, joint_1_to_tip_dir)
    dot_j1 = np.clip(dot_j1, -1.0, 1.0)
    theta_2_s2_intm = np.pi - np.arccos(dot_j1)
    theta_2_s2_sub = cos2ndLaw_angle(link_x_s2, link_2, link_3)
    tmp_theta_2_s2 = np.pi - (theta_2_s2_intm + theta_2_s2_sub)
    if not (min_length <= link_x <= max_length):
        theta_2_s2 = tmp_theta_2_s2

    # Stage 3 limit checks
    deg125 = np.radians(125.0)
    if theta_0_s2 < 0.0 or theta_0_s2 > deg125:
        # We clamp to [0, 125 deg], then re-solve link geometry
        if theta_0_s2 < 0.0:
            theta_0_s3 = 0.0
        else:
            theta_0_s3 = deg125

        j1p_x_s3 = -link_1 * np.cos(theta_0_s3)
        j1p_y_s3 =  link_1 * np.sin(theta_0_s3)
        joint_1_posi_s3 = np.array([j1p_x_s3, j1p_y_s3, 0.0])

        link_x_vect_s3 = target_posi_fc - joint_1_posi_s3
        link_x_s3 = np.linalg.norm(link_x_vect_s3)
        link_x_s3 = np.clip(link_x_s3, min_length, max_length)

        joint_1_x_ndir = np.array([
            -np.cos(theta_0_s3),
             np.sin(theta_0_s3),
             0.0
        ])
        joint_1_z_dir = np.array([0.0, 0.0, 1.0])
        joint_1_y_dir = np.array([
             np.sin(theta_0_s3),
             np.cos(theta_0_s3),
             0.0
        ])

        # Recompute geometry
        # 1) Base plane
        joint_1_to_tip_base = np.array([link_x_vect_s3[0], link_x_vect_s3[1], 0.0])
        base_norm = np.linalg.norm(joint_1_to_tip_base)
        joint_1_to_tip_base_max = min(link_x_s3, base_norm)
        if base_norm > 1e-8:
            dir_ = joint_1_to_tip_base / base_norm
        else:
            dir_ = np.array([1.0, 0.0, 0.0])
        joint_1_to_tip_base_limit = joint_1_to_tip_base_max * dir_

        tip_base_height = np.sqrt(max(link_x_s3**2 - joint_1_to_tip_base_max**2, 0.0))

        # 2) Angles
        joint_1_to_tip_base_y_dir_length = np.dot(joint_1_to_tip_base_limit, joint_1_y_dir)
        theta_1_s3 = np.arctan2(joint_1_to_tip_base_y_dir_length, tip_base_height)
        theta_3_s3 = np.pi - cos2ndLaw_angle(link_2, link_3, link_x_s3)

        tip_position = (link_1 * joint_1_x_ndir
                        + joint_1_to_tip_base_limit
                        + tip_base_height * joint_1_z_dir)
        tip_position_norm = np.linalg.norm(tip_position)
        theta_2_s3_intm = cos2ndLaw_angle(link_1, link_x_s3, tip_position_norm)
        theta_2_s3_sub = cos2ndLaw_angle(link_x_s3, link_2, link_3)
        theta_2_s3 = np.pi - (theta_2_s3_intm + theta_2_s3_sub)

        # Choose stage3 angles
        final_theta_0 = theta_0_s3
        final_theta_1 = theta_1_s3
        final_theta_2 = theta_2_s3
        final_theta_3 = theta_3_s3
    else:
        # Stage2 angles are acceptable
        final_theta_0 = theta_0_s2
        final_theta_1 = theta_1_s2
        final_theta_2 = theta_2_s2
        final_theta_3 = theta_3_s2

    # Final joint angles (4)
    # The original code does a shift: final_theta_1 -= 90 deg in radians.
    joint_angle = np.array([
        final_theta_0,
        final_theta_1 - np.radians(90.0),
        final_theta_2,
        final_theta_3
    ], dtype=np.float64)

    return joint_angle