import numpy as np

from .geometry_utils_np import rotmat_to_quat, quat_to_rotmat, rotate_vector


def Transform_Wrist2Hand_np(
    quat_wrist: np.ndarray,  # (4,) format: w, x, y, z
    pos_wrist: np.ndarray,   # (3,) format: x, y, z
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms a single wrist quaternion and position to hand coordinates.

    Args:
        quat_wrist (np.ndarray): A wrist quaternion [w, x, y, z] of shape (4,)
        pos_wrist (np.ndarray): A wrist position [x, y, z] of shape (3,)

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - A hand quaternion [w, x, y, z] of shape (4,)
            - A hand position [x, y, z] of shape (3,)
    """
    # Normalize the input wrist quaternion
    qw_norm = np.linalg.norm(quat_wrist)
    if qw_norm > 0:
        quat_wrist = quat_wrist / qw_norm

    # Compute the wrist rotation matrix
    R_wrist = quat_to_rotmat(quat_wrist)

    # Construct the hand rotation matrix from the wrist matrix
    R_hand = np.zeros_like(R_wrist)
    # X_hand = -Y_wrist, Y_hand = X_wrist, Z_hand = Z_wrist
    R_hand[:, 0] = -R_wrist[:, 1]
    R_hand[:, 1] = R_wrist[:, 0]
    R_hand[:, 2] = R_wrist[:, 2]

    # Convert the hand rotation matrix to a quaternion
    quat_hand = rotmat_to_quat(R_hand)
    # Normalize the output hand quaternion
    norm_hand = np.linalg.norm(quat_hand)
    if norm_hand > 0:
        quat_hand = quat_hand / norm_hand

    # Translation vector t from wrist to hand
    t = np.array([0.0, -0.0165, -0.1026], dtype=np.float32)

    # pos_hand = pos_wrist - R_hand.dot(t)
    pos_hand = pos_wrist - R_hand.dot(t)

    return quat_hand, pos_hand


def FK_Thumb_np(
    joint_angle: np.ndarray,   # (4,) [θ0, θ1, θ2, θ3]
    hand_quat: np.ndarray,    # (4,) [w, x, y, z]
    hand_pos: np.ndarray,     # (3,) [x, y, z]
) -> np.ndarray:
    """
    Computes the forward kinematics for a thumb given single-input joint angles, hand quaternion, and hand position.

    Args:
        joint_angle (np.ndarray): Array of 4 thumb joint angles [θ0, θ1, θ2, θ3], shape (4,)
        hand_quat (np.ndarray): Hand orientation as a quaternion [w, x, y, z], shape (4,)
        hand_pos (np.ndarray): Hand position [x, y, z], shape (3,)

    Returns:
        np.ndarray: The 3D contact point of the thumb in the global frame, shape (3,)
    """

    link_1 = 0.0466
    link_2 = 0.0401
    link_3 = 0.027
    finger_deviation = -5.0  # in degrees

    Finger_base = np.array([-0.01044496692, -0.02255, -0.05777929249], dtype=np.float32)

    # Convert the hand quaternion to a rotation matrix
    norm_hand_quat = np.linalg.norm(hand_quat)
    if norm_hand_quat > 0:
        hand_quat = hand_quat / norm_hand_quat
    R_hand = quat_to_rotmat(hand_quat)
    hand_X_direction = R_hand[:, 0]
    hand_Y_direction = R_hand[:, 1]
    hand_Z_direction = R_hand[:, 2]

    # Finger base in the global frame
    finger_base_pose = (hand_pos
                        + hand_X_direction * Finger_base[0]
                        + hand_Y_direction * Finger_base[1]
                        + hand_Z_direction * Finger_base[2])

    # Construct thumb local coordinate axes
    dev_rad = np.deg2rad(finger_deviation)
    finger_z_axis = (np.sin(dev_rad) * hand_X_direction
                     + np.cos(dev_rad) * hand_Z_direction)
    finger_z_axis /= np.linalg.norm(finger_z_axis)
    finger_y_axis = hand_Y_direction
    finger_x_axis = np.cross(finger_y_axis, finger_z_axis)

    # Define rotation axes
    flexion_axis = np.array([0, 0, 1], dtype=np.float32)
    roll_axis = np.array([1, 0, 0], dtype=np.float32)
    right_direction = np.array([-1, 0, 0], dtype=np.float32)

    # Link transformations
    IP_to_Tip = link_3 * right_direction
    IP_rot = rotate_vector(IP_to_Tip, -joint_angle[3], flexion_axis)

    MP_interm_link = 0.0095
    MP_to_Tip = link_2 * right_direction + IP_rot
    MP_rot = rotate_vector(MP_to_Tip, -joint_angle[2], flexion_axis)

    CMC_to_Tip = (link_1 - MP_interm_link) * right_direction + MP_rot
    CMC_rot = rotate_vector(CMC_to_Tip, -joint_angle[1], roll_axis)

    Thumb_base_to_Tip = rotate_vector(CMC_rot, -joint_angle[0], flexion_axis)

    # Final contact point
    contact_point = (finger_base_pose
                     + Thumb_base_to_Tip[0] * finger_x_axis
                     + Thumb_base_to_Tip[1] * finger_y_axis
                     + Thumb_base_to_Tip[2] * finger_z_axis)

    return contact_point


def FK_Finger_np(
    finger_type: str,
    joint_angles: np.ndarray,  # (4,) [θ0, θ1, θ2, θ3]
    hand_quat: np.ndarray,     # (4,) [w, x, y, z]
    hand_pos: np.ndarray,      # (3,) [x, y, z]
) -> np.ndarray:
    """
    Computes the forward kinematics for a specified finger (Index, Middle, or Ring) given single-input joint angles,
    hand quaternion, and hand position.

    Args:
        finger_type (str): Type of the finger: 'Index', 'Middle', or 'Ring'
        joint_angles (np.ndarray): 4 joint angles [θ0, θ1, θ2, θ3], shape (4,)
        hand_quat (np.ndarray): Hand orientation as a quaternion [w, x, y, z], shape (4,)
        hand_pos (np.ndarray): Hand position [x, y, z], shape (3,)

    Returns:
        np.ndarray: The 3D contact point of the specified finger in the global frame, shape (3,)
    """
    # Link lengths
    link_0 = 0.0095
    link_1 = 0.0401
    link_2 = 0.0298
    link_3 = 0.025

    # Finger base and deviation
    if finger_type == "Index":
        Finger_base = np.array([-0.03234003159, -0.0099, 0.00318800373], dtype=np.float32)
        finger_deviation = -5.0
    elif finger_type == "Middle":
        Finger_base = np.array([0.0, -0.0099, 0.0046], dtype=np.float32)
        finger_deviation = 0.0
    elif finger_type == "Ring":
        Finger_base = np.array([0.03234003159, -0.0099, 0.00318800373], dtype=np.float32)
        finger_deviation = 5.0
    else:
        raise ValueError("Invalid finger_type. Must be 'Index', 'Middle', or 'Ring'.")

    # Consistent with the original Torch code: joint_angles = -joint_angles
    joint_angles = -joint_angles

    # Compute the hand rotation matrix
    norm_hand_quat = np.linalg.norm(hand_quat)
    if norm_hand_quat > 0:
        hand_quat = hand_quat / norm_hand_quat
    R_hand = quat_to_rotmat(hand_quat)
    hand_X_direction = R_hand[:, 0]
    hand_Y_direction = R_hand[:, 1]
    hand_Z_direction = R_hand[:, 2]

    # Finger base in the global frame
    finger_base_pose = (hand_pos
                        + hand_X_direction * Finger_base[0]
                        + hand_Y_direction * Finger_base[1]
                        + hand_Z_direction * Finger_base[2])

    # Finger local z-axis (including deviation)
    dev_rad = np.deg2rad(finger_deviation)
    finger_z_axis = (np.sin(dev_rad) * hand_X_direction
                     + np.cos(dev_rad) * hand_Z_direction)
    finger_z_axis /= np.linalg.norm(finger_z_axis)
    finger_y_axis = hand_Y_direction
    finger_x_axis = np.cross(finger_y_axis, finger_z_axis)

    # Rotation axes
    upright_direction = np.array([0, 0, 1], dtype=np.float32)
    horiz_rot_axis = np.array([1, 0, 0], dtype=np.float32)
    verti_rot_axis = np.array([0, 0, -1], dtype=np.float32)

    # Link transformations
    DIP_to_Tip = link_3 * upright_direction
    DIP_rot = rotate_vector(DIP_to_Tip, joint_angles[3], horiz_rot_axis)

    PIP_to_Tip = link_2 * upright_direction + DIP_rot
    PIP_rot = rotate_vector(PIP_to_Tip, joint_angles[2], horiz_rot_axis)

    MCP_to_Tip = link_1 * upright_direction + PIP_rot
    MCP_rot = rotate_vector(MCP_to_Tip, joint_angles[1], horiz_rot_axis)

    MCP_origin_to_Tip = rotate_vector(MCP_rot, joint_angles[0], verti_rot_axis)

    # Final contact point
    contact_point = (finger_base_pose
                     + MCP_origin_to_Tip[0] * finger_x_axis
                     + MCP_origin_to_Tip[1] * finger_y_axis
                     + MCP_origin_to_Tip[2] * finger_z_axis)

    return contact_point