import numpy as np
from typing import Literal
from .geometry_utils_np import (
    quat_to_rotmat,
)


def rotate_vector(vector, angle, axis):
    # Ensure the axis is a unit vector
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)

    # Convert the angle to radians if it is not
    # angle = np.radians(angle)

    # Rodrigues' rotation formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product = np.cross(axis, vector)

    dot_product = np.dot(axis, vector)

    rotated_vector = (
        vector * cos_angle
        + cross_product * sin_angle
        + axis * dot_product * (1 - cos_angle)
    )

    return rotated_vector


def vector_normal_to(vector, normal, isNormalize):
    result_vec = vector - np.dot(vector, normal) * normal

    if isNormalize:
        return result_vec / np.linalg.norm(result_vec)
    else:
        return result_vec


def transform_vector_using_yz_axis(vector, y_axis, z_axis):
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis, z_axis)
    return np.array(
        [np.dot(vector, x_axis), np.dot(vector, y_axis), np.dot(vector, z_axis)]
    )


def cos2ndLaw_length(side_length_1, side_length_2, angle_in_rad):
    return np.sqrt(
        side_length_1**2
        + side_length_2**2
        - 2 * side_length_1 * side_length_2 * np.cos(angle_in_rad)
    )


def cos2ndLaw_angle(side_length_1, side_length_2, far_length):
    data = (side_length_1**2 + side_length_2**2 - far_length**2) / (
        2 * side_length_1 * side_length_2
    )
    data = np.clip(
        data, -1.0, 1.0
    )  # Ensuring the value is within the valid range for arccos
    return np.arccos(data)


def ik_thumb_v2_np(
    contact_point: np.ndarray,  # (3,)
    opening_direction: np.ndarray,  # (4, )
    hand_quat: np.ndarray,  # (4,)  [w,x,y,z]
    hand_pos: np.ndarray,  # (3,)
) -> np.ndarray:
    hand_pose = np.eye(4, dtype=np.float32)
    R_hand = quat_to_rotmat(hand_quat)  # (3,3)
    hand_pose[0:3, 0:3] = R_hand
    hand_pose[0, 3] = hand_pos[0]
    hand_pose[1, 3] = hand_pos[1]
    hand_pose[2, 3] = hand_pos[2]

    link_1 = 0.0462
    link_2 = 0.0335
    link_3 = 0.0195

    finger_deviation = -5

    joint_angle = [0.0, 0.0, 0.0, 0.0]

    Finger_base = np.array([-0.0145, 0.01423, -0.04425])

    hand_center = hand_pose[:3, 3]
    hand_X_direction = hand_pose[:3, 0]
    hand_Y_direction = hand_pose[:3, 1]
    hand_Z_direction = hand_pose[:3, 2]

    finger_base_pose = (
        hand_center
        + hand_X_direction * Finger_base[0]
        + hand_Y_direction * Finger_base[1]
        + hand_Z_direction * Finger_base[2]
    )

    finger_z_axis = (
        np.sin(finger_deviation / 180.0 * np.pi) * hand_X_direction
        + np.cos(finger_deviation / 180.0 * np.pi) * hand_Z_direction
    )
    finger_z_axis = finger_z_axis / np.linalg.norm(finger_z_axis)
    finger_y_axis = hand_Y_direction
    finger_x_axis = np.cross(finger_y_axis, finger_z_axis)

    contact_point = contact_point
    finger_heading = -opening_direction

    target_posi = contact_point - finger_base_pose

    target_posi_fc = transform_vector_using_yz_axis(
        target_posi, finger_y_axis, finger_z_axis
    )
    target_posi_fc[2] = max(0.0, target_posi_fc[2])

    target_heading = finger_heading / np.linalg.norm(finger_heading)
    target_heading_fc = transform_vector_using_yz_axis(
        target_heading, finger_y_axis, finger_z_axis
    )

    # Stage 1: If the target point is reachable with the target direction
    on_base_plane = np.array(
        [
            target_posi_fc[0]
            - (target_posi_fc[2] / target_heading_fc[2]) * target_heading_fc[0],
            target_posi_fc[1]
            - (target_posi_fc[2] / target_heading_fc[2]) * target_heading_fc[1],
            0.0,
        ]
    )

    theta_0_s1 = np.arctan2(on_base_plane[1], -on_base_plane[0])

    j1_x_ndir = np.array([-np.cos(theta_0_s1), np.sin(theta_0_s1), 0.0])
    j1_z_dir = np.array([0.0, 0.0, 1.0])
    j1_y_dir = np.array([np.sin(theta_0_s1), np.cos(theta_0_s1), 0.0])

    theta_1_s1 = np.arctan2(
        np.dot(target_heading_fc, j1_y_dir), np.dot(target_heading_fc, j1_z_dir)
    )

    if theta_1_s1 > 0.5 * np.pi:
        theta_1_s1 -= np.pi
    elif theta_1_s1 < -0.5 * np.pi:
        theta_1_s1 += np.pi

    link_x_vect = target_posi_fc - link_1 * j1_x_ndir
    link_x = np.linalg.norm(link_x_vect)

    theta_3_s1 = np.pi - cos2ndLaw_angle(link_2, link_3, link_x)

    theta_2_intm_s1_safety = np.dot(link_x_vect, j1_x_ndir) / link_x
    theta_2_intm_s1_safety = np.clip(theta_2_intm_s1_safety, -1.0, 1.0)

    theta_2_intm_s1 = np.arccos(theta_2_intm_s1_safety)
    theta_2_sub_s1 = cos2ndLaw_angle(link_x, link_2, link_3)

    theta_2_s1 = theta_2_intm_s1 - theta_2_sub_s1

    # Stage 2: If link_x is out of range, neglect heading direction
    max_length = link_2 + link_3
    min_length = np.sqrt(link_2**2 + link_3**2)

    if min_length <= link_x <= max_length:
        theta_0_s2 = theta_0_s1
        theta_1_s2 = theta_1_s1
        theta_2_s2 = theta_2_s1
        theta_3_s2 = theta_3_s1
    else:
        if link_x > max_length:
            theta_3_s2 = 0
            link_x = max_length
        elif link_x < min_length:
            theta_3_s2 = 90 * np.pi / 180.0
            link_x = min_length
        else:
            link_x = max_length
            theta_3_s2 = 0

        Target_base_plane = np.array([target_posi_fc[0], target_posi_fc[1], 0.0])

        joint_0_to_tip_base_length = min(
            np.linalg.norm(Target_base_plane), link_1 + link_2 + link_3
        )

        joint_1_to_tip_plane_length_sq = max(
            (joint_0_to_tip_base_length - link_1) ** 2,
            link_x**2 - target_posi_fc[2] ** 2,
        )
        joint_1_to_tip_plane_length = np.sqrt(joint_1_to_tip_plane_length_sq)

        joint_1_y_dir_length = joint_0_to_tip_base_length * np.sin(
            cos2ndLaw_angle(
                joint_0_to_tip_base_length, link_1, joint_1_to_tip_plane_length
            )
        )

        target_direction = target_posi_fc / np.linalg.norm(target_posi_fc)
        direction_check = np.array([target_direction[1], -target_direction[0], 0.0])

        if np.dot(target_heading_fc, direction_check) > 0:
            theta_1_s2 = np.arctan2(
                joint_1_y_dir_length,
                np.sqrt(link_x**2 - joint_1_to_tip_plane_length_sq),
            )

            theta_0_s2_intm = np.arctan2(target_posi_fc[1], -target_posi_fc[0])
            theta_0_s2_sub = cos2ndLaw_angle(
                link_1, joint_0_to_tip_base_length, joint_1_to_tip_plane_length
            )

            theta_0_s2 = theta_0_s2_intm - theta_0_s2_sub
        else:
            theta_1_s2 = -np.arctan2(
                joint_1_y_dir_length,
                np.sqrt(link_x**2 - joint_1_to_tip_plane_length_sq),
            )

            theta_0_s2_intm = np.arctan2(target_posi_fc[1], -target_posi_fc[0])
            theta_0_s2_sub = cos2ndLaw_angle(
                link_1, joint_0_to_tip_base_length, joint_1_to_tip_plane_length
            )

            theta_0_s2 = theta_0_s2_intm + theta_0_s2_sub

        joint_1_point = np.array(
            [-link_1 * np.cos(theta_0_s2), link_1 * np.sin(theta_0_s2), 0]
        )  ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        joint_1_dir = joint_1_point / np.linalg.norm(
            joint_1_point
        )  ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        joint_1_to_tip = (
            target_posi_fc - joint_1_point
        )  ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        joint_1_to_tip_dir = joint_1_to_tip / np.linalg.norm(
            joint_1_to_tip
        )  ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        theta_2_s2_intm_safety = np.dot(joint_1_dir, joint_1_to_tip_dir)
        theta_2_s2_intm_safety = np.clip(theta_2_s2_intm_safety, -1.0, 1.0)

        theta_2_s2_intm = np.pi - np.arccos(theta_2_s2_intm_safety)

        # theta_2_s2_intm=np.pi- np.arccos(np.dot(joint_1_dir,joint_1_to_tip_dir))               ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        #   theta_2_s2_intm = cos2ndLaw_angle(link_1, link_x, np.sqrt(joint_0_to_tip_base_length**2 + link_x**2 - joint_1_to_tip_plane_length_sq))
        theta_2_s2_sub = cos2ndLaw_angle(link_x, link_2, link_3)

        theta_2_s2 = np.pi - (theta_2_s2_intm + theta_2_s2_sub)

    # Stage 3: If theta_0 is out of range (0, 125)
    if 0 <= theta_0_s2 <= 125:
        theta_0_s3 = theta_0_s2
        theta_1_s3 = theta_1_s2
        theta_2_s3 = theta_2_s2
        theta_3_s3 = theta_3_s2
    else:
        theta_0_s3 = max(0.0, min(theta_0_s2, 125.0))

        joint_1_posi = np.array(
            [-link_1 * np.cos(theta_0_s3), link_1 * np.sin(theta_0_s3), 0.0]
        )

        link_x_vect = target_posi_fc - joint_1_posi
        link_x = np.linalg.norm(link_x_vect)

        link_x = max(min_length, min(link_x, max_length))

        joint_1_x_ndir = np.array([-np.cos(theta_0_s3), np.sin(theta_0_s3), 0.0])
        joint_1_z_dir = np.array([0.0, 0.0, 1.0])
        joint_1_y_dir = np.array([np.sin(theta_0_s3), np.cos(theta_0_s3), 0.0])

        joint_1_to_tip_base = np.array([link_x_vect[0], link_x_vect[1], 0.0])
        joint_1_to_tip_base_max = min(link_x, np.linalg.norm(joint_1_to_tip_base))

        joint_1_to_tip_base_limit = joint_1_to_tip_base_max * (
            joint_1_to_tip_base / np.linalg.norm(joint_1_to_tip_base)
        )
        tip_base_height = np.sqrt(link_x**2 - joint_1_to_tip_base_max**2)

        joint_1_to_tip_base_y_dir_length = np.dot(
            joint_1_to_tip_base_limit, joint_1_y_dir
        )

        theta_1_s3 = np.arctan2(joint_1_to_tip_base_y_dir_length, tip_base_height)

        theta_3_s3 = np.pi - cos2ndLaw_angle(link_2, link_3, link_x)

        tip_position = (
            link_1 * joint_1_x_ndir
            + joint_1_to_tip_base_limit
            + tip_base_height * joint_1_z_dir
        )

        theta_2_s3_intm = cos2ndLaw_angle(link_1, link_x, np.linalg.norm(tip_position))
        theta_2_s3_sub = cos2ndLaw_angle(link_x, link_2, link_3)

        theta_2_s3 = np.pi - (theta_2_s3_intm + theta_2_s3_sub)

    joint_angle[0] = theta_0_s3 * 180.0 / np.pi
    joint_angle[1] = theta_1_s3 * 180.0 / np.pi - 90.0
    joint_angle[2] = theta_2_s3 * 180.0 / np.pi
    joint_angle[3] = theta_3_s3 * 180.0 / np.pi

    return np.asarray(joint_angle, dtype=np.float64)


def ik_finger_v2_np(
    finger_type: Literal["Index", "Middle", "Ring"],
    contact_point: np.ndarray,  # (3,)
    hand_quat: np.ndarray,  # (4,)  [w,x,y,z]
    hand_pos: np.ndarray,  # (3,)
) -> np.ndarray:
    hand_pose = np.eye(4, dtype=np.float32)
    R_hand = quat_to_rotmat(hand_quat)  # (3,3)
    hand_pose[0:3, 0:3] = R_hand
    hand_pose[0, 3] = hand_pos[0]
    hand_pose[1, 3] = hand_pos[1]
    hand_pose[2, 3] = hand_pos[2]

    link_1 = 0.0462
    link_2 = 0.0335
    link_3 = 0.0195

    j0_sat_range = 0.01
    j0_conv_range = 0.01

    link_0 = 0.019
    link_1 = 0.0335
    link_2 = 0.0267
    link_3 = 0.0195

    link_3_palmar_base = 0.013
    link_3_palmar_min = 0.0065
    link_3_finger_tip = 0.0195

    link_3_lateral_half_width = 0.009

    joint_angle = [0.0, 0.0, 0.0, 0.0]

    if finger_type == "Index":
        # Finger_base=np.array([-0.03234003159,-0.0099,0.00318800373])
        Finger_base = np.array([-0.02924, -0.00895, -0.01078])
        finger_deviation = -5
        j0_upper_ref = 15
        j0_lower_ref = -45
    elif finger_type == "Middle":
        # Finger_base=np.array([0,-0.0099,0.0046])
        Finger_base = np.array([0, -0.00895, -0.0095])
        finger_deviation = 0
        j0_upper_ref = 15
        j0_lower_ref = -15
    elif finger_type == "Ring":
        # Finger_base=np.array([0.03234003159,-0.0099,0.00318800373])
        Finger_base = np.array([0.02924, -0.00895, -0.01078])
        finger_deviation = 5
        j0_upper_ref = 45
        j0_lower_ref = -15

    hand_center = hand_pose[:3, 3]
    hand_X_direction = hand_pose[:3, 0]
    hand_Y_direction = hand_pose[:3, 1]
    hand_Z_direction = hand_pose[:3, 2]

    finger_base_pose = (
        hand_center
        + hand_X_direction * Finger_base[0]
        + hand_Y_direction * Finger_base[1]
        + hand_Z_direction * Finger_base[2]
    )

    finger_z_axis = (
        np.sin(finger_deviation / 180.0 * np.pi) * hand_X_direction
        + np.cos(finger_deviation / 180.0 * np.pi) * hand_Z_direction
    )
    finger_z_axis = finger_z_axis / np.linalg.norm(finger_z_axis)
    finger_y_axis = hand_Y_direction
    finger_x_axis = np.cross(finger_y_axis, finger_z_axis)

    targetPosition = contact_point

    target_posi = targetPosition - finger_base_pose

    target_posi_fc = transform_vector_using_yz_axis(
        target_posi, finger_y_axis, finger_z_axis
    )
    upright_direction = np.array([0.0, 0.0, 1.0])

    if np.abs(target_posi_fc[2]) < j0_sat_range:
        joint_angle[0] = 0

    elif np.abs(target_posi_fc[2]) < (j0_sat_range + j0_conv_range):
        joint_angle[0] = (
            np.arctan2(target_posi_fc[0], target_posi_fc[2]) / np.pi * 180.0
        )

        j0_lower = (j0_lower_ref / j0_conv_range) * (
            np.abs(target_posi_fc[2]) - j0_sat_range
        )
        j0_upper = (j0_upper_ref / j0_conv_range) * (
            np.abs(target_posi_fc[2]) - j0_sat_range
        )

        joint_angle[0] = min(max(joint_angle[0], j0_lower), j0_upper)

    else:
        joint_angle[0] = (
            np.arctan2(target_posi_fc[0], target_posi_fc[2]) / np.pi * 180.0
        )
        joint_angle[0] = min(max(joint_angle[0], j0_lower_ref), j0_upper_ref)

    max_length = link_1 + link_2 + link_3
    min_length = np.sqrt(link_2**2 + (link_1 - link_3) ** 2)

    finger_direction_modified_z = rotate_vector(
        upright_direction, joint_angle[0] / 180 * np.pi, np.array([0.0, 1.0, 0.0])
    )

    target_posi_fc -= link_0 * finger_direction_modified_z

    target_distance = np.linalg.norm(target_posi_fc)

    # DIP angle is coupled with target_distance. the profile could be changed based on needs
    if target_distance > max_length:
        joint_angle[3] = 0.0
        Tip_distance = max_length
    elif target_distance > min_length:
        joint_angle[3] = 90 * (max_length - target_distance) / (max_length - min_length)
        Tip_distance = target_distance
    else:
        joint_angle[3] = 90
        Tip_distance = min_length

    link_x = cos2ndLaw_length(link_2, link_3, (180.0 - joint_angle[3]) / 180.0 * np.pi)

    theta_intm = cos2ndLaw_angle(link_1, link_x, Tip_distance)
    theta_sub = cos2ndLaw_angle(link_2, link_x, link_3)

    joint_angle[2] = 180.0 - (theta_intm + theta_sub) * 180.0 / np.pi

    if target_posi_fc[2] > 0:
        theta_goal_dir = np.arctan2(
            np.sqrt(target_posi_fc[0] ** 2 + target_posi_fc[2] ** 2),
            target_posi_fc[1],
        )
    else:
        theta_goal_dir = np.arctan2(
            -np.sqrt(target_posi_fc[0] ** 2 + target_posi_fc[2] ** 2),
            target_posi_fc[1],
        )

    theta_goal_sub = cos2ndLaw_angle(link_1, Tip_distance, link_x)

    joint_angle[1] = 90.0 - (theta_goal_dir + theta_goal_sub) * 180.0 / np.pi

    return np.asarray(joint_angle, dtype=np.float64)
