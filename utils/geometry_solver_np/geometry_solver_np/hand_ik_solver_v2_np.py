import numpy as np
from typing import Literal
from .geometry_utils_np import (
    quat_to_rotmat,
    rotate_vector,
    cos2ndLaw_angle,
    cos2ndLaw_length,
    transform_vector_using_yz_axis
)


def IK_Finger_V2_np(
    finger_type: Literal["Index", "Middle", "Ring"],
    contact_point: np.ndarray,  # (3,)
    hand_quat: np.ndarray,  # (4,)  [w,x,y,z]
    hand_pos: np.ndarray,  # (3,)
) -> np.ndarray:
    # ── constants ─────────────────────────────────────────────────────
    j0_sat_range = 0.01
    j0_conv_range = 0.01

    link_0 = 0.019
    link_1 = 0.0335
    link_2 = 0.0267
    link_3 = 0.0195

    joint_angle = [0.0, 0.0, 0.0, 0.0]

    # ── finger-specific offsets & limits ───────────────────────────────
    if finger_type == "Index":
        finger_base = np.array([-0.02924, -0.00895, -0.01078])
        finger_deviation = -5.0
        j0_upper_ref = 15
        j0_lower_ref = -45
    elif finger_type == "Middle":
        finger_base = np.array([0.00000, -0.00895, -0.00950])
        finger_deviation = 0.0
        j0_upper_ref = 15
        j0_lower_ref = -15
    elif finger_type == "Ring":
        finger_base = np.array([0.02924, -0.00895, -0.01078])
        finger_deviation = 5.0
        j0_upper_ref = 45
        j0_lower_ref = -15

    # ── hand pose to basis vectors ────────────────────────────────────
    R_hand = quat_to_rotmat(hand_quat)  # (3,3)
    hand_X_direction = R_hand[:, 0]
    hand_Y_direction = R_hand[:, 1]
    hand_Z_direction = R_hand[:, 2]
    hand_center = hand_pos

    # finger base in world
    finger_base_pos = (
        hand_center
        + hand_X_direction * finger_base[0]
        + hand_Y_direction * finger_base[1]
        + hand_Z_direction * finger_base[2]
    )

    # finger local axes
    finger_z_axis = (
        np.sin(np.deg2rad(finger_deviation)) * hand_X_direction
        + np.cos(np.deg2rad(finger_deviation)) * hand_Z_direction
    )
    finger_z_axis /= np.linalg.norm(finger_z_axis)
    finger_y_axis = hand_Y_direction
    finger_x_axis = np.cross(finger_y_axis, finger_z_axis)
    finger_x_axis /= np.linalg.norm(finger_x_axis)

    # ── target in finger frame ────────────────────────────────────────
    target_posi = contact_point - finger_base_pos
    target_posi_fc = transform_vector_using_yz_axis(target_posi, finger_y_axis, finger_z_axis)

    if np.abs(target_posi_fc[2]) < j0_sat_range:
        joint_angle[0] = 0
    elif np.abs(target_posi_fc[2]) < (j0_sat_range + j0_conv_range):
        joint_angle[0] = np.arctan2(target_posi_fc[0], target_posi_fc[2]) / np.pi * 180.0
        if joint_angle[0] < -90:
            joint_angle[0] += 180
        elif joint_angle[0] > 90:
            joint_angle[0] -= 180
        j0_lower = (j0_lower_ref / j0_conv_range) * (np.abs(target_posi_fc[2]) - j0_sat_range)
        j0_upper = (j0_upper_ref / j0_conv_range) * (np.abs(target_posi_fc[2]) - j0_sat_range)
        joint_angle[0] = min(max(joint_angle[0], j0_lower), j0_upper)   
    else:
        joint_angle[0] = np.arctan2(target_posi_fc[0], target_posi_fc[2]) / np.pi * 180.0
        if joint_angle[0] < -90:
            joint_angle[0] += 180
        elif joint_angle[0] > 90:
            joint_angle[0] -= 180
        joint_angle[0] = min(max(joint_angle[0], j0_lower_ref), j0_upper_ref)

    # ── remaining joints (planar IK) ──────────────────────────────────
    max_len = link_1 + link_2 + link_3
    min_len = np.sqrt(link_2**2 + (link_1 - link_3) ** 2)

    upright_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    finger_dir_z = rotate_vector(finger_z_axis, np.deg2rad(joint_angle[0]), finger_y_axis)
    target_posi_fc = transform_vector_using_yz_axis(target_posi, finger_y_axis, finger_dir_z)
    target_posi_fc -= np.dot(target_posi_fc, np.array([1.0, 0.0, 0.0])) * np.array([1.0, 0.0, 0.0])
    target_posi_fc -= link_0 * upright_direction
    target_dist = np.linalg.norm(target_posi_fc)

    # J3 (DIP)
    if target_dist > max_len:
        joint_angle[3] = 0.0
        tip_dist = max_len
    elif target_dist > min_len:
        joint_angle[3] = 90.0 * (max_len - target_dist) / (max_len - min_len)
        tip_dist = target_dist
    else:
        joint_angle[3] = 90.0
        tip_dist = min_len

    link_x = cos2ndLaw_length(link_2, link_3, np.deg2rad(180.0 - joint_angle[3]))
    theta_intm = cos2ndLaw_angle(link_1, link_x, tip_dist)
    theta_sub = cos2ndLaw_angle(link_2, link_x, link_3)
    joint_angle[2] = 180.0 - np.rad2deg(theta_intm + theta_sub)

    # J1 (MCP flexion)
    if target_posi_fc[2] > 0.0:
        theta_goal_dir = np.arctan2(np.sqrt(target_posi_fc[0]**2 + target_posi_fc[2]**2), target_posi_fc[1])
    else:
        theta_goal_dir = np.arctan2(-np.sqrt(target_posi_fc[0]**2 + target_posi_fc[2]**2), target_posi_fc[1])
    theta_goal_sub = cos2ndLaw_angle(link_1, tip_dist, link_x)
    joint_angle[1] = 90.0 - np.rad2deg(theta_goal_dir + theta_goal_sub)

    joint_angles_rad = np.radians(np.asarray(joint_angle, dtype=np.float32))
    return joint_angles_rad


def IK_Thumb_V2_np(
    contact_point: np.ndarray,        # (3,)
    hand_quat: np.ndarray,            # (4,)  [w, x, y, z]
    hand_pos: np.ndarray,             # (3,)
) -> np.ndarray:
    # ── link lengths ───────────────────────────────────────────────
    link_1 = 0.0462
    link_2 = 0.0335
    link_3 = 0.0195

    finger_deviation = -5.0  # (deg)

    # output container
    joint_angle = [0.0, 0.0, 0.0, 0.0]

    # ── thumb base offset (local hand frame) ───────────────────────
    finger_base_local = np.array([-0.01423, -0.0145, -0.04425])

    # ── hand frame axes from quaternion ────────────────────────────
    R_hand = quat_to_rotmat(hand_quat)                     # (3, 3)
    x_hand, y_hand, z_hand = R_hand[:, 0], R_hand[:, 1], R_hand[:, 2]

    # ── thumb base position in world ──────────────────────────────
    finger_base_pos = (
        hand_pos
        + x_hand * finger_base_local[0]
        + y_hand * finger_base_local[1]
        + z_hand * finger_base_local[2]
    )

    # ── build local axes for thumb frame ──────────────────────────
    finger_z_axis = (
        np.sin(np.deg2rad(finger_deviation)) * x_hand
        + np.cos(np.deg2rad(finger_deviation)) * z_hand
    )
    finger_z_axis /= np.linalg.norm(finger_z_axis)
    finger_y_axis = y_hand
    finger_x_axis = np.cross(finger_y_axis, finger_z_axis)
    finger_x_axis /= np.linalg.norm(finger_x_axis)

    # ── target position / heading expressed in thumb frame ────────
    is_large_opening = False
    target_posi = contact_point - finger_base_pos
    target_posi_fc = transform_vector_using_yz_axis(target_posi, finger_y_axis, finger_z_axis)
    finger_heading = hand_pos - contact_point

    if target_posi_fc[2] < 0:
        is_large_opening = True
    else:
        target_posi_fc[2] = max(0.0, target_posi_fc[2])

        target_heading = finger_heading / np.linalg.norm(finger_heading)
        target_heading_fc = transform_vector_using_yz_axis(target_heading, finger_y_axis, finger_z_axis)

        # Stage 1: If the target point is reachable with the target direction
        on_base_plane = np.array([
            target_posi_fc[0] - (target_posi_fc[2] / target_heading_fc[2]) * target_heading_fc[0],
            target_posi_fc[1] - (target_posi_fc[2] / target_heading_fc[2]) * target_heading_fc[1],
            0.0
        ])

        theta_0_s1 = np.arctan2(on_base_plane[1], -on_base_plane[0])

        j1_x_ndir = np.array([-np.cos(theta_0_s1), np.sin(theta_0_s1), 0.0])  
        j1_z_dir = np.array([0.0, 0.0, 1.0])  
        j1_y_dir = np.array([np.sin(theta_0_s1), np.cos(theta_0_s1), 0.0])

        theta_1_s1 = np.arctan2(np.dot(target_heading_fc, j1_y_dir), np.dot(target_heading_fc, j1_z_dir))

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

            joint_0_to_tip_base_length = min(np.linalg.norm(Target_base_plane), link_1 + link_2 + link_3)

            joint_1_to_tip_plane_length_sq = max((joint_0_to_tip_base_length - link_1)**2, link_x**2 - target_posi_fc[2]**2)
            joint_1_to_tip_plane_length = np.sqrt(joint_1_to_tip_plane_length_sq)

            joint_1_y_dir_length = joint_0_to_tip_base_length * np.sin(cos2ndLaw_angle(joint_0_to_tip_base_length, link_1, joint_1_to_tip_plane_length))

            target_direction = target_posi_fc / np.linalg.norm(target_posi_fc)
            direction_check = np.array([target_direction[1], -target_direction[0], 0.0])

            if np.dot(target_heading_fc, direction_check) > 0:
                theta_1_s2 = np.arctan2(joint_1_y_dir_length, np.sqrt(link_x**2 - joint_1_to_tip_plane_length_sq))

                theta_0_s2_intm = np.arctan2(target_posi_fc[1], -target_posi_fc[0])
                theta_0_s2_sub = cos2ndLaw_angle(link_1, joint_0_to_tip_base_length, joint_1_to_tip_plane_length)

                theta_0_s2 = theta_0_s2_intm - theta_0_s2_sub
            else:
                theta_1_s2 = -np.arctan2(joint_1_y_dir_length, np.sqrt(link_x**2 - joint_1_to_tip_plane_length_sq))

                theta_0_s2_intm = np.arctan2(target_posi_fc[1], -target_posi_fc[0])
                theta_0_s2_sub = cos2ndLaw_angle(link_1, joint_0_to_tip_base_length, joint_1_to_tip_plane_length)

                theta_0_s2 = theta_0_s2_intm + theta_0_s2_sub

            joint_1_point = np.array([-link_1 * np.cos(theta_0_s2), link_1 * np.sin(theta_0_s2), 0])

            joint_1_dir = joint_1_point / np.linalg.norm(joint_1_point)

            joint_1_to_tip = target_posi_fc - joint_1_point

            joint_1_to_tip_dir = joint_1_to_tip / np.linalg.norm(joint_1_to_tip)

            theta_2_s2_intm_safety = np.dot(joint_1_dir, joint_1_to_tip_dir)
            theta_2_s2_intm_safety = np.clip(theta_2_s2_intm_safety, -1.0, 1.0)

            theta_2_s2_intm = np.pi - np.arccos(theta_2_s2_intm_safety)    

            theta_2_s2_sub = cos2ndLaw_angle(link_x, link_2, link_3)

            theta_2_s2 = np.pi - (theta_2_s2_intm + theta_2_s2_sub)

        # Stage 3: If theta_0 is out of range (0, 125)
        if 0 <= theta_0_s2 <= (125 / 180) * np.pi:
            theta_0_s3 = theta_0_s2
            theta_1_s3 = theta_1_s2
            theta_2_s3 = theta_2_s2
            theta_3_s3 = theta_3_s2
        else:
            theta_0_s3 = max(0.0, min(theta_0_s2, (125 / 180) * np.pi))

            joint_1_posi = np.array([-link_1 * np.cos(theta_0_s3), link_1 * np.sin(theta_0_s3), 0.0])

            link_x_vect = target_posi_fc - joint_1_posi
            link_x = np.linalg.norm(link_x_vect)

            link_x = max(min_length, min(link_x, max_length))
            joint_1_x_ndir = np.array([-np.cos(theta_0_s3), np.sin(theta_0_s3), 0.0])
            joint_1_z_dir = np.array([0.0, 0.0, 1.0])
            joint_1_y_dir = np.array([np.sin(theta_0_s3), np.cos(theta_0_s3), 0.0])

            joint_1_to_tip_base = np.array([link_x_vect[0], link_x_vect[1], 0.0])
            joint_1_to_tip_base_max = min(link_x, np.linalg.norm(joint_1_to_tip_base))

            joint_1_to_tip_base_limit = joint_1_to_tip_base_max * (joint_1_to_tip_base / np.linalg.norm(joint_1_to_tip_base))
            tip_base_height = np.sqrt(link_x**2 - joint_1_to_tip_base_max**2)

            joint_1_to_tip_base_y_dir_length = np.dot(joint_1_to_tip_base_limit, joint_1_y_dir)

            theta_1_s3 = np.arctan2(joint_1_to_tip_base_y_dir_length, tip_base_height)

            theta_3_s3 = np.pi - cos2ndLaw_angle(link_2, link_3, link_x)

            tip_position = link_1 * joint_1_x_ndir + joint_1_to_tip_base_limit + tip_base_height * joint_1_z_dir

            theta_2_s3_intm = cos2ndLaw_angle(link_1, link_x, np.linalg.norm(tip_position))
            theta_2_s3_sub = cos2ndLaw_angle(link_x, link_2, link_3)

            theta_2_s3 = np.pi - (theta_2_s3_intm + theta_2_s3_sub)

        joint_angle[0] = theta_0_s3 * 180.0 / np.pi
        joint_angle[1] = theta_1_s3 * 180.0 / np.pi - 90.0
        joint_angle[2] = theta_2_s3 * 180.0 / np.pi
        joint_angle[3] = theta_3_s3 * 180.0 / np.pi 

    if is_large_opening:
        target_posi = contact_point - finger_base_pos

        target_posi_fc = transform_vector_using_yz_axis(target_posi, finger_y_axis, finger_z_axis)

        theta_1 = -0.5 * np.pi

        theta_0 = np.arctan2(target_posi_fc[1], -target_posi_fc[0])

        if theta_0 > (125 / 180) * np.pi:
            theta_0 = (125 / 180) * np.pi
            target_posi_fc_norm = np.linalg.norm(target_posi_fc)
            target_posi_fc = [-target_posi_fc_norm * np.cos(theta_0), target_posi_fc_norm * np.sin(theta_0), target_posi_fc[2]]

        elif theta_0 < 0:
            theta_0 = 0
            target_posi_fc_norm = np.linalg.norm(target_posi_fc)
            target_posi_fc = [-target_posi_fc_norm, 0, target_posi_fc[2]]

        j1_posi = finger_base_pos + link_1 * (-np.cos(theta_0)) * finger_x_axis + link_1 * (np.sin(theta_0)) * finger_y_axis

        j1_to_contact = contact_point - j1_posi

        link_x = np.linalg.norm(j1_to_contact)

        max_length = link_2 + link_3
        min_length = np.sqrt(link_2**2 + link_3**2)

        if min_length <= link_x <= max_length:
            theta_3 = np.pi - cos2ndLaw_angle(link_2, link_3, link_x)

            theta_2_sub = cos2ndLaw_angle(link_x, link_2, link_3)

            j1_to_contact_fc = transform_vector_using_yz_axis(j1_to_contact, finger_y_axis, finger_z_axis)

            theta_2_intm = np.arctan2(-j1_to_contact_fc[2], np.sqrt(j1_to_contact_fc[0]**2 + j1_to_contact_fc[1]**2))

            theta_2 = -theta_2_sub - theta_2_intm

        else:
            if link_x < min_length:
                link_x = min_length
                theta_3 = 0.5 * np.pi
            else:
                link_x = max_length
                theta_3 = 0

        theta_2_sub = cos2ndLaw_angle(link_x, link_2, link_3)

        j1_to_contact = j1_to_contact / np.linalg.norm(j1_to_contact) * link_x

        j1_to_contact_fc = transform_vector_using_yz_axis(j1_to_contact, finger_y_axis, finger_z_axis)

        theta_2_intm = np.arctan2(-j1_to_contact_fc[2], np.sqrt(j1_to_contact_fc[0]**2 + j1_to_contact_fc[1]**2))

        theta_2 = -theta_2_sub - theta_2_intm

        joint_angle[0] = theta_0 * 180.0 / np.pi
        joint_angle[1] = theta_1 * 180.0 / np.pi
        joint_angle[2] = theta_2 * 180.0 / np.pi
        joint_angle[3] = theta_3 * 180.0 / np.pi

    joint_angle_rad = np.radians(np.asarray(joint_angle, dtype=np.float32))

    return joint_angle_rad
