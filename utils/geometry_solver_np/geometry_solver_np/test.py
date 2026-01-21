import numpy as
def get_joint_angle(contact_info, hand_pose, available_modification):
    # contact_info_finger=Contact_info(finger=item.finger,target_segment=item.target_segment,contact_type=item.contact_type,dsq_set=object_dsq,object_pose=object_pose,grasping_axis=grasping_axis,opening_direction=openingDirection,contact_point=contact_point,contact_normal=contact_normal,finger_direction='dummy',joint_angle='dummy')

    j0_sat_range = 0.01
    j0_conv_range = 0.01

    link_0 = 0.019
    link_1 = 0.0335
    link_2 = 0.0267
    link_3 = 0.0195

    link_3_palmar_base = 0.01625

    link_3_palmar_min = 0.013
    link_3_finger_tip = 0.0195

    link_3_lateral_half_width = 0.009

    joint_angle = [0.0, 0.0, 0.0, 0.0]

    if contact_info.finger == "Index":
        # Finger_base=np.array([-0.03234003159,-0.0099,0.00318800373])
        Finger_base = np.array([-0.02924, -0.00895, -0.01078])
        finger_deviation = -5
        j0_upper_ref = 15
        j0_lower_ref = -45
    elif contact_info.finger == "Middle":
        # Finger_base=np.array([0,-0.0099,0.0046])
        Finger_base = np.array([0, -0.00895, -0.0095])
        finger_deviation = 0
        j0_upper_ref = 15
        j0_lower_ref = -15
    elif contact_info.finger == "Ring":
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

    if is_fingertip:

        targetPosition = contact_info.contact_point

        target_posi = targetPosition - finger_base_pose

        target_posi_fc = transform_vector_using_yz_axis(
            target_posi, finger_y_axis, finger_z_axis
        )
        upright_direction = np.array([0.0, 0.0, 1.0])

        # joint_angle[0]=np.arctan2(target_posi_fc[0], target_posi_fc[2]) / np.pi * 180.0

        # joint_angle[0]=min(max(joint_angle[0],j0_lower),j0_upper)

        if np.abs(target_posi_fc[2]) < j0_sat_range:
            joint_angle[0] = 0

        elif np.abs(target_posi_fc[2]) < (j0_sat_range + j0_conv_range):
            joint_angle[0] = (
                np.arctan2(target_posi_fc[0], target_posi_fc[2]) / np.pi * 180.0
            )
            if joint_angle[0] < -90:
                joint_angle[0] += 180
            elif joint_angle[0] > 90:
                joint_angle[0] -= 180

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
            if joint_angle[0] < -90:
                joint_angle[0] += 180
            elif joint_angle[0] > 90:
                joint_angle[0] -= 180
            joint_angle[0] = min(max(joint_angle[0], j0_lower_ref), j0_upper_ref)

        max_length = link_1 + link_2 + link_3
        min_length = np.sqrt(link_2**2 + (link_1 - link_3) ** 2)

        finger_z_heading = rotate_vector(
            finger_z_axis, joint_angle[0] / 180 * np.pi, finger_y_axis
        )
        target_posi_fc = transform_vector_using_yz_axis(
            target_posi, finger_y_axis, finger_z_heading
        )
        upright_direction = np.array([0.0, 0.0, 1.0])
        target_posi_fc -= np.dot(target_posi_fc, np.array([1.0, 0.0, 0.0])) * np.array(
            [1.0, 0.0, 0.0]
        )
        target_posi_fc -= link_0 * upright_direction

        target_distance = np.linalg.norm(target_posi_fc)

        # DIP angle is coupled with target_distance. the profile could be changed based on needs
        if target_distance > max_length:
            joint_angle[3] = 0.0
            Tip_distance = max_length
        elif target_distance > min_length:
            joint_angle[3] = (
                90 * (max_length - target_distance) / (max_length - min_length)
            )
            Tip_distance = target_distance
        else:
            joint_angle[3] = 90
            Tip_distance = min_length

        link_x = cos2ndLaw_length(
            link_2, link_3, (180.0 - joint_angle[3]) / 180.0 * np.pi
        )

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

    return joint_angle, condition, link_3


def get_joint_angle_Thumb(contact_info, hand_pose, available_modification):
    # contact_info_finger=Contact_info(finger=item.finger,target_segment=item.target_segment,contact_type=item.contact_type,dsq_set=object_dsq,object_pose=object_pose,grasping_axis=grasping_axis,opening_direction=openingDirection,contact_point=contact_point,contact_normal=contact_normal,finger_direction='dummy',joint_angle='dummy')

    link_1 = 0.0462
    link_2 = 0.0335
    link_3 = 0.0195

    link_3_finger_tip = 0.0195

    link_3_palmar_min = 0.013

    link_3_lateral_half_width = 0.009

    finger_deviation = -5

    joint_angle = [0.0, 0.0, 0.0, 0.0]

    # Finger_base=np.array([-0.01044496692,-0.02255,-0.05777929249])
    Finger_base = np.array([-0.01423, -0.0145, -0.04425])

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

    contact_point = contact_info.contact_point
    finger_heading = -contact_info.opening_direction

    condition = "default"

    is_fingertip = False

    is_reverse_theta_3 = False

    is_large_opening = False

    if (
        (contact_info.contact_type == "finger tip")
        or (contact_info.contact_type == "lateral side")
        or (is_fingertip)
    ):
        target_posi = contact_point - finger_base_pose

        target_posi_fc = transform_vector_using_yz_axis(
            target_posi, finger_y_axis, finger_z_axis
        )

        if target_posi_fc[2] < 0:
            is_large_opening = True

        else:
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

                Target_base_plane = np.array(
                    [target_posi_fc[0], target_posi_fc[1], 0.0]
                )

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
                direction_check = np.array(
                    [target_direction[1], -target_direction[0], 0.0]
                )

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

                #   theta_2_s2_intm = cos2ndLaw_angle(link_1, link_x, np.sqrt(joint_0_to_tip_base_length**2 + link_x**2 - joint_1_to_tip_plane_length_sq))
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

                joint_1_posi = np.array(
                    [-link_1 * np.cos(theta_0_s3), link_1 * np.sin(theta_0_s3), 0.0]
                )

                link_x_vect = target_posi_fc - joint_1_posi
                link_x = np.linalg.norm(link_x_vect)

                link_x = max(min_length, min(link_x, max_length))

                joint_1_x_ndir = np.array(
                    [-np.cos(theta_0_s3), np.sin(theta_0_s3), 0.0]
                )
                joint_1_z_dir = np.array([0.0, 0.0, 1.0])
                joint_1_y_dir = np.array([np.sin(theta_0_s3), np.cos(theta_0_s3), 0.0])

                joint_1_to_tip_base = np.array([link_x_vect[0], link_x_vect[1], 0.0])
                joint_1_to_tip_base_max = min(
                    link_x, np.linalg.norm(joint_1_to_tip_base)
                )

                joint_1_to_tip_base_limit = joint_1_to_tip_base_max * (
                    joint_1_to_tip_base / np.linalg.norm(joint_1_to_tip_base)
                )
                tip_base_height = np.sqrt(link_x**2 - joint_1_to_tip_base_max**2)

                joint_1_to_tip_base_y_dir_length = np.dot(
                    joint_1_to_tip_base_limit, joint_1_y_dir
                )

                theta_1_s3 = np.arctan2(
                    joint_1_to_tip_base_y_dir_length, tip_base_height
                )

                theta_3_s3 = np.pi - cos2ndLaw_angle(link_2, link_3, link_x)

                tip_position = (
                    link_1 * joint_1_x_ndir
                    + joint_1_to_tip_base_limit
                    + tip_base_height * joint_1_z_dir
                )

                theta_2_s3_intm = cos2ndLaw_angle(
                    link_1, link_x, np.linalg.norm(tip_position)
                )
                theta_2_s3_sub = cos2ndLaw_angle(link_x, link_2, link_3)

                theta_2_s3 = np.pi - (theta_2_s3_intm + theta_2_s3_sub)

            joint_angle[0] = theta_0_s3 * 180.0 / np.pi
            joint_angle[1] = theta_1_s3 * 180.0 / np.pi - 90.0
            joint_angle[2] = theta_2_s3 * 180.0 / np.pi
            joint_angle[3] = theta_3_s3 * 180.0 / np.pi

            # if (contact_info.contact_type=='palmar side'):
            #   joint_angle[3] = -joint_angle[3]
            #   joint_intm=cos2ndLaw_angle(link_2,link_x,link_3)
            #   joint_angle[2] = joint_angle[2]+2*joint_intm/np.pi*180

            # joint_angle = [-angle for angle in joint_angle]

    if is_large_opening:
        target_posi = contact_point - finger_base_pose

        target_posi_fc = transform_vector_using_yz_axis(
            target_posi, finger_y_axis, finger_z_axis
        )

        theta_1 = -0.5 * np.pi

        theta_0 = np.arctan2(target_posi_fc[1], -target_posi_fc[0])

        if theta_0 > (125 / 180) * np.pi:
            theta_0 = (125 / 180) * np.pi
            target_posi_fc_norm = np.linalg.norm(target_posi_fc)
            target_posi_fc = [
                -target_posi_fc_norm * np.cos(theta_0),
                target_posi_fc_norm * np.sin(theta_0),
                target_posi_fc[2],
            ]

        elif theta_0 < 0:
            theta_0 = 0
            target_posi_fc_norm = np.linalg.norm(target_posi_fc)
            target_posi_fc = [-target_posi_fc_norm, 0, target_posi_fc[2]]

        j1_posi = (
            finger_base_pose
            + link_1 * (-np.cos(theta_0)) * finger_x_axis
            + link_1 * (np.sin(theta_0)) * finger_y_axis
        )

        j1_to_contact = contact_point - j1_posi

        link_x = np.linalg.norm(j1_to_contact)

        max_length = link_2 + link_3
        min_length = np.sqrt(link_2**2 + link_3**2)

        if min_length <= link_x <= max_length:
            theta_3 = np.pi - cos2ndLaw_angle(link_2, link_3, link_x)

            theta_2_sub = cos2ndLaw_angle(link_x, link_2, link_3)

            j1_to_contact_fc = transform_vector_using_yz_axis(
                j1_to_contact, finger_y_axis, finger_z_axis
            )

            theta_2_intm = np.arctan2(
                -j1_to_contact_fc[2],
                np.sqrt(j1_to_contact_fc[0] ** 2 + j1_to_contact_fc[1] ** 2),
            )

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

            j1_to_contact_fc = transform_vector_using_yz_axis(
                j1_to_contact, finger_y_axis, finger_z_axis
            )

            theta_2_intm = np.arctan2(
                -j1_to_contact_fc[2],
                np.sqrt(j1_to_contact_fc[0] ** 2 + j1_to_contact_fc[1] ** 2),
            )

            theta_2 = -theta_2_sub - theta_2_intm

        joint_angle[0] = theta_0 * 180.0 / np.pi
        joint_angle[1] = theta_1 * 180.0 / np.pi
        joint_angle[2] = theta_2 * 180.0 / np.pi
        joint_angle[3] = theta_3 * 180.0 / np.pi

    return joint_angle, condition, link_3
