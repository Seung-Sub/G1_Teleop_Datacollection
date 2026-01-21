import numpy as np
from typing import Literal

from .geometry_utils_np import quat_mul, axis_angle_from_quat, quat_conjugate


def compute_pose_error(
    t01: np.ndarray,                      # (3,)  source position
    q01: np.ndarray,                      # (4,)  source quat (w, x, y, z)
    t02: np.ndarray,                      # (3,)  target position
    q02: np.ndarray,                      # (4,)  target quat (w, x, y, z)
    rot_error_type: Literal["quat", "axis_angle"] = "axis_angle",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Position/orientation error between frames (NumPy, no batch).

    Args:
        t01, q01 : source frame pose
        t02, q02 : target frame pose
        rot_error_type : "quat" → return quaternion error,
                         "axis_angle" → return axis‑angle error

    Returns:
        pos_error   – (3,)
        rot_error   – (4,) if "quat", (3,) if "axis_angle"
    """
    # quaternion error  q_err = q_target * inv(q_source)
    # inv(q_source) = conj(q_source) / ||q_source||²
    q1_conj = quat_conjugate(q01)
    q1_norm_sq = float(np.dot(q01, q01))          # scalar
    q1_inv = q1_conj / q1_norm_sq

    quat_error = quat_mul(q02, q1_inv)            # (4,)

    # position error
    pos_error = t02.astype(np.float32) - t01.astype(np.float32)

    # return in requested form
    if rot_error_type == "quat":
        return pos_error, quat_error.astype(np.float32)
    elif rot_error_type == "axis_angle":
        axis_angle_error = axis_angle_from_quat(quat_error).astype(np.float32)
        return pos_error, axis_angle_error
    else:
        raise ValueError("rot_error_type must be 'quat' or 'axis_angle'.")


def IK_Arm_np(
    jacobian: np.ndarray,        # (6, N)
    joint_pos: np.ndarray,       # (N,)
    cur_ee_pos: np.ndarray,      # (3,)
    cur_ee_quat: np.ndarray,     # (4,)
    des_ee_pos: np.ndarray,      # (3,)
    des_ee_quat: np.ndarray,     # (4,)
    lambda_val: float = 0.01,
) -> np.ndarray:
    # Ensure float32 once
    J = jacobian.astype(np.float32, copy=False)
    q = joint_pos.astype(np.float32, copy=False)

    # 1. Compute error
    pos_err, ori_err = compute_pose_error(
        cur_ee_pos, cur_ee_quat, des_ee_pos, des_ee_quat
    )
    delta_x = np.concatenate((pos_err, ori_err))  # (6,)

    # 2. DLS solve via np.linalg.solve
    JT = J.T                      # (N, 6)
    JJt = J @ JT                  # (6, 6)
    I = np.eye(6, dtype=np.float32)

    A = JJt + (lambda_val ** 2) * I
    b = delta_x

    delta_q = JT @ np.linalg.solve(A, b)  # solve Ax = b

    # 3. Return new joint state
    return q + delta_q