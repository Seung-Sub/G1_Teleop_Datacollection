import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return unit‑length vector."""
    n = np.linalg.norm(v)
    return v / max(n, eps)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (w, x, y, z). No batch support."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)

def rotate_vector(v: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
    axis_norm = np.float32(np.linalg.norm(axis))
    if axis_norm > 0.0:
        axis = axis / axis_norm

    cos_a = np.float32(np.cos(angle))
    sin_a = np.float32(np.sin(angle))

    cross = np.array([
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0]
    ], dtype=np.float32)

    dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2]

    one = np.float32(1.0)
    rotated = (
        v * cos_a +
        cross * sin_a +
        axis * dot * (one - cos_a)
    )
    return rotated


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Converts a single 3x3 rotation matrix to a quaternion (w, x, y, z).

    Args:
        R (np.ndarray): A 3x3 rotation matrix

    Returns:
        np.ndarray: A quaternion in the format [w, x, y, z] with shape (4,)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    quat = np.zeros(4, dtype=np.float32)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    quat[:] = [w, x, y, z]
    norm_q = np.linalg.norm(quat)
    if norm_q > 0.0:
        quat /= norm_q

    return quat


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Converts a single quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Args:
        quat (np.ndarray): A quaternion [w, x, y, z] (shape: (4,))

    Returns:
        np.ndarray: A 3x3 rotation matrix
    """
    w, x, y, z = quat
    R = np.empty((3, 3), dtype=np.float32)

    R[0, 0] = 1 - 2 * (y**2 + z**2)
    R[0, 1] = 2 * (x * y - z * w)
    R[0, 2] = 2 * (x * z + y * w)
    R[1, 0] = 2 * (x * y + z * w)
    R[1, 1] = 1 - 2 * (x**2 + z**2)
    R[1, 2] = 2 * (y * z - x * w)
    R[2, 0] = 2 * (x * z - y * w)
    R[2, 1] = 2 * (y * z + x * w)
    R[2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product q = q1 * q2 (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.array([w, x, y, z], dtype=q1.dtype)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of quaternion (normalized output)."""
    return normalize(quat_conjugate(q))

def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return unit‑length quaternion."""
    n = np.linalg.norm(q)
    return q if n < eps else q / n


def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to axis‑angle (θ * n).
    Matches the logic of the Torch implementation.
    """
    q = quat.copy()
    if q[0] < 0.0:            # ensure shortest rotation (positive w)
        q *= -1.0

    w, x, y, z = q
    mag = np.linalg.norm([x, y, z])
    half_angle = np.arctan2(mag, w)
    angle = 2.0 * half_angle

    if abs(angle) > eps:
        scale = np.sin(half_angle) / angle
    else:
        scale = 0.5 - angle * angle / 48.0  # Taylor approximation

    if scale < 1e-8:
        return np.zeros(3, dtype=q.dtype)

    return np.array([x, y, z], dtype=q.dtype) / scale            # (3,)


def cos2ndLaw_angle(side_length_1: float, side_length_2: float, far_length: float) -> float:
    """
    Given three sides (side_length_1, side_length_2, far_length),
    returns the angle (in radians) opposite to 'far_length' using the law of cosines:
      cos(angle) = (a^2 + b^2 - c^2) / (2ab).
    """
    data = (side_length_1**2 + side_length_2**2 - far_length**2) / (2.0 * side_length_1 * side_length_2)
    data_clamped = np.clip(data, -1.0, 1.0)
    return np.arccos(data_clamped)

def cos2ndLaw_length(side_length_1: float, side_length_2: float, angle_in_rad: float) -> float:
    """
    Returns the third side (opposite the given angle_in_rad) of a triangle with known side_length_1 and side_length_2
    using the law of cosines:
      c^2 = a^2 + b^2 - 2ab cos(angle)
    """
    return np.sqrt(side_length_1**2 + side_length_2**2 - 2.0 * side_length_1 * side_length_2 * np.cos(angle_in_rad))

def transform_vector(
    vector: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray
) -> np.ndarray:
    """
    Projects 'vector' onto the directions x_axis, y_axis, z_axis and returns the result as a 3D vector.
    
    Equivalent to the batch version but for a single sample:
      [ dot(vector, x_axis), dot(vector, y_axis), dot(vector, z_axis) ]
    """
    return np.array([
        np.dot(vector, x_axis),
        np.dot(vector, y_axis),
        np.dot(vector, z_axis)
    ], dtype=np.float64)

def transform_vector_using_yz_axis(
    vector: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
    """
    Projects `vector` onto a right-handed frame whose Y/Z axes are given.

    X-axis is reconstructed as  X = Y × Z  (so the frame keeps
    a right-handed orientation).  All axes are normalised first.

    Returns
    -------
    np.ndarray, shape (3,)
        [ dot(v, X), dot(v, Y), dot(v, Z) ]  in float64 precision
    """
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)

    return np.array(
        [np.dot(vector, x_axis), np.dot(vector, y_axis), np.dot(vector, z_axis)],
        dtype=np.float64,
    )
