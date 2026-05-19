import numpy as np

def mat_update(prev_mat, mat):
    if np.linalg.det(mat) == 0:
        return prev_mat, False # Return previous matrix and False flag if the new matrix is non-singular (determinant ≠ 0).
    else:
        return mat, True


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def cosine_ease(alpha: float) -> float:
    """Cosine ease-in-out: 0→1 smooth S-curve. alpha in [0, 1]."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * a))


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp quaternions in xyzw form. Returns unit-quaternion.
    Uses shortest-path (negate q1 if dot < 0) and falls back to nlerp for
    very small angles (numerically safe)."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d  = -d
    if d > 0.9995:
        q = (1.0 - t) * q0 + t * q1
        return q / (np.linalg.norm(q) + 1e-12)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(theta) + 1e-12
    w0 = np.sin((1.0 - t) * theta) / s
    w1 = np.sin(t * theta) / s
    return w0 * q0 + w1 * q1


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion [x, y, z, w]. Branch-stable form
    (Shepperd / Shoemake). Avoids the scipy import in tight loops."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.empty((3, 3))
    R[0, 0] = 1.0 - 2.0 * (yy + zz); R[0, 1] = 2.0 * (xy - wz);     R[0, 2] = 2.0 * (xz + wy)
    R[1, 0] = 2.0 * (xy + wz);       R[1, 1] = 1.0 - 2.0 * (xx + zz); R[1, 2] = 2.0 * (yz - wx)
    R[2, 0] = 2.0 * (xz - wy);       R[2, 1] = 2.0 * (yz + wx);     R[2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def se3_interp(T_from: np.ndarray, T_to: np.ndarray, alpha: float) -> np.ndarray:
    """SE(3) interpolation. Translation: linear. Rotation: quaternion slerp.
    alpha in [0, 1]; alpha=0 → T_from, alpha=1 → T_to. 50Hz tight-loop safe
    (no scipy import per call)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    q0 = _mat_to_quat_xyzw(T_from[:3, :3])
    q1 = _mat_to_quat_xyzw(T_to[:3, :3])
    q  = _quat_slerp(q0, q1, a)
    R  = _quat_xyzw_to_mat(q)
    p  = (1.0 - a) * T_from[:3, 3] + a * T_to[:3, 3]
    T  = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = p
    return T