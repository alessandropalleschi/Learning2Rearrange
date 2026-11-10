"""Pose math with PyBullet's conventions (quaternions x, y, z, w; Euler roll, pitch, yaw about the fixed X, Y, Z axes;
rotation matrices as flat row-major 9-tuples), in NumPy. Lets the backend-neutral code (primitives, suction, scene queries)
run on MuJoCo, whose own quaternions are w, x, y, z (`to_mj` / `from_mj`)."""
import numpy as np


def to_mj(q):
    """x, y, z, w -> w, x, y, z."""
    x, y, z, w = q
    return np.array([w, x, y, z], float)


def from_mj(q):
    """w, x, y, z -> x, y, z, w."""
    w, x, y, z = q
    return (float(x), float(y), float(z), float(w))


def quat_multiply(a, b):
    """Hamilton product a * b (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q, v):
    """Rotate the vector v by the unit quaternion q."""
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def quat_from_euler(euler):
    """PyBullet getQuaternionFromEuler: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = euler
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def euler_from_quat(q):
    """PyBullet getEulerFromQuaternion (inverse of `quat_from_euler`)."""
    x, y, z, w = q
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (float(roll), float(np.arcsin(sinp)), float(yaw))


def matrix_from_quat(q):
    """Flat row-major 3x3 rotation matrix (PyBullet getMatrixFromQuaternion)."""
    x, y, z, w = q
    return (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))


def quat_from_matrix(m):
    """Unit quaternion (x, y, z, w) of a 3x3 rotation matrix (any layout NumPy can reshape to 3x3)."""
    m = np.asarray(m, float).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    q = np.asarray(q)
    return tuple(float(v) for v in q / np.linalg.norm(q))


def multiply_transforms(pos_a, orn_a, pos_b, orn_b):
    """(pos, orn) of the pose b expressed in a, composed: T_a * T_b (PyBullet multiplyTransforms)."""
    pos = np.asarray(pos_a, float) + quat_rotate(orn_a, pos_b)
    return tuple(float(v) for v in pos), quat_multiply(orn_a, orn_b)


def invert_transform(pos, orn):
    """Inverse pose (PyBullet invertTransform)."""
    inverse = quat_conjugate(orn)
    return tuple(float(v) for v in -quat_rotate(inverse, pos)), inverse
