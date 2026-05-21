"""Shared math helpers for asteroid flyby syssim nodes."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


TETRAHEDRAL_WHEEL_AXES = np.array(
    [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ],
    dtype=float,
)
for row in TETRAHEDRAL_WHEEL_AXES:
    row /= np.linalg.norm(row)

TORQUE_ALLOCATION_MATRIX = np.linalg.pinv(TETRAHEDRAL_WHEEL_AXES.T)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def normalize_quaternion_wxyz(q_wxyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float)
    if q.shape != (4,):
        q = q.reshape(4)
    if not np.all(np.isfinite(q)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=float)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=float)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def rotation_from_wxyz(q_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(quat_wxyz_to_xyzw(normalize_quaternion_wxyz(q_wxyz)))


def wxyz_from_rotation(rot: Rotation) -> np.ndarray:
    return quat_xyzw_to_wxyz(rot.as_quat())


def quat_conjugate_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=float)
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]], dtype=float)


def quat_multiply_wxyz(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a_wxyz, dtype=float)
    bw, bx, by, bz = np.asarray(b_wxyz, dtype=float)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def error_quaternion_wxyz(q_bi_wxyz: np.ndarray, q_cmd_bi_wxyz: np.ndarray) -> np.ndarray:
    q_bi_wxyz = normalize(np.asarray(q_bi_wxyz, dtype=float))
    q_cmd_bi_wxyz = normalize(np.asarray(q_cmd_bi_wxyz, dtype=float))
    q_cmd_inv = quat_conjugate_wxyz(q_cmd_bi_wxyz)
    return normalize(quat_multiply_wxyz(q_cmd_inv, q_bi_wxyz))


def center_pointing_quaternion_wxyz(
    position_sc_i_m: np.ndarray,
    velocity_sc_i_mps: np.ndarray | None = None,
    prev_q_cmd_bi_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    x_b_i = normalize(-position_sc_i_m)
    if np.linalg.norm(x_b_i) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    y_ref_i = None
    if prev_q_cmd_bi_wxyz is not None:
        prev_r_bi = rotation_from_wxyz(prev_q_cmd_bi_wxyz)
        prev_y_i = prev_r_bi.apply(np.array([0.0, 1.0, 0.0], dtype=float))
        y_ref_i = prev_y_i - np.dot(prev_y_i, x_b_i) * x_b_i

    if (y_ref_i is None or np.linalg.norm(y_ref_i) < 1e-9) and velocity_sc_i_mps is not None:
        h_orbit_i = np.cross(position_sc_i_m, velocity_sc_i_mps)
        if np.linalg.norm(h_orbit_i) > 1e-9:
            y_ref_i = np.cross(h_orbit_i, x_b_i)

    if y_ref_i is None or np.linalg.norm(y_ref_i) < 1e-9:
        z_ref_i = np.array([0.0, 0.0, 1.0], dtype=float)
        y_ref_i = np.cross(z_ref_i, x_b_i)
        if np.linalg.norm(y_ref_i) < 1e-9:
            y_ref_i = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), x_b_i)

    y_b_i = normalize(y_ref_i)
    z_b_i = normalize(np.cross(x_b_i, y_b_i))
    y_b_i = normalize(np.cross(z_b_i, x_b_i))

    q_cmd = wxyz_from_rotation(Rotation.from_matrix(np.column_stack([x_b_i, y_b_i, z_b_i])))
    if prev_q_cmd_bi_wxyz is not None and np.dot(q_cmd, prev_q_cmd_bi_wxyz) < 0.0:
        q_cmd = -q_cmd
    if prev_q_cmd_bi_wxyz is not None:
        prev_rot = rotation_from_wxyz(prev_q_cmd_bi_wxyz)
        cmd_rot = rotation_from_wxyz(q_cmd)
        delta = cmd_rot * prev_rot.inv()
        max_step = np.deg2rad(55.0)
        angle = delta.magnitude()
        if angle > max_step:
            limited_rot = Rotation.from_rotvec(delta.as_rotvec() * (max_step / angle)) * prev_rot
            q_cmd = wxyz_from_rotation(limited_rot)
    return q_cmd
