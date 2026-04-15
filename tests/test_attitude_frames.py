import numpy as np
from scipy.spatial.transform import Rotation

from asteroid_flyby_syssim.flyby_sim import (
    _attitude_error_vector_body,
    _compute_center_pointing_quaternion_wxyz,
    _error_quaternion_wxyz,
)


def _wxyz_from_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def test_wie_error_sign_restoring_for_positive_z_error():
    """If current attitude has +z error wrt command, error vector should be +z.

    This matches the Wie-style controller sign where tau ~ -Kp*q_err drives the
    state back toward command.
    """
    q_cmd_bi = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q_bi = _wxyz_from_xyzw(Rotation.from_euler("z", 10.0, degrees=True).as_quat())

    q_err = _attitude_error_vector_body(q_bi_wxyz=q_bi, q_cmd_bi_wxyz=q_cmd_bi)

    assert q_err[2] > 0.0
    assert abs(q_err[0]) < 1e-12
    assert abs(q_err[1]) < 1e-12


def test_center_pointing_command_continuity_with_prev_attitude_reference():
    """Consecutive look commands should remain smooth across track progression."""
    prev_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    positions = [
        np.array([2.0e6, 8.0e5, 6.0e5]),
        np.array([1.8e6, 7.0e5, 5.0e5]),
        np.array([1.6e6, 6.0e5, 4.0e5]),
        np.array([1.4e6, 5.0e5, 3.0e5]),
    ]
    velocities = [
        np.array([-200.0, -120.0, -80.0]),
        np.array([-210.0, -118.0, -76.0]),
        np.array([-220.0, -116.0, -72.0]),
        np.array([-230.0, -114.0, -68.0]),
    ]

    max_step_deg = 0.0
    prev_rot = Rotation.from_quat([prev_q[1], prev_q[2], prev_q[3], prev_q[0]])

    for r_i, v_i in zip(positions, velocities):
        q_cmd = _compute_center_pointing_quaternion_wxyz(
            position_sc_i_m=r_i,
            velocity_sc_i_mps=v_i,
            prev_q_cmd_bi_wxyz=prev_q,
        )
        rot = Rotation.from_quat([q_cmd[1], q_cmd[2], q_cmd[3], q_cmd[0]])
        step_deg = np.degrees((rot * prev_rot.inv()).magnitude())
        max_step_deg = max(max_step_deg, float(step_deg))

        prev_q = q_cmd
        prev_rot = rot

    # This threshold is intentionally loose; we just want to catch discontinuous flips.
    assert max_step_deg < 60.0


def test_error_quaternion_scalar_negative_for_long_way_rotation():
    """A >180 deg error should yield negative scalar, enabling shortest-path sign flip."""
    q_cmd_bi = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q_bi = _wxyz_from_xyzw(Rotation.from_euler("z", 200.0, degrees=True).as_quat())

    q_err = _error_quaternion_wxyz(q_bi_wxyz=q_bi, q_cmd_bi_wxyz=q_cmd_bi)

    assert q_err[0] < 0.0
    assert q_err[3] > 0.0
