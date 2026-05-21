"""Attitude control nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from syssim.core import EmptySpec, InputPort, NodeDifferential, OutputPort, input_port, output_port

from .common import error_quaternion_wxyz


@dataclass
class NodeAttitudeControllerInputs:
    """Input ports for quaternion feedback attitude control."""

    q_cmd: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(4,))
    """Commanded body-to-inertial quaternion [w, x, y, z]."""
    q: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(4,))
    """Measured or simulated body-to-inertial quaternion [w, x, y, z]."""
    w_cmd: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Commanded body angular velocity [rad/s]."""
    w: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Measured or simulated body angular velocity [rad/s]."""
    h_rw: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Total reaction-wheel angular momentum in body coordinates [N m s]."""


@dataclass
class NodeAttitudeControllerOutputs:
    """Output ports for quaternion feedback attitude control."""

    tau_cmd_body: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Commanded body torque [N m]."""
    q_err: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Vector part of the attitude error quaternion."""
    w_err: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Body angular-rate tracking error [rad/s]."""


class NodeAttitudeController(
    NodeDifferential[np.ndarray, NodeAttitudeControllerInputs, NodeAttitudeControllerOutputs, EmptySpec, EmptySpec]
):
    """Quaternion feedback controller following Wie et al. (1989) Eq. (9)."""

    Inputs = NodeAttitudeControllerInputs
    Outputs = NodeAttitudeControllerOutputs

    def __init__(
        self,
        kp: float,
        kd: float,
        ki: float,
        integral_limit: float,
        inertia_kgm2: tuple[float, float, float],
        **kwargs,
    ):
        self._k_scalar = kp
        self._d_scalar = kd
        self._ki = ki
        self._integral_limit = integral_limit
        self._inertia_mat = np.diag(np.asarray(inertia_kgm2, dtype=float))
        self._k_mat = self._k_scalar * self._inertia_mat
        self._d_mat = self._d_scalar * self._inertia_mat
        super().__init__(np.zeros(3, dtype=float), **kwargs)
        self._i = self.i
        self._o = self.o

    def initialize(self):
        self._t = 0.0
        self.reset_state()

    def reset_state(self, sim_time: float = 0.0):
        self.state = np.zeros(3, dtype=float)
        self._t = float(sim_time)

    def update(self, sim_time: float):
        q_cmd_bi = self.i.q_cmd.read().value
        q_bi = self.i.q.read().value
        w_cmd_b_rps = self.i.w_cmd.read().value
        w_b_rps = self.i.w.read().value
        h_rw_b_nms = self.i.h_rw.read().value

        if q_cmd_bi is None:
            q_cmd_bi = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if q_bi is None:
            q_bi = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if w_cmd_b_rps is None:
            w_cmd_b_rps = np.zeros(3, dtype=float)
        if w_b_rps is None:
            w_b_rps = np.zeros(3, dtype=float)
        if h_rw_b_nms is None:
            h_rw_b_nms = np.zeros(3, dtype=float)

        q_err = error_quaternion_wxyz(q_bi_wxyz=q_bi, q_cmd_bi_wxyz=q_cmd_bi)
        q_err_vec = q_err[1:4]
        q_err_scalar = q_err[0]
        w_err_b_rps = w_b_rps - w_cmd_b_rps

        dt = sim_time - self._t
        if dt > 0.0:
            self.state = np.clip(self.state + q_err_vec * dt, -self._integral_limit, self._integral_limit)
        self._t = sim_time

        sign_shortest = 1.0 if q_err_scalar >= 0.0 else -1.0
        gyro_term_b_nm = np.cross(w_b_rps, self._inertia_mat @ w_b_rps + h_rw_b_nms)
        tau_cmd_b_nm = (
            -gyro_term_b_nm
            - (self._d_mat @ w_err_b_rps)
            - sign_shortest * (self._k_mat @ q_err_vec)
            - self._ki * self.state
        )

        self.o.tau_cmd_body.write(tau_cmd_b_nm, sim_time)
        self.o.q_err.write(q_err_vec, sim_time)
        self.o.w_err.write(w_err_b_rps, sim_time)
