"""Rigid-body attitude dynamics nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from syssim.core import EmptySpec, InputPort, NodeDifferential, OutputPort, input_port, output_port

from .common import normalize_quaternion_wxyz, rotation_from_wxyz, wxyz_from_rotation


@dataclass
class NodeAttitudeDynamicsInputs:
    """Input ports for rigid-body attitude dynamics."""

    tau_body: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """External body torque applied to the spacecraft [N m]."""
    h_rw_body: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Reaction-wheel angular momentum in body coordinates [N m s]."""


@dataclass
class NodeAttitudeDynamicsOutputs:
    """Output ports for rigid-body attitude dynamics."""

    q: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(4,))
    """Body-to-inertial attitude quaternion in scalar-first order [w, x, y, z]."""
    w: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Body angular velocity [rad/s]."""


class NodeAttitudeDynamics(
    NodeDifferential[np.ndarray, NodeAttitudeDynamicsInputs, NodeAttitudeDynamicsOutputs, EmptySpec, EmptySpec]
):
    """Rigid-body spacecraft attitude dynamics with reaction-wheel coupling."""

    Inputs = NodeAttitudeDynamicsInputs
    Outputs = NodeAttitudeDynamicsOutputs

    def __init__(self, inertia_kgm2: tuple[float, float, float], x0: np.ndarray, **kwargs):
        """Initialize the attitude dynamics node.

        Parameters
        ----------
        inertia_kgm2 : tuple[float, float, float]
            Principal moments of inertia about the body axes [kg m^2].
        x0 : np.ndarray
            Initial state ``[qw, qx, qy, qz, wx, wy, wz]``.
        **kwargs
            Additional keyword arguments forwarded to ``NodeDifferential``.
        """
        self._j = np.diag(np.array(inertia_kgm2, dtype=float))
        self._j_inv = np.linalg.inv(self._j)
        super().__init__(np.asarray(x0, dtype=float), **kwargs)
        self._i = self.i
        self._o = self.o

    def initialize(self):
        """Reset simulation time and state before a run."""
        self._t = 0.0
        self.reset_state()

    def reset_state(self, x0: np.ndarray | None = None, sim_time: float = 0.0):
        """Reset the integrated attitude state.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Replacement state ``[qw, qx, qy, qz, wx, wy, wz]``.
        sim_time : float, optional
            Simulation time associated with the reset state [s].
        """
        self.state = np.array(self.initial_state if x0 is None else x0, dtype=float)
        self._t = float(sim_time)

    def update(self, sim_time: float):
        """Advance attitude dynamics to the requested simulation time.

        Parameters
        ----------
        sim_time : float
            Current simulation time [s].
        """
        dt = sim_time - self._t
        if dt <= 0.0:
            return

        tau = self.i.tau_body.read().value
        h_rw = self.i.h_rw_body.read().value
        if tau is None:
            tau = np.zeros(3, dtype=float)
        if h_rw is None:
            h_rw = np.zeros(3, dtype=float)

        q = normalize_quaternion_wxyz(self.state[0:4])
        w = np.asarray(self.state[4:7], dtype=float)
        if not np.all(np.isfinite(w)):
            w = np.zeros(3, dtype=float)

        tau = np.asarray(tau, dtype=float)
        h_rw = np.asarray(h_rw, dtype=float)
        if not np.all(np.isfinite(tau)):
            tau = np.zeros(3, dtype=float)
        if not np.all(np.isfinite(h_rw)):
            h_rw = np.zeros(3, dtype=float)

        ang_acc = self._j_inv @ (tau - np.cross(w, self._j @ w + h_rw))
        w_new = w + ang_acc * dt

        r_bi = rotation_from_wxyz(q)
        delta_r = Rotation.from_rotvec(w_new * dt)
        q_new = normalize_quaternion_wxyz(wxyz_from_rotation(r_bi * delta_r))

        self.state = np.concatenate([q_new, w_new])
        self._t = sim_time

        self.o.q.write(q_new, sim_time)
        self.o.w.write(w_new, sim_time)
