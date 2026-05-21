"""Translational flyby dynamics nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from syssim.core import EmptySpec, InputPort, NodeDifferential, OutputPort, input_port, output_port


@dataclass
class NodeHyperbolicDynamicsInputs:
    """Input ports for hyperbolic translational dynamics."""

    gravity_accel: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Asteroid gravity acceleration in inertial/body-aligned frame [m/s^2]."""


@dataclass
class NodeHyperbolicDynamicsOutputs:
    """Output ports for hyperbolic translational dynamics."""

    position: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft position relative to asteroid center [m]."""
    velocity: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft velocity relative to asteroid center [m/s]."""
    gravity_error: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Difference between spherical-harmonic gravity and point-mass gravity [m/s^2]."""


class NodeHyperbolicDynamics(
    NodeDifferential[np.ndarray, NodeHyperbolicDynamicsInputs, NodeHyperbolicDynamicsOutputs, EmptySpec, EmptySpec]
):
    """Translational dynamics with scipy RK45 integration."""

    Inputs = NodeHyperbolicDynamicsInputs
    Outputs = NodeHyperbolicDynamicsOutputs

    def __init__(self, x0: np.ndarray, mu: float, **kwargs):
        self._mu = mu
        super().__init__(np.asarray(x0, dtype=float), **kwargs)
        self._i = self.i
        self._o = self.o

    def initialize(self):
        self._t = 0.0
        self._last_gravity_accel = np.zeros(3, dtype=float)
        self._last_gravity_error = np.zeros(3, dtype=float)
        self.reset_state()

    def reset_state(self, x0: np.ndarray | None = None, sim_time: float = 0.0):
        self.state = np.array(self.initial_state if x0 is None else x0, dtype=float)
        self._t = float(sim_time)
        self._last_gravity_accel = np.zeros(3, dtype=float)
        self._last_gravity_error = np.zeros(3, dtype=float)

    def update(self, sim_time: float):
        dt = sim_time - self._t
        if dt <= 0.0:
            return

        acc_grav = self.i.gravity_accel.read().value
        if acc_grav is None or np.any(np.isnan(acc_grav)):
            acc_grav = np.zeros(3, dtype=float)
        acc_grav = np.asarray(acc_grav, dtype=float)
        self._last_gravity_accel = acc_grav.copy()

        r = self.state[0:3]
        r_norm = np.linalg.norm(r)
        acc_kepler = -self._mu / r_norm**3 * r if r_norm > 1e-6 else np.zeros(3, dtype=float)
        gravity_error = acc_grav - acc_kepler
        self._last_gravity_error = gravity_error.copy()

        def rhs(t, state):
            del t
            return np.concatenate([state[3:6], acc_grav])

        result = solve_ivp(rhs, [self._t, sim_time], self.state, method="RK45", dense_output=False, max_step=dt)
        if result.status == 0:
            self.state = result.y[:, -1]
        else:
            v = self.state[3:6] + acc_grav * dt
            r = self.state[0:3] + v * dt
            self.state = np.concatenate([r, v])

        self._t = sim_time
        self.o.position.write(self.state[0:3], sim_time)
        self.o.velocity.write(self.state[3:6], sim_time)
        self.o.gravity_error.write(gravity_error, sim_time)
