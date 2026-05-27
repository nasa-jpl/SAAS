"""Reaction wheel torque allocation and aggregation nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from syssim.core import EmptySpec, InputPort, Node, NodeDifferential, OutputPort, input_port, output_port

from .common import TORQUE_ALLOCATION_MATRIX, normalize


@dataclass
class NodeTorqueAllocatorInputs:
    """Input ports for wheel torque allocation."""

    tau_cmd_body: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Commanded spacecraft body torque [N m]."""


@dataclass
class NodeTorqueAllocatorOutputs:
    """Output ports for wheel torque allocation."""

    tau_cmd_wheel: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(4,))
    """Four-wheel scalar torque command vector [N m]."""


class NodeTorqueAllocator(Node[NodeTorqueAllocatorInputs, NodeTorqueAllocatorOutputs, EmptySpec, EmptySpec]):
    """Allocate body-frame torque command to four tetrahedral wheels."""

    Inputs = NodeTorqueAllocatorInputs
    Outputs = NodeTorqueAllocatorOutputs

    def __init__(self, **kwargs):
        """Initialize the torque allocator node.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to ``Node``.
        """
        super().__init__(**kwargs)
        self._i = self.i
        self._o = self.o

    def update(self, sim_time: float):
        """Allocate a body torque command to tetrahedral wheel axes.

        Parameters
        ----------
        sim_time : float
            Current simulation time [s].
        """
        tau_cmd_body = self.i.tau_cmd_body.read().value
        if tau_cmd_body is None:
            tau_cmd_body = np.zeros(3, dtype=float)
        self.o.tau_cmd_wheel.write(-TORQUE_ALLOCATION_MATRIX @ tau_cmd_body, sim_time)


@dataclass
class NodeReactionWheelInputs:
    """Input ports for a single reaction wheel."""

    tau_cmd: InputPort[np.ndarray] = input_port(np.ndarray)
    """Wheel torque command vector or scalar command [N m]."""


@dataclass
class NodeReactionWheelOutputs:
    """Output ports for a single reaction wheel."""

    h_rw: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel angular momentum contribution in body coordinates [N m s]."""
    tau_rw: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Body torque from the wheel including disturbance terms [N m]."""
    omega: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(1,))
    """Wheel angular speed [rad/s]."""
    wheel_torque: OutputPort[float] = output_port(float)
    """Effective scalar motor torque after lag and saturation [N m]."""


class NodeReactionWheel(
    NodeDifferential[np.ndarray, NodeReactionWheelInputs, NodeReactionWheelOutputs, EmptySpec, EmptySpec]
):
    """Single reaction wheel with saturation, lag, friction, and disturbances."""

    Inputs = NodeReactionWheelInputs
    Outputs = NodeReactionWheelOutputs

    def __init__(self, wheel_idx: int, cfg: Any, wheel_axis: np.ndarray, **kwargs):
        """Initialize a single reaction wheel model.

        Parameters
        ----------
        wheel_idx : int
            Index of the wheel within the tetrahedral wheel set.
        cfg : Any
            Configuration object with wheel inertia, limits, friction, lag, and jitter fields.
        wheel_axis : np.ndarray
            Wheel spin axis expressed in body coordinates.
        **kwargs
            Additional keyword arguments forwarded to ``NodeDifferential``.
        """
        self._wheel_idx = wheel_idx
        self._cfg = cfg
        self._wheel_axis = normalize(wheel_axis)
        self._j = cfg.wheel_inertia_kgm2[0]
        self._tau_max = cfg.torque_max_nm[0]
        self._omega_max = cfg.wheel_speed_max_rads[0]
        self._h_max = cfg.momentum_max_nms[0]
        super().__init__(np.array([0.0], dtype=float), **kwargs)
        self._i = self.i
        self._o = self.o

    def initialize(self):
        """Reset wheel speed, command lag, and deterministic noise state."""
        self._t = 0.0
        self._tau_cmd_lagged = 0.0
        self._rng = np.random.default_rng(7 + self._wheel_idx)
        self.reset_state()

    def reset_state(self, omega_rads: float = 0.0, sim_time: float = 0.0):
        """Reset wheel angular speed and internal lag/noise state.

        Parameters
        ----------
        omega_rads : float, optional
            Initial wheel angular speed [rad/s].
        sim_time : float, optional
            Simulation time associated with the reset state [s].
        """
        self.state = np.array([omega_rads], dtype=float)
        self._t = float(sim_time)
        self._tau_cmd_lagged = 0.0
        self._rng = np.random.default_rng(7 + self._wheel_idx)

    def update(self, sim_time: float):
        """Advance wheel speed and write momentum and torque outputs.

        Parameters
        ----------
        sim_time : float
            Current simulation time [s].
        """
        dt = sim_time - self._t
        if dt <= 0.0:
            return

        tau_cmd = self.i.tau_cmd.read().value
        if tau_cmd is None:
            tau_cmd = np.array([0.0])
        tau_cmd_scalar = float(tau_cmd[self._wheel_idx]) if isinstance(tau_cmd, np.ndarray) and tau_cmd.size > 1 else float(tau_cmd)
        tau_cmd_scalar = np.clip(tau_cmd_scalar, -self._tau_max, self._tau_max)

        alpha = np.exp(-dt / max(self._cfg.command_lag_tau_s, 1e-6))
        self._tau_cmd_lagged = alpha * self._tau_cmd_lagged + (1.0 - alpha) * tau_cmd_scalar

        omega = self.state[0]
        tau_visc = self._cfg.viscous_friction_nms * omega
        tau_coul = self._cfg.coulomb_friction_nm * np.tanh(omega / 0.01)
        tau_fric = tau_visc + tau_coul
        imbalance = self._cfg.imbalance_coeff_nm_per_rads2 * (omega * omega) * np.sin(
            2.0 * np.pi * self._cfg.imbalance_freq_hz * sim_time
        )
        jitter = self._rng.normal(0.0, self._cfg.jitter_std_nm)

        domega = (self._tau_cmd_lagged - tau_fric) / max(self._j, 1e-9)
        omega_new = omega + domega * dt
        omega_limit_from_h = self._h_max / max(self._j, 1e-9)
        omega_new = np.clip(omega_new, -min(self._omega_max, omega_limit_from_h), min(self._omega_max, omega_limit_from_h))

        effective_domega = (omega_new - omega) / max(dt, 1e-9)
        tau_rw_wheel = self._j * effective_domega
        h_rw_body = self._j * omega_new * self._wheel_axis
        tau_rw_body = -tau_rw_wheel * self._wheel_axis - (imbalance + jitter) * self._wheel_axis

        self.state = np.array([omega_new], dtype=float)
        self._t = sim_time

        self.o.h_rw.write(h_rw_body, sim_time)
        self.o.tau_rw.write(tau_rw_body, sim_time)
        self.o.omega.write(np.array([omega_new], dtype=float), sim_time)
        self.o.wheel_torque.write(float(tau_rw_wheel), sim_time)


@dataclass
class NodeWheelAggregatorInputs:
    """Input ports for four-wheel momentum and torque aggregation."""

    h_rw_0: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel 0 angular momentum in body coordinates [N m s]."""
    h_rw_1: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel 1 angular momentum in body coordinates [N m s]."""
    h_rw_2: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel 2 angular momentum in body coordinates [N m s]."""
    h_rw_3: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel 3 angular momentum in body coordinates [N m s]."""
    tau_rw_0: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel 0 body torque [N m]."""
    tau_rw_1: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel 1 body torque [N m]."""
    tau_rw_2: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel 2 body torque [N m]."""
    tau_rw_3: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Wheel 3 body torque [N m]."""


@dataclass
class NodeWheelAggregatorOutputs:
    """Output ports for total reaction-wheel effects."""

    h_rw_total: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Total reaction-wheel angular momentum in body coordinates [N m s]."""
    tau_rw_total: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Total reaction-wheel body torque [N m]."""


class NodeWheelAggregator(Node[NodeWheelAggregatorInputs, NodeWheelAggregatorOutputs, EmptySpec, EmptySpec]):
    """Sum momentum and torque from all four wheels."""

    Inputs = NodeWheelAggregatorInputs
    Outputs = NodeWheelAggregatorOutputs

    def __init__(self, **kwargs):
        """Initialize the wheel aggregation node.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to ``Node``.
        """
        super().__init__(**kwargs)
        self._i = self.i
        self._o = self.o

    def update(self, sim_time: float):
        """Sum wheel momentum and torque contributions.

        Parameters
        ----------
        sim_time : float
            Current simulation time [s].
        """
        h_rw = np.zeros(3, dtype=float)
        tau_rw = np.zeros(3, dtype=float)
        for idx in range(4):
            h = getattr(self.i, f"h_rw_{idx}").read().value
            tau = getattr(self.i, f"tau_rw_{idx}").read().value
            if h is not None:
                h_rw += h
            if tau is not None:
                tau_rw += tau
        self.o.h_rw_total.write(h_rw, sim_time)
        self.o.tau_rw_total.write(tau_rw, sim_time)
