"""Three-axis rate gyroscope sensor node.

This module implements a realistic MEMS gyroscope sensor that measures
angular velocity with bias, scale factors, white noise, and bias random walk
(Allan noise). Characteristics are configurable and can be faulted.

References:
    - Basilisk IMU documentation: https://avslab.github.io/basilisk/
    - Allan variance: IEEE 1293-1998, "IEEE Standard for Processes and
      Equipment for Processing Printed Circuit Boards"
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from syssim.core import InputPort, Node, NodeParameter, OutputPort, input_port, output_port, parameter


@dataclass
class NodeGyroscopeInputs:
    """Input ports for gyroscope node."""

    angular_velocity: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """True angular velocity [rad/s] in body frame, shape (3,)"""


@dataclass
class NodeGyroscopeOutputs:
    """Output ports for gyroscope node."""

    measurement: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Measured angular velocity [rad/s] with noise/bias, shape (3,)"""


@dataclass
class NodeGyroscopeParameters:
    """Faultable gyroscope calibration parameters."""

    bias_rad_s: NodeParameter[np.ndarray] = parameter(default_factory=lambda: np.zeros(3), value_type=np.ndarray, dtype=float, shape=(3,))
    """Additive gyroscope bias [rad/s] for x, y, z body axes."""
    scale_errors: NodeParameter[np.ndarray] = parameter(default_factory=lambda: np.zeros(3), value_type=np.ndarray, dtype=float, shape=(3,))
    """Multiplicative scale-factor error for x, y, z body axes."""


@dataclass
class GyroscopeConfig:
    """Configuration for gyroscope sensor model.

    Attributes
    ----------
    bias_rad_s : tuple[float, float, float]
        Constant measurement bias (°/s converted to rad/s) for x, y, z axes.
        Typical value: 0.1-1.0 deg/hr = 5e-6 rad/s (very small).
    scale_errors : tuple[float, float, float]
        Scale factor errors (dimensionless, fractional) for each axis.
        E.g., 0.01 means 1% scale error. Typical: 0.01-0.05 (1-5%).
    white_noise_std_rad_s : float
        Standard deviation of white (uncorrelated) noise.
        Typical values:
        - High-grade tactical: ~0.1-0.5 deg/hr = 5e-6 rad/s
        - Consumer MEMS: ~10-100 deg/hr = 1e-4 rad/s
    bias_random_walk_std_rad_s2 : float
        Standard deviation of bias random walk (Brownian motion noise).
        Manifests as slow bias drift over time. Typical: 1e-6 rad/s² for
        tactical-grade gyros.
    sample_rate_hz : float
        Sample rate of the gyroscope in Hz. Default: 100 Hz (0.01 s timestep).
    rng_seed : Optional[int]
        Seed for numpy random number generator. If None, unseeded.
    """

    bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Constant additive bias [rad/s] for x, y, z axes."""
    scale_errors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Fractional scale-factor error for x, y, z axes."""
    white_noise_std_rad_s: float = 1e-4
    """Standard deviation of white angular-rate noise [rad/s]."""
    bias_random_walk_std_rad_s2: float = 1e-6
    """Standard deviation of bias random-walk acceleration [rad/s^2]."""
    sample_rate_hz: float = 100.0
    """Nominal gyroscope sampling rate [Hz]."""
    rng_seed: Optional[int] = None
    """Optional seed for deterministic sensor noise generation."""


class NodeGyroscope(Node[NodeGyroscopeInputs, NodeGyroscopeOutputs, NodeGyroscopeParameters, GyroscopeConfig]):
    """Three-axis MEMS gyroscope sensor node.

    Simulates a real gyroscope by adding realistic error sources:
    - Constant bias (offset from zero)
    - Scale factor errors (gain mismatch)
    - White noise (random uncorrelated noise each sample)
    - Bias random walk (slow drift of bias over time, Allan noise)

    The gyroscope reads true angular velocity from the spacecraft attitude
    dynamics node and outputs a noisy measurement suitable for use in
    attitude estimation or control algorithms.

    Errors are independent per axis. Bias and scale factors can be faulted
    via the NodeParameter system.
    """

    Inputs = NodeGyroscopeInputs
    Outputs = NodeGyroscopeOutputs
    Parameters = NodeGyroscopeParameters
    Config = GyroscopeConfig

    def __init__(
        self,
        config: Optional[GyroscopeConfig] = None,
        name: str = "Gyroscope",
        **kwargs,
    ):
        """Initialize gyroscope sensor node.

        Parameters
        ----------
        config : GyroscopeConfig, optional
            Configuration object with sensor parameters. If None, defaults
            to GyroscopeConfig() with nominal values.
        name : str
            Name of this node (used for configuration file lookup).
        **kwargs
            Forwarded to parent Node class, such as sample period/frequency.
        """
        super().__init__(config=config, name=name, **kwargs)
        self._i = self.i
        self._o = self.o
        self._p = self.p

        self._white_noise_std_rad_s = self.config.white_noise_std_rad_s
        self._bias_random_walk_std_rad_s2 = self.config.bias_random_walk_std_rad_s2
        self._sample_rate_hz = self.config.sample_rate_hz
        self._rng_seed = self.config.rng_seed

        self.p.bias_rad_s.set_nominal(np.array(self.config.bias_rad_s, dtype=float))
        self.p.scale_errors.set_nominal(np.array(self.config.scale_errors, dtype=float))
        self._param_bias = self.p.bias_rad_s
        self._param_scale = self.p.scale_errors

    def initialize(self):
        """Initialize gyroscope state before simulation.

        Sets up:
        - Random number generator with optional seed
        - Bias drift state (random walk accumulation)
        """
        self._rng = np.random.default_rng(self._rng_seed)
        self._bias_drift = np.array([0.0, 0.0, 0.0], dtype=float)

    def reset_state(self):
        self._rng = np.random.default_rng(self._rng_seed)
        self._bias_drift = np.array([0.0, 0.0, 0.0], dtype=float)

    def update(self, sim_time: float):
        """Update gyroscope measurement.

        Reads true angular velocity, applies noise and bias models, outputs measurement.

        Parameters
        ----------
        sim_time : float
            Current simulation time (seconds).
        """
        # Read true angular velocity from input port
        w_true = self.i.angular_velocity.read().value
        if w_true is None:
            # No input connected; assume zero rate
            w_true = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            w_true = np.asarray(w_true, dtype=float)

        # Get current (possibly faulted) parameters
        bias = self._param_bias.value  # shape (3,)
        scale_errors = self._param_scale.value  # shape (3,)

        # Compute scale factors: (1 + scale_error_fraction)
        scale_factors = 1.0 + scale_errors

        # ---- Update bias random walk (Allan noise) ----
        dt = 1.0 / self._sample_rate_hz
        bias_walk_increment = self._rng.normal(
            0.0, self._bias_random_walk_std_rad_s2 * np.sqrt(dt), size=3
        )
        self._bias_drift += bias_walk_increment

        # ---- Compute noisy measurement ----
        w_measured = (w_true * scale_factors) + bias + self._bias_drift

        # Add white noise (uncorrelated per axis)
        white_noise = self._rng.normal(
            0.0, self._white_noise_std_rad_s, size=3
        )
        w_measured += white_noise

        # ---- Output measurement ----
        self.o.measurement.write(w_measured, sim_time)
