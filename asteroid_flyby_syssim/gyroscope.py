"""Three-axis rate gyroscope sensor node.

This module implements a realistic MEMS gyroscope sensor that measures
angular velocity with bias, scale factors, white noise, and bias random walk
(Allan noise). Characteristics are configurable and can be faulted.

References:
    - Basilisk IMU documentation: https://avslab.github.io/basilisk/
    - Allan variance: IEEE 1293-1998, "IEEE Standard for Processes and
      Equipment for Processing Printed Circuit Boards"
"""

from typing import NamedTuple, Optional
from dataclasses import dataclass

import numpy as np
from syssim.core import Node, InputPort, OutputPort
from syssim.core.node import NodeParameter


class NodeGyroscopeInputs(NamedTuple):
    """Input ports for gyroscope node."""

    angular_velocity: InputPort
    """True angular velocity [rad/s] in body frame, shape (3,)"""


class NodeGyroscopeOutputs(NamedTuple):
    """Output ports for gyroscope node."""

    measurement: OutputPort
    """Measured angular velocity [rad/s] with noise/bias, shape (3,)"""


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
    scale_errors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    white_noise_std_rad_s: float = 1e-4
    bias_random_walk_std_rad_s2: float = 1e-6
    sample_rate_hz: float = 100.0
    rng_seed: Optional[int] = None


class NodeGyroscope(Node):
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
            Forwarded to parent Node class (e.g., sample_frequency, config file path).
        """
        if config is None:
            config = GyroscopeConfig()
        self._config = config

        # Save config values before calling super().__init__() because the
        # syssim Node base class may overwrite self._config with a dict
        # loaded from a TOML config file.
        self._bias_rad_s = tuple(config.bias_rad_s)
        self._scale_errors = tuple(config.scale_errors)
        self._white_noise_std_rad_s = config.white_noise_std_rad_s
        self._bias_random_walk_std_rad_s2 = config.bias_random_walk_std_rad_s2
        self._sample_rate_hz = config.sample_rate_hz
        self._rng_seed = config.rng_seed

        # Define input/output ports
        self._i = NodeGyroscopeInputs(InputPort("angular_velocity", self))
        self._o = NodeGyroscopeOutputs(OutputPort("measurement", self))

        # Create faultable parameters for bias and scale errors
        self._param_bias = NodeParameter(
            "bias_rad_s", np.array(config.bias_rad_s, dtype=float)
        )
        self._param_scale = NodeParameter(
            "scale_errors", np.array(config.scale_errors, dtype=float)
        )

        # Initialize parent Node
        super().__init__(self._i, self._o, parameters=(self._param_bias, self._param_scale), name=name, **kwargs)

    def initialize(self):
        """Initialize gyroscope state before simulation.

        Sets up:
        - Random number generator with optional seed
        - Bias drift state (random walk accumulation)
        """
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
        w_true = self._i.angular_velocity.read()
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
        self._o.measurement.shift_out(w_measured, sim_time)

    @property
    def i(self) -> NodeGyroscopeInputs:
        """Input ports."""
        return self._i

    @property
    def o(self) -> NodeGyroscopeOutputs:
        """Output ports."""
        return self._o
