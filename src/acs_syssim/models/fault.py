from syssim.core.fault import Fault
import numpy as np
import warnings
from typing import Optional, List, Union
from scipy.spatial.transform import Rotation as R

class DiagonalInertiaPerturbFault(Fault):
    """Fault that perturbs each axis of a 3x3 diagonal inertia matrix by a random amount."""
    def __init__(self, name: str, stddev: float = 0.1, seed: int = None, trigger_time: float = 0.0):
        """
        Args:
            name (str): Name of the fault.
            stddev (float): Standard deviation of the random perturbation (fractional, e.g., 0.01 for 1%).
            seed (int, optional): Random seed for reproducibility.
            trigger_time (float): time at which the fault becomes active
        """
        super().__init__(name, trigger_time)
        self.stddev = stddev
        self.rng = np.random.default_rng(seed)
        # Generate random perturbation factors for each axis
        self._perturbation = 1.0 + self.rng.normal(0, self.stddev, size=3)

    def action(self, value: np.ndarray):
        """Perturb the diagonal elements of a 3x3 diagonal inertia matrix by a random amount."""
        if not self.triggered:
            return value
        assert isinstance(value, np.ndarray), "Value must be a numpy ndarray."
        assert value.shape == (3, 3), "Inertia matrix must be 3x3."
        assert np.allclose(value, np.diag(np.diagonal(value))), "Matrix must be diagonal."
        perturbed_diag = np.diagonal(value) * self._perturbation
        return np.diag(perturbed_diag)

class RandomizeFault(Fault):
    """Fault that randomizes the input value within a specified range."""
    def __init__(self, name: str, low: float, high: float, seed: int = None, trigger_time: float = 0.0):
        """
        Args:
            name (str): Name of the fault.
            low (float): Lower bound of the randomization range.
            high (float): Upper bound of the randomization range.
            seed (int, optional): Random seed for reproducibility.
            trigger_time (float): time at which the fault becomes active
        """
        super().__init__(name, trigger_time)
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def action(self, value: np.ndarray):
        """Randomize the input value within the specified range."""
        if not self.triggered:
            return value
        if not isinstance(value, (np.ndarray, float, int)):
            warnings.warn(f"RandomizeFault: value of type {type(value)} is not supported. Returning value unchanged.")
            return value
        if isinstance(value, (float, int)):
            value = np.array([value])
        # Randomize the value
        randomized = self.rng.uniform(self.low, self.high, size=value.shape)
        return randomized

# New fault implementations

class SensorBiasCreep(Fault):
    """Introduces a gradually increasing additive bias to sensor measurements."""
    def __init__(self, name: str, initial_bias: Union[float, np.ndarray] = 0.0,
                 drift_per_call: Union[float, np.ndarray] = 0.0,
                 noise_std: float = 0.0, seed: Optional[int] = None, trigger_time: float = 0.0):
        super().__init__(name, trigger_time)
        self.bias = np.array(initial_bias, dtype=float)
        self.drift = np.array(drift_per_call, dtype=float)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, float, int]):
        if not self.triggered:
            return value
        if not isinstance(value, (np.ndarray, float, int)):
            warnings.warn(f"SensorBiasCreep: unsupported type {type(value)}; returning unchanged.")
            return value
        arr = np.array(value, dtype=float) if isinstance(value, (float, int)) else value.astype(float)
        # Broadcast bias to the input shape if necessary
        bias_to_add = np.broadcast_to(self.bias, arr.shape)
        out = arr + bias_to_add
        # Update the bias (creep) for future calls
        noise = self.rng.normal(0.0, self.noise_std, size=np.shape(self.bias)) if self.noise_std > 0 else np.zeros_like(self.bias)
        self.bias = self.bias + self.drift + noise
        return out

class RandomSensorNoise(Fault):
    """Adds zero-mean Gaussian noise to sensor measurements."""
    def __init__(self, name: str, stddev: float, trigger_time: float, seed: Optional[int] = None):
        super().__init__(name, trigger_time)
        self.stddev = float(stddev)
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, float, int]):
        if not self.triggered:
            return value
        if not isinstance(value, (np.ndarray, float, int)):
            warnings.warn(f"RandomSensorNoise: unsupported type {type(value)}; returning unchanged.")
            return value
        arr = np.array(value, dtype=float) if isinstance(value, (float, int)) else value.astype(float)
        noise = self.rng.normal(0.0, self.stddev, size=arr.shape)
        return arr + noise

class ReactionWheelJitter(Fault):
    """
    Simulates friction/lag in reaction wheel torque commands using a 1st-order lag
    (low-pass) and optional small jitter noise. Each call updates internal last output.
    """
    def __init__(self, name: str, alpha: float = 0.9, jitter_std: float = 0.0, seed: Optional[int] = None, trigger_time: float = 0.0):
        """
        alpha: weighting of previous output (0..1). Higher alpha => more lag.
        jitter_std: added Gaussian noise amplitude.
        """
        super().__init__(name, trigger_time)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.jitter_std = float(jitter_std)
        self.last: Optional[np.ndarray] = None
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, float, int]):
        if not self.triggered:
            return value
        if not isinstance(value, (np.ndarray, float, int)):
            warnings.warn(f"ReactionWheelJitter: unsupported type {type(value)}; returning unchanged.")
            return value
        arr = np.array(value, dtype=float) if isinstance(value, (float, int)) else value.astype(float)
        if self.last is None:
            out = arr.copy()
        else:
            out = self.alpha * self.last + (1.0 - self.alpha) * arr
        if self.jitter_std > 0:
            out = out + self.rng.normal(0.0, self.jitter_std, size=out.shape)
        self.last = out.copy()
        return out

class EncoderDrift(Fault):
    """Introduces a slow additive drift to encoder/rate readings (per-call creep)."""
    def __init__(self, name: str, initial_offset: Union[float, np.ndarray] = 0.0,
                 drift_per_call: Union[float, np.ndarray] = 0.0, noise_std: float = 0.0, seed: Optional[int] = None, trigger_time: float = 0.0):
        super().__init__(name, trigger_time)
        self.offset = np.array(initial_offset, dtype=float)
        self.drift = np.array(drift_per_call, dtype=float)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, float, int]):
        if not self.triggered:
            return value
        if not isinstance(value, (np.ndarray, float, int)):
            warnings.warn(f"EncoderDrift: unsupported type {type(value)}; returning unchanged.")
            return value
        arr = np.array(value, dtype=float) if isinstance(value, (float, int)) else value.astype(float)
        offset_to_add = np.broadcast_to(self.offset, arr.shape)
        out = arr + offset_to_add
        noise = self.rng.normal(0.0, self.noise_std, size=np.shape(self.offset)) if self.noise_std > 0 else np.zeros_like(self.offset)
        self.offset = self.offset + self.drift + noise
        return out

class ReactionWheelAxisMisalignment(Fault):
    """
    Applies a small static rotation (misalignment) to a 3-vector torque command.
    misalignment_angles: (rx, ry, rz) in radians representing small Euler rotations.
    """
    def __init__(self, name: str, misalignment_angles: Union[float, List[float], np.ndarray] = 0.0, trigger_time: float = 0.0):
        super().__init__(name, trigger_time)
        ang = np.array(misalignment_angles, dtype=float)
        if ang.size == 1:
            ang = np.array([0.0, 0.0, ang.item()])
        if ang.size != 3:
            raise ValueError("misalignment_angles must be scalar or length-3 (rx,ry,rz)")
        rx, ry, rz = ang
        # Rotation matrices (Z * Y * X)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        self.R = Rz @ Ry @ Rx

    def action(self, value: Union[np.ndarray, float, int]):
        if not self.triggered:
            return value
        if not isinstance(value, (np.ndarray, float, int)):
            warnings.warn(f"ReactionWheelAxisMisalignment: unsupported type {type(value)}; returning unchanged.")
            return value
        arr = np.array(value, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == 3:
            return self.R.dot(arr)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return (self.R @ arr.T).T
        warnings.warn("ReactionWheelAxisMisalignment: expected shape (3,) or (N,3); returning unchanged.")
        return value

class PeriodicFault(Fault):
    """Applies an inner fault periodically using a sine function (fault active when sin > 1 after trigger_time)."""
    def __init__(self, name: str, fault: Fault, period: float = 1.0, phase: float = 0.0, trigger_time: float = 0.0):
        super().__init__(name, trigger_time)
        self.fault = fault
        self.period = float(period)
        self.phase = float(phase)
        self.time = 0.0
        self.fault.active = True  # ensure inner fault is active when this is active

    def update(self, time: float):
        super().update(time)
        self.time = time

    def action(self, value):
        if not self.triggered:
            return value
        sine_val = np.sin(2 * np.pi * (self.time / self.period) + self.phase)
        if sine_val > 1.0:
            return self.fault.action(value)
        return value

class CascadingFault(Fault):
    """
    Triggers a chain of faults. Call trigger() to start the cascade.
    Each interval (calls) advances to the next fault in the list.
    """
    def __init__(self, name: str, faults: List[Fault], interval: int = 1, trigger_time: float = 0.0):
        super().__init__(name, trigger_time)
        self.faults = list(faults)
        self.interval = max(1, int(interval))
        self.cascade_active = False
        self._step = 0
        self._idx = 0

    def update(self, time: float):
        # Call base class update first to handle triggered status
        super().update(time)
        # auto-trigger cascade when fault is triggered
        if self.triggered and not self.cascade_active:
            self.trigger_cascade()

    def trigger_cascade(self):
        """Begin the cascade from the first fault."""
        self.cascade_active = True
        self._step = 0
        self._idx = 0

    def action(self, value):
        if not self.cascade_active:
            return value
        if self._idx >= len(self.faults):
            self.cascade_active = False
            return value
        out = self.faults[self._idx].action(value)
        self._step += 1
        if self._step >= self.interval:
            self._idx += 1
            self._step = 0
        # If reached end, deactivate next call
        if self._idx >= len(self.faults):
            self.cascade_active = False
        return out

class QuaternionRotationNoise(Fault):
    """Add a small random rotation to quaternion inputs.
    
    The input is expected to be a quaternion in scalar-first order [w, x, y, z]
    (that matches the project's convention). The fault generates a small rotation
    vector from N(0, stddev) (radians) and composes it with the input quaternion.
    
    Supports 1D quaternion (4,) and stacked quaternions (N,4).
    """
    def __init__(self, name: str, stddev: float = 0.01, seed: Optional[int] = None, trigger_time: float = 0.0):
        super().__init__(name, trigger_time)
        self.stddev = float(stddev)
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, list, tuple]):
        if not self.triggered:
            return value
        if value is None:
            return value
        if not isinstance(value, (np.ndarray, list, tuple)):
            warnings.warn(f"QuaternionRotationNoise: unsupported type {type(value)}; returning unchanged.")
            return value

        arr = np.array(value, dtype=float)

        # Single quaternion
        if arr.ndim == 1 and arr.shape[0] == 4:
            rotvec = self.rng.normal(0.0, self.stddev, size=3)
            r_noise = R.from_rotvec(rotvec)
            try:
                r_in = R.from_quat(arr, scalar_first=True)
            except Exception:
                # Fallback: try non-scalar-first if user's data differs
                r_in = R.from_quat(arr)
            r_out = r_noise * r_in
            return r_out.as_quat(scalar_first=True)

        # Batched quaternions (N,4)
        if arr.ndim == 2 and arr.shape[1] == 4:
            n = arr.shape[0]
            rotvecs = self.rng.normal(0.0, self.stddev, size=(n, 3))
            r_noise = R.from_rotvec(rotvecs)
            try:
                r_in = R.from_quat(arr, scalar_first=True)
            except Exception:
                r_in = R.from_quat(arr)
            r_out = r_noise * r_in
            return r_out.as_quat(scalar_first=True)

        warnings.warn("QuaternionRotationNoise: expected quaternion shape (4,) or (N,4); returning unchanged.")
        return value

class SetUnitQuaternionFault(Fault):
    """When triggered, replace quaternion input with the unit (identity) quaternion [1,0,0,0] (scalar-first by default).

    Supports single quaternion (4,) and batched quaternions (N,4). If input is None,
    returns a single unit quaternion.

    Args:
        scalar_first (bool): If True, output [w,x,y,z]. If False, output [x,y,z,w].
    """
    def __init__(self, name: str, trigger_time: float = 0.0, scalar_first: bool = True):
        super().__init__(name, trigger_time)
        self.scalar_first = scalar_first

    def action(self, value: Union[np.ndarray, list, tuple, None]):
        if not self.triggered:
            return value
        if self.scalar_first:
            unit = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            unit = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        if value is None:
            return unit
        if not isinstance(value, (np.ndarray, list, tuple)):
            return unit
        arr = np.array(value, dtype=float)
        if arr.ndim == 1 and arr.size == 4:
            return unit
        if arr.ndim == 2 and arr.shape[1] == 4:
            return np.tile(unit, (arr.shape[0], 1))
        return unit

class SensorBiasCreepQuaternion(Fault):
    """Introduce a gradually changing rotational bias to quaternion measurements.

    Bias, drift, and noise are interpreted as small rotation vectors (rotvecs, radians)
    and composed with the input quaternion as R_bias * R_input. Supports single
    quaternion (4,) and batched quaternions (N,4). The class updates the internal
    bias by adding the drift and a small noise rotvec on each call.
    """
    def __init__(self, name: str, initial_bias: Union[float, List[float], np.ndarray] = 0.0,
                 drift_per_call: Union[float, List[float], np.ndarray] = 0.0,
                 noise_std: float = 0.0, seed: Optional[int] = None, trigger_time: float = 0.0,
                 scalar_first: bool = True):
        super().__init__(name, trigger_time)
        b = np.array(initial_bias, dtype=float)
        if b.size == 1:
            # interpret scalar as rotation about z-axis
            b = np.array([0.0, 0.0, b.item()], dtype=float)
        if b.size != 3:
            raise ValueError("initial_bias must be scalar or length-3 rotvec")
        self.bias = b

        d = np.array(drift_per_call, dtype=float)
        if d.size == 1:
            d = np.array([0.0, 0.0, d.item()], dtype=float)
        if d.size != 3:
            raise ValueError("drift_per_call must be scalar or length-3 rotvec")
        self.drift = d

        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.scalar_first = bool(scalar_first)

    def action(self, value: Union[np.ndarray, list, tuple, None]):
        if not self.triggered:
            return value
        if value is None:
            return value
        if not isinstance(value, (np.ndarray, list, tuple)):
            warnings.warn(f"SensorBiasCreepQuaternion: unsupported type {type(value)}; returning unchanged.")
            return value

        arr = np.array(value, dtype=float)

        # single quaternion
        if arr.ndim == 1 and arr.shape[0] == 4:
            try:
                r_in = R.from_quat(arr, scalar_first=self.scalar_first)
            except Exception:
                r_in = R.from_quat(arr)
            r_bias = R.from_rotvec(self.bias)
            r_out = r_bias * r_in
            # update bias (creep) for next call
            noise = self.rng.normal(0.0, self.noise_std, size=self.bias.shape) if self.noise_std > 0 else np.zeros_like(self.bias)
            self.bias = self.bias + self.drift + noise
            return r_out.as_quat(scalar_first=self.scalar_first)

        # batched quaternions
        if arr.ndim == 2 and arr.shape[1] == 4:
            n = arr.shape[0]
            try:
                r_in = R.from_quat(arr, scalar_first=self.scalar_first)
            except Exception:
                r_in = R.from_quat(arr)
            r_bias = R.from_rotvec(np.tile(self.bias, (n, 1)))
            r_out = r_bias * r_in
            noise = self.rng.normal(0.0, self.noise_std, size=self.bias.shape) if self.noise_std > 0 else np.zeros_like(self.bias)
            self.bias = self.bias + self.drift + noise
            return r_out.as_quat(scalar_first=self.scalar_first)

        warnings.warn("SensorBiasCreepQuaternion: expected quaternion shape (4,) or (N,4); returning unchanged.")
        return value