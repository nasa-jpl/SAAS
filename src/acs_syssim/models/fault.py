from syssim.core.fault import Fault
import numpy as np
import warnings
from typing import Optional, List, Union

class DiagonalInertiaPerturbFault(Fault):
    """Fault that perturbs each axis of a 3x3 diagonal inertia matrix by a random amount."""
    def __init__(self, name: str, stddev: float = 0.1, seed: int = None):
        """
        Args:
            name (str): Name of the fault.
            stddev (float): Standard deviation of the random perturbation (fractional, e.g., 0.01 for 1%).
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(name)
        self.stddev = stddev
        self.rng = np.random.default_rng(seed)
        # Generate random perturbation factors for each axis
        self._perturbation = 1.0 + self.rng.normal(0, self.stddev, size=3)

    def action(self, value: np.ndarray):
        """Perturb the diagonal elements of a 3x3 diagonal inertia matrix by a random amount."""
        assert isinstance(value, np.ndarray), "Value must be a numpy ndarray."
        assert value.shape == (3, 3), "Inertia matrix must be 3x3."
        assert np.allclose(value, np.diag(np.diagonal(value))), "Matrix must be diagonal."
        perturbed_diag = np.diagonal(value) * self._perturbation
        return np.diag(perturbed_diag)

class RandomizeFault(Fault):
    """Fault that randomizes the input value within a specified range."""
    def __init__(self, name: str, low: float, high: float, seed: int = None):
        """
        Args:
            name (str): Name of the fault.
            low (float): Lower bound of the randomization range.
            high (float): Upper bound of the randomization range.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(name)
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def action(self, value: np.ndarray):
        """Randomize the input value within the specified range."""
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
                 noise_std: float = 0.0, seed: Optional[int] = None):
        super().__init__(name)
        self.bias = np.array(initial_bias, dtype=float)
        self.drift = np.array(drift_per_call, dtype=float)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, float, int]):
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
    def __init__(self, name: str, stddev: float, seed: Optional[int] = None):
        super().__init__(name)
        self.stddev = float(stddev)
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, float, int]):
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
    def __init__(self, name: str, alpha: float = 0.9, jitter_std: float = 0.0, seed: Optional[int] = None):
        """
        alpha: weighting of previous output (0..1). Higher alpha => more lag.
        jitter_std: added Gaussian noise amplitude.
        """
        super().__init__(name)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.jitter_std = float(jitter_std)
        self.last: Optional[np.ndarray] = None
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, float, int]):
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
                 drift_per_call: Union[float, np.ndarray] = 0.0, noise_std: float = 0.0, seed: Optional[int] = None):
        super().__init__(name)
        self.offset = np.array(initial_offset, dtype=float)
        self.drift = np.array(drift_per_call, dtype=float)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def action(self, value: Union[np.ndarray, float, int]):
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
    def __init__(self, name: str, misalignment_angles: Union[float, List[float], np.ndarray] = 0.0):
        super().__init__(name)
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
    """Applies an inner fault periodically (every `period` calls)."""
    def __init__(self, name: str, fault: Fault, period: int = 1, offset: int = 0):
        super().__init__(name)
        self.fault = fault
        self.period = max(1, int(period))
        self.offset = int(offset)
        self.step = 0

    def action(self, value):
        self.step += 1
        if ((self.step - self.offset) % self.period) == 0:
            return self.fault.action(value)
        return value

class CascadingFault(Fault):
    """
    Triggers a chain of faults. Call trigger() to start the cascade.
    Each interval (calls) advances to the next fault in the list.
    """
    def __init__(self, name: str, faults: List[Fault], interval: int = 1):
        super().__init__(name)
        self.faults = list(faults)
        self.interval = max(1, int(interval))
        self.active = False
        self._step = 0
        self._idx = 0

    def trigger(self):
        """Begin the cascade from the first fault."""
        self.active = True
        self._step = 0
        self._idx = 0

    def action(self, value):
        if not self.active:
            return value
        if self._idx >= len(self.faults):
            self.active = False
            return value
        out = self.faults[self._idx].action(value)
        self._step += 1
        if self._step >= self.interval:
            self._idx += 1
            self._step = 0
        # If reached end, deactivate next call
        if self._idx >= len(self.faults):
            self.active = False
        return out