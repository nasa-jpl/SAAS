"""
Autonomous Mode Commander (AMC) Node.

Provides a simple stateful AMC implementing three modes:
- nominal: all systems healthy -> nadir pointing
- degraded: some faults but not critical -> communications attitude for a random hold
- critical: capability below threshold -> safe-mode attitude

Ports:
- Input: health (dict[str, bool]) from MONSID
- Outputs: output_w_cmd (3,), output_q_cmd (4,), output_mode (str)
"""

from typing import NamedTuple, Dict, Tuple
import math
import random
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import numpy as np
from syssim import NodeDifferential, InputPort, OutputPort



class NodeAutonomyInputs(NamedTuple):
    health: InputPort
    est_q: InputPort


class NodeAutonomyOutputs(NamedTuple):
    output_w_cmd: OutputPort
    output_q_cmd: OutputPort
    output_mode: OutputPort


class AutonomousModeCommander(NodeDifferential):
    """
    Node that selects nominal / degraded / critical modes based on a health dict.
    Integrates pointing logic for nadir and safe mode attitudes.
    Config options (all optional):
      - rw_total, enc_total, gyro_total, sru_total: nominal counts (defaults: 8,8,2,2)
      - degraded_hold_range: (min_s, max_s) for random degraded hold (default: (60,300))
      - mu, R_planet, altitude: orbital parameters for nadir pointing
    """

    def __init__(self, **kwargs):
        self._i = NodeAutonomyInputs(
            InputPort("health", self),
            InputPort("est_q", self),
        )
        self._o = NodeAutonomyOutputs(
            OutputPort("output_w_cmd", self),
            OutputPort("output_q_cmd", self),
            OutputPort("output_mode", self),
        )
        super().__init__(None, self._i, self._o, **kwargs)

        # internal state
        self.mode = "nominal"
        self._degraded_time_remaining = 0.0

    def initialize(self):
        cfg = self._config if hasattr(self, "_config") else {}
        self.rw_total = int(cfg.get("rw_total", 8))
        self.enc_total = int(cfg.get("enc_total", 8))
        self.gyro_total = int(cfg.get("gyro_total", 2))
        self.sru_total = int(cfg.get("sru_total", 2))

        self.degraded_hold_range: Tuple[float, float] = tuple(
            cfg.get("degraded_hold_range", (5.0, 10.0))
        )

        self.coms_pointing_accuracy = cfg.get("coms_pointing_accuracy", 0.1)

        # Orbital parameters for nadir pointing
        self._mu = float(cfg.get("mu", 3.986e14))  # Earth's gravitational parameter in m^3/s^2
        self._R_planet = float(cfg.get("R_planet", 6371e3))  # Earth's radius in meters
        self._altitude = float(cfg.get("altitude", 500e3))  # Orbit altitude in meters
        self._orbit_radius = self._R_planet + self._altitude

        self._orbit_velocity = np.sqrt(self._mu / self._orbit_radius)

        # Calculate an orbital angular rate for a circular orbit
        self._nadir_angular_rate = self._orbit_velocity / self._orbit_radius
        self._w_nadir = np.array([0.0, self._nadir_angular_rate, 0.0])

        # Safe mode is assumed to be non-rotating in inertial frame
        self._w_safe = np.array([0.0, 0.0, 0.0])
        self._q_safe = R.from_euler("xyz", [10.0, 15.0, 20.0], degrees=True).as_quat(canonical=True, scalar_first=True)

        # seed randomness if provided
        if "random_seed" in cfg:
            random.seed(cfg["random_seed"])

    def _calculate_nadir_pointing(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate nadir pointing quaternion and angular rate at time t."""
        orbit_angle = self._nadir_angular_rate * t

        xy_position_eci = self._orbit_radius * np.array([np.cos(orbit_angle), np.sin(orbit_angle), 0.0])

        z_body_eci = -xy_position_eci / np.linalg.norm(xy_position_eci)

        nadir_rotation = R.align_vectors([[0, 0, 1], [1, 0, 0]], [z_body_eci, [0, 0, 1]])[0].as_quat(canonical=True, scalar_first=True)
        
        return nadir_rotation, self._w_nadir

    # --- helpers ---
    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        if n <= 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n

    @staticmethod
    def _count_prefixed_keys(health: Dict[str, str], prefix: str) -> int:
        if health is None:
            return 0
        c = 0
        for k, v in health.items():
            if not isinstance(k, str):
                continue
            if k.lower().startswith(prefix.lower()) and v == "Healthy":
                c += 1
        return c

    def _evaluate_capability(self, health: Dict[str, bool]) -> Dict[str, int]:
        return {
            "rw": self._count_prefixed_keys(health, "rwa_"),
            "enc": self._count_prefixed_keys(health, "enc_"),
            "gyro": self._count_prefixed_keys(health, "imu_"),
            "sru": self._count_prefixed_keys(health, "sru_"),
        }

    def _is_nominal(self, counts: Dict[str, int]) -> bool:
        return (
            counts["rw"] >= self.rw_total
            and counts["enc"] >= self.enc_total
            and counts["gyro"] >= self.gyro_total
            and counts["sru"] >= self.sru_total
        )

    def _is_critical(self, counts: Dict[str, int]) -> bool:
        # critical: <=3 reaction wheels OR <=1 SRU (sensor)
        return (counts["rw"] <= 3) or (counts["sru"] <= 1)

    # --- Node API ---
    def update(self, sim_time: float):
        # dt available via self.period if set; fall back to 0.0
        dt = getattr(self, "period", 0.0) or 0.0

        # Read inputs
        est_q = self._i.est_q.read()
        health = self._i.health.read()

        # Calculate pointing commands
        nadir_q, nadir_w = self._calculate_nadir_pointing(sim_time)
        safe_q, safe_w = self._q_safe, self._w_safe

        # If we have no health info
        if health is None:
            self._o.output_w_cmd.shift_out(safe_w)
            self._o.output_q_cmd.shift_out(safe_q)
            self._o.output_mode.shift_out("unknown")
            return

        counts = self._evaluate_capability(health)

        # decide target mode
        if self._is_critical(counts):
            target_mode = "critical"
        elif self._is_nominal(counts):
            target_mode = "nominal"
        else:
            target_mode = "degraded"

        # handle transitions
        if target_mode != self.mode:
            # Print mode transition using tqdm write
            tqdm.write(f"Mode transition: {self.mode} -> {target_mode} at t={sim_time:.4f}s")
            if target_mode == "degraded":
                # Enter degraded: reset degraded timer to random value in range
                self._degraded_time_remaining = random.uniform(*self.degraded_hold_range)

            self.mode = target_mode

        # produce commands based on current mode and degraded timer
        if self.mode == "nominal":
            w_cmd = nadir_w
            q_cmd = nadir_q
        elif self.mode == "degraded":
            if self._degraded_time_remaining <= 0.0:
                # timer expired -> resume nadir pointing
                w_cmd = nadir_w
                q_cmd = nadir_q
            else:
                # hold safe attitude and decrement timer
                r_cmd = R.from_quat(safe_q)
                r_est = R.from_quat(est_q)
                r_err = r_cmd * r_est.inv()
                if r_err.magnitude() < self.coms_pointing_accuracy:
                    self._degraded_time_remaining = max(
                        0.0, self._degraded_time_remaining - dt
                    )
                w_cmd = safe_w
                q_cmd = safe_q
        else:  # critical
            w_cmd = safe_w
            q_cmd = safe_q

        # publish outputs
        self._o.output_w_cmd.shift_out(w_cmd)
        self._o.output_q_cmd.shift_out(q_cmd)
        self._o.output_mode.shift_out(self.mode)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o
        return self._o
