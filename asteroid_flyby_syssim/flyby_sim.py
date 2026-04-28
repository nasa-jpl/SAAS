"""Asteroid flyby simulation with syssim nodes.

This module builds a syssim graph for a spacecraft performing a hyperbolic
asteroid flyby while keeping a camera boresight pointed at asteroid center
using reaction-wheel attitude control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import imageio.v2 as imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from syssim.core import InputPort, Node, NodeDifferential, NodeSystem, OutputPort
from syssim.nodes.io import ExternalInputNode, ExternalOutputNode

from .asteroid_camera import NodeAsteroidCamera
from .asteroid_gravity import NodeAsteroidGravity
from .gyroscope import NodeGyroscope


# Tetrahedral reaction wheel configuration.
# Four wheels arranged at vertices of a tetrahedron, normalized.
TETRAHEDRAL_WHEEL_AXES = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
], dtype=float)
for i in range(4):
    TETRAHEDRAL_WHEEL_AXES[i] /= np.linalg.norm(TETRAHEDRAL_WHEEL_AXES[i])

# Torque allocation matrix: inverse of the axes matrix for command allocation.
# Each column is a wheel axis; rows map body-frame tau_cmd to wheel tau_cmd.
TORQUE_ALLOCATION_MATRIX = np.linalg.pinv(TETRAHEDRAL_WHEEL_AXES.T)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=float)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=float)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _rotation_from_wxyz(q_wxyz: np.ndarray) -> Rotation:
    q_wxyz = _normalize(np.asarray(q_wxyz, dtype=float))
    return Rotation.from_quat(_quat_wxyz_to_xyzw(q_wxyz))


def _wxyz_from_rotation(rot: Rotation) -> np.ndarray:
    return _quat_xyzw_to_wxyz(rot.as_quat())


def _quat_conjugate_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=float)
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]], dtype=float)


def _quat_multiply_wxyz(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a_wxyz, dtype=float)
    bw, bx, by, bz = np.asarray(b_wxyz, dtype=float)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _error_quaternion_wxyz(q_bi_wxyz: np.ndarray, q_cmd_bi_wxyz: np.ndarray) -> np.ndarray:
    """Compute q_e = q_c^{-1} ⊗ q in scalar-first convention (w, x, y, z)."""
    q_bi_wxyz = _normalize(np.asarray(q_bi_wxyz, dtype=float))
    q_cmd_bi_wxyz = _normalize(np.asarray(q_cmd_bi_wxyz, dtype=float))
    q_cmd_inv = _quat_conjugate_wxyz(q_cmd_bi_wxyz)
    q_err = _quat_multiply_wxyz(q_cmd_inv, q_bi_wxyz)
    return _normalize(q_err)


def _attitude_error_vector_body(q_bi_wxyz: np.ndarray, q_cmd_bi_wxyz: np.ndarray) -> np.ndarray:
    """Return vector part of q_e = q_c^{-1} ⊗ q for Wie-style feedback."""
    q_err = _error_quaternion_wxyz(q_bi_wxyz=q_bi_wxyz, q_cmd_bi_wxyz=q_cmd_bi_wxyz)
    return q_err[1:4]


def _compute_center_pointing_quaternion_wxyz(
    position_sc_i_m: np.ndarray,
    velocity_sc_i_mps: np.ndarray | None = None,
    prev_q_cmd_bi_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Compute body->inertial command quaternion for center-pointing with roll continuity."""
    x_b_i = _normalize(-position_sc_i_m)
    if np.linalg.norm(x_b_i) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    y_ref_i = None
    if prev_q_cmd_bi_wxyz is not None:
        prev_r_bi = _rotation_from_wxyz(prev_q_cmd_bi_wxyz)
        prev_y_i = prev_r_bi.apply(np.array([0.0, 1.0, 0.0], dtype=float))
        y_ref_i = prev_y_i - np.dot(prev_y_i, x_b_i) * x_b_i

    if (y_ref_i is None or np.linalg.norm(y_ref_i) < 1e-9) and velocity_sc_i_mps is not None:
        h_orbit_i = np.cross(position_sc_i_m, velocity_sc_i_mps)
        if np.linalg.norm(h_orbit_i) > 1e-9:
            y_ref_i = np.cross(h_orbit_i, x_b_i)

    if y_ref_i is None or np.linalg.norm(y_ref_i) < 1e-9:
        z_ref_i = np.array([0.0, 0.0, 1.0], dtype=float)
        y_ref_i = np.cross(z_ref_i, x_b_i)
        if np.linalg.norm(y_ref_i) < 1e-9:
            y_ref_i = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), x_b_i)

    y_b_i = _normalize(y_ref_i)
    z_b_i = _normalize(np.cross(x_b_i, y_b_i))
    y_b_i = _normalize(np.cross(z_b_i, x_b_i))

    r_bi = Rotation.from_matrix(np.column_stack([x_b_i, y_b_i, z_b_i]))
    q_cmd = _wxyz_from_rotation(r_bi)

    if prev_q_cmd_bi_wxyz is not None and np.dot(q_cmd, prev_q_cmd_bi_wxyz) < 0.0:
        q_cmd = -q_cmd
    return q_cmd


def _quat_look_rotation(forward_i: np.ndarray, up_hint_i: np.ndarray | None = None) -> np.ndarray:
    x_axis = _normalize(forward_i)
    if np.linalg.norm(x_axis) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    if up_hint_i is None:
        up_hint_i = np.array([0.0, 0.0, 1.0], dtype=float)
    up_hint_i = _normalize(up_hint_i)

    y_axis = np.cross(up_hint_i, x_axis)
    if np.linalg.norm(y_axis) < 1e-9:
        up_hint_i = np.array([0.0, 1.0, 0.0], dtype=float)
        y_axis = np.cross(up_hint_i, x_axis)
    y_axis = _normalize(y_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))

    # Rotation matrix mapping body axes to inertial axes.
    r_bi = np.column_stack([x_axis, y_axis, z_axis])
    return _wxyz_from_rotation(Rotation.from_matrix(r_bi))


def _hyperbolic_state_from_params(
    mu: float,
    periapsis_radius_m: float,
    external_angle: float,
    true_anomaly_deg: float,
    inbound_ra_deg: float,
    inbound_dec_deg: float,
    bplane_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an approximate hyperbolic state in asteroid-centered inertial frame.

    The shape (e, p) follows standard two-body hyperbola definitions:
    - a = -mu / v_inf^2
    - e = 1 + rp * v_inf^2 / mu
    - p = rp * (1 + e)

    Orientation uses an inbound asymptote direction and B-plane angle to define a
    flyby plane basis. This yields a practical parameterized flyby initialization.
    """
    rp = periapsis_radius_m
    nu = np.deg2rad(true_anomaly_deg)
    external_angle = np.deg2rad(external_angle)

    e = -1 / np.cos(external_angle)
    p = rp * (1.0 + e)

    ra = np.deg2rad(inbound_ra_deg)
    dec = np.deg2rad(inbound_dec_deg)
    s_hat = _normalize(np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]))

    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(ref, s_hat)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)

    b1 = _normalize(np.cross(s_hat, ref))
    b2 = _normalize(np.cross(s_hat, b1))
    theta = np.deg2rad(bplane_angle_deg)
    b_hat = _normalize(np.cos(theta) * b1 + np.sin(theta) * b2)

    h_hat = _normalize(np.cross(s_hat, b_hat))
    e_hat = _normalize(np.cross(h_hat, s_hat))
    p_hat = _normalize(np.cross(h_hat, e_hat))

    r_mag = p / (1.0 + e * np.cos(nu))
    r = r_mag * (np.cos(nu) * e_hat + np.sin(nu) * p_hat)

    v_scale = np.sqrt(mu / p)
    v = v_scale * (-np.sin(nu) * e_hat + (e + np.cos(nu)) * p_hat)
    return r, v


@dataclass
class SimulationConfig:
    duration_s: float = 36000.0
    dt_s: float = 0.5
    start_date_utc: str = "2028-01-01T00:00:00+00:00"


@dataclass
class AsteroidConfig:
    asteroid: str = "Ceres"
    gravity_lmax: int | None = 12


@dataclass
class FlybyConfig:
    periapsis_radius_m: float = 8.0e5
    external_angle_deg: float = 140.0
    true_anomaly0_deg: float = -90.0
    inbound_ra_deg: float = 10.0
    inbound_dec_deg: float = 10.0
    bplane_angle_deg: float = 1.0


@dataclass
class SpacecraftConfig:
    inertia_kgm2: tuple[float, float, float] = (70.0, 60.0, 45.0)
    boresight_body: tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass
class ControllerConfig:
    kp: float = 0.037
    kd: float = 1.2
    ki: float = 0.0
    integral_limit: float = 0.2


@dataclass
class ReactionWheelConfig:
    """Three-axis wheel model with realistic constraints.

    Model sources:
    - Markley et al. (NASA NTRS 20110015369) for wheel-array torque/momentum
      envelope interpretation and saturation rationale.
    - Basilisk reaction wheel state effector docs for practical wheel state
      integration and command/omega interface conventions.

    Simulated effects:
    - Motor torque saturation
    - Wheel-speed and momentum saturation
    - First-order motor command lag
    - Viscous + Coulomb bearing friction
    - Optional imbalance/jitter body-torque disturbance
    """

    # Default hardware target: Blue Canyon Technologies RW2
    # (max momentum 2.0 Nms, max torque 0.12 Nm, dynamic unbalance < 250 g-mm^2).
    wheel_inertia_kgm2: tuple[float, float, float] = (
        2.0 / (6000.0 * 2.0 * np.pi / 60.0),
        2.0 / (6000.0 * 2.0 * np.pi / 60.0),
        2.0 / (6000.0 * 2.0 * np.pi / 60.0),
    )
    torque_max_nm: tuple[float, float, float] = (0.12, 0.12, 0.12)
    wheel_speed_max_rads: tuple[float, float, float] = (6000.0 * 2.0 * np.pi / 60.0,) * 3
    momentum_max_nms: tuple[float, float, float] = (2.0, 2.0, 2.0)
    command_lag_tau_s: float = 0.05
    viscous_friction_nms: float = 1.0e-5
    coulomb_friction_nm: float = 2.0e-5
    jitter_std_nm: float = 1.0e-5
    # 250 g-mm^2 = 2.5e-7 kg*m^2, used as a conservative imbalance torque coefficient upper bound.
    imbalance_coeff_nm_per_rads2: float = 2.5e-7
    imbalance_freq_hz: float = 37.0


@dataclass
class OutputConfig:
    output_dir: str = "./outputs"
    run_name: str = "asteroid_flyby"
    render_video: bool = True
    render_fps: float = 1/500
    trajectory_fps: float = 20.0
    camera_width: int = 512
    camera_height: int = 512
    camera_fov_deg: float = 42.0
    spp: int = 16
    use_integrator_mask: bool = False


@dataclass
class FlybyRunConfig:
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    asteroid: AsteroidConfig = field(default_factory=AsteroidConfig)
    flyby: FlybyConfig = field(default_factory=FlybyConfig)
    spacecraft: SpacecraftConfig = field(default_factory=SpacecraftConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    rwa: ReactionWheelConfig = field(default_factory=ReactionWheelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


class NodeHyperbolicDynamicsInputs(NamedTuple):
    gravity_accel: InputPort


class NodeHyperbolicDynamicsOutputs(NamedTuple):
    position: OutputPort
    velocity: OutputPort
    gravity_error: OutputPort  # Difference between gravity model and Keplerian


class NodeHyperbolicDynamics(NodeDifferential):
    """Translational dynamics with scipy RK45 integrator.
    
    Integrates position and velocity using gravity acceleration from the
    gravity model node. Also computes and outputs the difference between
    the gravity model acceleration and point-mass Keplerian gravity.
    """

    def __init__(self, x0: np.ndarray, mu: float, **kwargs):
        self._mu = mu
        self._i = NodeHyperbolicDynamicsInputs(InputPort("gravity_accel", self))
        self._o = NodeHyperbolicDynamicsOutputs(
            OutputPort("position", self),
            OutputPort("velocity", self),
            OutputPort("gravity_error", self),
        )
        super().__init__(x0, self._i, self._o, **kwargs)

    def initialize(self):
        self._t = 0.0
        self._last_gravity_accel = np.zeros(3, dtype=float)
        self._last_gravity_error = np.zeros(3, dtype=float)

    def update(self, sim_time: float):
        dt = sim_time - self._t
        if dt <= 0.0:
            return

        # Read gravity model acceleration
        acc_grav = self._i.gravity_accel.read()
        if acc_grav is None or np.any(np.isnan(acc_grav)):
            acc_grav = np.zeros(3, dtype=float)
        
        self._last_gravity_accel = acc_grav.copy()

        # Compute Keplerian (point-mass) gravity
        r = self._x[0:3]
        r_norm = np.linalg.norm(r)
        if r_norm > 1e-6:
            acc_kepler = -self._mu / r_norm**3 * r
        else:
            acc_kepler = np.zeros(3, dtype=float)

        # Gravity error: difference from Keplerian
        gravity_error = acc_grav - acc_kepler
        self._last_gravity_error = gravity_error.copy()

        # Use scipy RK45 integrator for better accuracy
        def rhs(t, state):
            # Use full gravity model acceleration (not Keplerian)
            v_local = state[3:6]
            return np.concatenate([v_local, acc_grav])

        result = solve_ivp(
            rhs,
            [self._t, sim_time],
            self._x,
            method='RK45',
            dense_output=False,
            max_step=dt,  # Limit step size
        )

        if result.status == 0:  # Successful integration
            self._x = result.y[:, -1]  # Take final state
        else:
            # Fallback to simple Euler if integration fails
            v = self._x[3:6]
            v = v + acc_grav * dt
            r = self._x[0:3] + v * dt
            self._x = np.concatenate([r, v])

        self._t = sim_time

        # Output position, velocity, and gravity error
        pos = self._x[0:3]
        vel = self._x[3:6]

        self._o.position.shift_out(pos, sim_time)
        self._o.velocity.shift_out(vel, sim_time)
        self._o.gravity_error.shift_out(gravity_error, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class NodeCenterPointingGuidanceInputs(NamedTuple):
    position: InputPort
    velocity: InputPort


class NodeCenterPointingGuidanceOutputs(NamedTuple):
    q_cmd: OutputPort
    w_cmd: OutputPort
    look_dir_cmd: OutputPort


class NodeCenterPointingGuidance(Node):
    def __init__(self, **kwargs):
        self._i = NodeCenterPointingGuidanceInputs(
            InputPort("position", self),
            InputPort("velocity", self),
        )
        self._o = NodeCenterPointingGuidanceOutputs(
            OutputPort("q_cmd", self),
            OutputPort("w_cmd", self),
            OutputPort("look_dir_cmd", self),
        )
        super().__init__(self._i, self._o, **kwargs)

    def initialize(self):
        self._q_cmd_prev_bi = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def update(self, sim_time: float):
        position_sc_i_m = self._i.position.read()
        velocity_sc_i_mps = self._i.velocity.read()

        if position_sc_i_m is None or np.linalg.norm(position_sc_i_m) < 1e-8:
            look_dir_i = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            look_dir_i = _normalize(-position_sc_i_m)

        if position_sc_i_m is None:
            q_cmd_bi = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            q_cmd_bi = _compute_center_pointing_quaternion_wxyz(
                position_sc_i_m=position_sc_i_m,
                velocity_sc_i_mps=velocity_sc_i_mps,
                prev_q_cmd_bi_wxyz=self._q_cmd_prev_bi,
            )
        self._q_cmd_prev_bi = q_cmd_bi

        w_cmd_b_rps = np.zeros(3, dtype=float)

        self._o.q_cmd.shift_out(q_cmd_bi, sim_time)
        self._o.w_cmd.shift_out(w_cmd_b_rps, sim_time)
        self._o.look_dir_cmd.shift_out(look_dir_i, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class NodeAttitudeControllerInputs(NamedTuple):
    q_cmd: InputPort
    q: InputPort
    w_cmd: InputPort
    w: InputPort
    h_rw: InputPort


class NodeAttitudeControllerOutputs(NamedTuple):
    tau_cmd_body: OutputPort
    q_err: OutputPort
    w_err: OutputPort


class NodeAttitudeController(NodeDifferential):
    """Quaternion feedback controller following Wie et al. (1989) Eq. (9).

    Uses q_e = q_c^{-1} ⊗ q, shortest-path sign(q_e_scalar), and D=dJ, K=kJ.
    """

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
        self._i = NodeAttitudeControllerInputs(
            InputPort("q_cmd", self),
            InputPort("q", self),
            InputPort("w_cmd", self),
            InputPort("w", self),
            InputPort("h_rw", self),
        )
        self._o = NodeAttitudeControllerOutputs(
            OutputPort("tau_cmd_body", self),
            OutputPort("q_err", self),
            OutputPort("w_err", self),
        )
        super().__init__(np.zeros(3, dtype=float), self._i, self._o, **kwargs)

    def initialize(self):
        self._t = 0.0

    def update(self, sim_time: float):
        q_cmd_bi = self._i.q_cmd.read()
        q_bi = self._i.q.read()
        w_cmd_b_rps = self._i.w_cmd.read()
        w_b_rps = self._i.w.read()
        h_rw_b_nms = self._i.h_rw.read()

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

        q_err = _error_quaternion_wxyz(q_bi_wxyz=q_bi, q_cmd_bi_wxyz=q_cmd_bi)
        q_err_vec = q_err[1:4]
        q_err_scalar = q_err[0]
        w_err_b_rps = w_b_rps - w_cmd_b_rps

        dt = sim_time - self._t
        if dt > 0.0:
            self._x = np.clip(self._x + q_err_vec * dt, -self._integral_limit, self._integral_limit)
        self._t = sim_time

        sign_shortest = 1.0 if q_err_scalar >= 0.0 else -1.0

        # Eq. (9): u = -w×(Jw) - Dw - sign(qe_scalar)Kqe_vec (+ I-term extension).
        gyro_term_b_nm = np.cross(w_b_rps, self._inertia_mat @ w_b_rps + h_rw_b_nms)
        tau_cmd_b_nm = (
            -gyro_term_b_nm
            - (self._d_mat @ w_err_b_rps)
            - sign_shortest * (self._k_mat @ q_err_vec)
            - self._ki * self._x
        )

        self._o.tau_cmd_body.shift_out(tau_cmd_b_nm, sim_time)
        self._o.q_err.shift_out(q_err_vec, sim_time)
        self._o.w_err.shift_out(w_err_b_rps, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class NodeTorqueAllocatorInputs(NamedTuple):
    tau_cmd_body: InputPort


class NodeTorqueAllocatorOutputs(NamedTuple):
    tau_cmd_wheel: OutputPort  # Shape (4,)


class NodeTorqueAllocator(Node):
    """Allocates body-frame torque command to four tetrahedral wheels."""

    def __init__(self, **kwargs):
        self._i = NodeTorqueAllocatorInputs(InputPort("tau_cmd_body", self))
        self._o = NodeTorqueAllocatorOutputs(OutputPort("tau_cmd_wheel", self))
        super().__init__(self._i, self._o, **kwargs)

    def update(self, sim_time: float):
        tau_cmd_body = self._i.tau_cmd_body.read()
        if tau_cmd_body is None:
            tau_cmd_body = np.zeros(3, dtype=float)

        # Allocate body torque to each wheel command: tau_cmd_i = (allocation_matrix @ tau_cmd_body)[i]
        # Negative because wheel torque opposes body torque.
        tau_cmd_wheel = -TORQUE_ALLOCATION_MATRIX @ tau_cmd_body

        self._o.tau_cmd_wheel.shift_out(tau_cmd_wheel, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class NodeReactionWheelInputs(NamedTuple):
    tau_cmd: InputPort


class NodeReactionWheelOutputs(NamedTuple):
    h_rw: OutputPort
    tau_rw: OutputPort
    omega: OutputPort
    wheel_torque: OutputPort


class NodeReactionWheel(NodeDifferential):
    """Single reaction wheel with saturation, lag, friction, and disturbances.
    
    Each wheel spins about its own axis direction (defined in body frame).
    Input tau_cmd is the scalar torque command for this wheel.
    Outputs are momentum (body frame), reaction torque (body frame), and angular speed.
    """

    def __init__(self, wheel_idx: int, cfg: ReactionWheelConfig, wheel_axis: np.ndarray, **kwargs):
        self._wheel_idx = wheel_idx
        self._cfg = cfg
        self._wheel_axis = _normalize(wheel_axis)  # Direction of spin in body frame
        self._j = cfg.wheel_inertia_kgm2[0]  # Use first component (all the same)
        self._tau_max = cfg.torque_max_nm[0]
        self._omega_max = cfg.wheel_speed_max_rads[0]
        self._h_max = cfg.momentum_max_nms[0]
        
        self._i = NodeReactionWheelInputs(InputPort("tau_cmd", self))
        self._o = NodeReactionWheelOutputs(
            OutputPort("h_rw", self),
            OutputPort("tau_rw", self),
            OutputPort("omega", self),
            OutputPort("wheel_torque", self),
        )
        # State is wheel angular speed (rad/s), scalar
        super().__init__(np.array([0.0], dtype=float), self._i, self._o, **kwargs)

    def initialize(self):
        self._t = 0.0
        self._tau_cmd_lagged = 0.0
        self._rng = np.random.default_rng(7 + self._wheel_idx)

    def update(self, sim_time: float):
        dt = sim_time - self._t
        if dt <= 0.0:
            return

        tau_cmd = self._i.tau_cmd.read()
        if tau_cmd is None:
            tau_cmd = np.array([0.0])
        tau_cmd_scalar = float(tau_cmd[self._wheel_idx]) if isinstance(tau_cmd, np.ndarray) else float(tau_cmd)

        # Clamp motor command
        tau_cmd_scalar = np.clip(tau_cmd_scalar, -self._tau_max, self._tau_max)

        # First-order lag on motor command
        alpha = np.exp(-dt / max(self._cfg.command_lag_tau_s, 1e-6))
        self._tau_cmd_lagged = alpha * self._tau_cmd_lagged + (1.0 - alpha) * tau_cmd_scalar

        omega = self._x[0]

        # Friction: viscous + Coulomb
        tau_visc = self._cfg.viscous_friction_nms * omega
        tau_coul = self._cfg.coulomb_friction_nm * np.tanh(omega / 0.01)
        tau_fric = tau_visc + tau_coul

        # Disturbances: imbalance + jitter
        imbalance = self._cfg.imbalance_coeff_nm_per_rads2 * (omega * omega) * np.sin(
            2.0 * np.pi * self._cfg.imbalance_freq_hz * sim_time
        )
        jitter = self._rng.normal(0.0, self._cfg.jitter_std_nm)

        # Angular acceleration: (tau_cmd - friction) / J
        domega = (self._tau_cmd_lagged - tau_fric) / max(self._j, 1e-9)
        omega_new = omega + domega * dt

        # Speed and momentum saturation
        omega_limit_from_h = self._h_max / max(self._j, 1e-9)
        omega_lim = min(self._omega_max, omega_limit_from_h)
        omega_new = np.clip(omega_new, -omega_lim, omega_lim)

        # Effective torque after saturation
        effective_domega = (omega_new - omega) / max(dt, 1e-9)
        tau_rw_wheel = self._j * effective_domega

        # Momentum and body torque (along wheel axis)
        h = self._j * omega_new
        h_rw_body = h * self._wheel_axis
        tau_rw_body = -tau_rw_wheel * self._wheel_axis - (imbalance + jitter) * self._wheel_axis

        self._x = np.array([omega_new], dtype=float)
        self._t = sim_time

        self._o.h_rw.shift_out(h_rw_body, sim_time)
        self._o.tau_rw.shift_out(tau_rw_body, sim_time)
        self._o.omega.shift_out(np.array([omega_new], dtype=float), sim_time)
        self._o.wheel_torque.shift_out(tau_rw_wheel, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class NodeWheelAggregatorInputs(NamedTuple):
    h_rw_0: InputPort
    h_rw_1: InputPort
    h_rw_2: InputPort
    h_rw_3: InputPort
    tau_rw_0: InputPort
    tau_rw_1: InputPort
    tau_rw_2: InputPort
    tau_rw_3: InputPort


class NodeWheelAggregatorOutputs(NamedTuple):
    h_rw_total: OutputPort
    tau_rw_total: OutputPort


class NodeWheelAggregator(Node):
    """Sums momentum and torque from all four wheels."""

    def __init__(self, **kwargs):
        self._i = NodeWheelAggregatorInputs(
            InputPort("h_rw_0", self),
            InputPort("h_rw_1", self),
            InputPort("h_rw_2", self),
            InputPort("h_rw_3", self),
            InputPort("tau_rw_0", self),
            InputPort("tau_rw_1", self),
            InputPort("tau_rw_2", self),
            InputPort("tau_rw_3", self),
        )
        self._o = NodeWheelAggregatorOutputs(
            OutputPort("h_rw_total", self),
            OutputPort("tau_rw_total", self),
        )
        super().__init__(self._i, self._o, **kwargs)

    def update(self, sim_time: float):
        h_rw = np.zeros(3, dtype=float)
        tau_rw = np.zeros(3, dtype=float)

        for i in range(4):
            h = getattr(self._i, f"h_rw_{i}").read()
            tau = getattr(self._i, f"tau_rw_{i}").read()
            if h is not None:
                h_rw += h
            if tau is not None:
                tau_rw += tau

        self._o.h_rw_total.shift_out(h_rw, sim_time)
        self._o.tau_rw_total.shift_out(tau_rw, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class NodeAttitudeDynamicsInputs(NamedTuple):
    tau_body: InputPort
    h_rw_body: InputPort


class NodeAttitudeDynamicsOutputs(NamedTuple):
    q: OutputPort
    w: OutputPort


class NodeAttitudeDynamics(NodeDifferential):
    def __init__(self, inertia_kgm2: tuple[float, float, float], x0: np.ndarray, **kwargs):
        self._j = np.diag(np.array(inertia_kgm2, dtype=float))
        self._j_inv = np.linalg.inv(self._j)
        self._i = NodeAttitudeDynamicsInputs(InputPort("tau_body", self), InputPort("h_rw_body", self))
        self._o = NodeAttitudeDynamicsOutputs(OutputPort("q", self), OutputPort("w", self))
        super().__init__(x0, self._i, self._o, **kwargs)

    def initialize(self):
        self._t = 0.0

    def update(self, sim_time: float):
        dt = sim_time - self._t
        if dt <= 0.0:
            return

        tau = self._i.tau_body.read()
        h_rw = self._i.h_rw_body.read()
        if tau is None:
            tau = np.zeros(3, dtype=float)
        if h_rw is None:
            h_rw = np.zeros(3, dtype=float)

        q = _normalize(self._x[0:4])
        w = self._x[4:7]

        ang_acc = self._j_inv @ (tau - np.cross(w, self._j @ w + h_rw))

        w_new = w + ang_acc * dt

        # Propagate attitude using scipy Rotation exponential map.
        # q maps body->inertial, and body-rate increments compose on the right.
        r_bi = _rotation_from_wxyz(q)
        delta_r = Rotation.from_rotvec(w_new * dt)
        q_new = _wxyz_from_rotation(r_bi * delta_r)
        q_new = _normalize(q_new)

        self._x = np.concatenate([q_new, w_new])
        self._t = sim_time

        self._o.q.shift_out(q_new, sim_time)
        self._o.w.shift_out(w_new, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class NodeLookVectorInputs(NamedTuple):
    q: InputPort
    position: InputPort


class NodeLookVectorOutputs(NamedTuple):
    look_vec: OutputPort
    look_target: OutputPort


class NodeLookVector(Node):
    def __init__(self, boresight_body: tuple[float, float, float], **kwargs):
        self._boresight_body = _normalize(np.array(boresight_body, dtype=float))
        self._i = NodeLookVectorInputs(InputPort("q", self), InputPort("position", self))
        self._o = NodeLookVectorOutputs(OutputPort("look_vec", self), OutputPort("look_target", self))
        super().__init__(self._i, self._o, **kwargs)

    def update(self, sim_time: float):
        q = self._i.q.read()
        pos = self._i.position.read()
        if q is None:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if pos is None:
            pos = np.zeros(3, dtype=float)

        r_bi = _rotation_from_wxyz(q)
        look_vec = _normalize(r_bi.apply(self._boresight_body))
        look_target = pos + max(np.linalg.norm(pos), 1.0) * look_vec

        self._o.look_vec.shift_out(look_vec, sim_time)
        self._o.look_target.shift_out(look_target, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class NodeTrajectoryAnimatorInputs(NamedTuple):
    position: InputPort
    look_vec: InputPort


class NodeTrajectoryAnimator(Node):
    def __init__(self, output_path: Path, fps: float, **kwargs):
        self._i = NodeTrajectoryAnimatorInputs(InputPort("position", self), InputPort("look_vec", self))
        self._output_path = Path(output_path)
        self._fps = fps
        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        self._t: list[float] = []
        self._r: list[np.ndarray] = []
        self._look: list[np.ndarray] = []

    def update(self, sim_time: float):
        r = self._i.position.read()
        look = self._i.look_vec.read()
        if r is None or look is None:
            return
        self._t.append(sim_time)
        self._r.append(np.array(r, dtype=float))
        self._look.append(np.array(look, dtype=float))

    def finalize(self, fault_history=None):
        if not self._r:
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        r = np.array(self._r)
        look = np.array(self._look)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        lim = np.max(np.linalg.norm(r, axis=1)) * 1.1
        lim = max(lim, 1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Flyby Trajectory and Camera Look Vector")

        ax.scatter([0.0], [0.0], [0.0], s=180, c="tab:brown", label="Asteroid Center")

        (traj_line,) = ax.plot([], [], [], c="tab:blue", lw=1.5, label="Spacecraft Trajectory")
        point = ax.scatter([], [], [], c="tab:red", s=20)
        quiv = None

        def _update(frame_idx: int):
            nonlocal quiv
            rr = r[: frame_idx + 1]
            traj_line.set_data(rr[:, 0], rr[:, 1])
            traj_line.set_3d_properties(rr[:, 2])

            cur = r[frame_idx]
            point._offsets3d = ([cur[0]], [cur[1]], [cur[2]])

            if quiv is not None:
                quiv.remove()
            arrow_scale = max(np.linalg.norm(cur) * 0.2, 1.0)
            lv = look[frame_idx] * arrow_scale
            quiv = ax.quiver(cur[0], cur[1], cur[2], lv[0], lv[1], lv[2], color="tab:green", linewidth=2)
            return traj_line, point

        ani = animation.FuncAnimation(fig, _update, frames=len(r), interval=1000.0 / max(self._fps, 1.0), blit=False)

        writer = animation.PillowWriter(fps=max(self._fps, 1.0))
        ani.save(self._output_path, writer=writer)
        plt.close(fig)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return ()


class NodeDataRecorderInputs(NamedTuple):
    position: InputPort
    velocity: InputPort
    q: InputPort
    w: InputPort
    q_err: InputPort
    tau_cmd_body: InputPort
    gravity_error: InputPort
    wheel_speed_0: InputPort
    wheel_speed_1: InputPort
    wheel_speed_2: InputPort
    wheel_speed_3: InputPort
    wheel_torque_0: InputPort
    wheel_torque_1: InputPort
    wheel_torque_2: InputPort
    wheel_torque_3: InputPort


class NodeDataRecorder(Node):
    def __init__(self, output_prefix: Path, **kwargs):
        self._i = NodeDataRecorderInputs(
            InputPort("position", self),
            InputPort("velocity", self),
            InputPort("q", self),
            InputPort("w", self),
            InputPort("q_err", self),
            InputPort("tau_cmd_body", self),
            InputPort("gravity_error", self),
            InputPort("wheel_speed_0", self),
            InputPort("wheel_speed_1", self),
            InputPort("wheel_speed_2", self),
            InputPort("wheel_speed_3", self),
            InputPort("wheel_torque_0", self),
            InputPort("wheel_torque_1", self),
            InputPort("wheel_torque_2", self),
            InputPort("wheel_torque_3", self),
        )
        self._output_prefix = Path(output_prefix)
        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        self._rows: list[dict[str, float]] = []

    def update(self, sim_time: float):
        pos = self._i.position.read()
        vel = self._i.velocity.read()
        q = self._i.q.read()
        w = self._i.w.read()
        q_err = self._i.q_err.read()
        tau_cmd = self._i.tau_cmd_body.read()
        g_err = self._i.gravity_error.read()
        ws = [
            self._i.wheel_speed_0.read(),
            self._i.wheel_speed_1.read(),
            self._i.wheel_speed_2.read(),
            self._i.wheel_speed_3.read(),
        ]
        wt = [
            self._i.wheel_torque_0.read(),
            self._i.wheel_torque_1.read(),
            self._i.wheel_torque_2.read(),
            self._i.wheel_torque_3.read(),
        ]
        if any(v is None for v in ([pos, vel, q, w, q_err, tau_cmd, g_err] + ws + wt)):
            return

        row = {
            "t": sim_time,
            "rx": float(pos[0]),
            "ry": float(pos[1]),
            "rz": float(pos[2]),
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "vz": float(vel[2]),
            "qw": float(q[0]),
            "qx": float(q[1]),
            "qy": float(q[2]),
            "qz": float(q[3]),
            "wx": float(w[0]),
            "wy": float(w[1]),
            "wz": float(w[2]),
            "qerrx": float(q_err[0]),
            "qerry": float(q_err[1]),
            "qerrz": float(q_err[2]),
            "tau_cmd_x": float(tau_cmd[0]),
            "tau_cmd_y": float(tau_cmd[1]),
            "tau_cmd_z": float(tau_cmd[2]),
            "g_err_x": float(g_err[0]),
            "g_err_y": float(g_err[1]),
            "g_err_z": float(g_err[2]),
            "ws0": float(ws[0][0]) if ws[0] is not None else 0.0,
            "ws1": float(ws[1][0]) if ws[1] is not None else 0.0,
            "ws2": float(ws[2][0]) if ws[2] is not None else 0.0,
            "ws3": float(ws[3][0]) if ws[3] is not None else 0.0,
            "wt0": float(wt[0]),
            "wt1": float(wt[1]),
            "wt2": float(wt[2]),
            "wt3": float(wt[3]),
        }
        self._rows.append(row)

    def _plot_components_with_magnitude(
        self,
        t: np.ndarray,
        components: np.ndarray,
        labels: tuple[str, str, str],
        title: str,
        ylabel: str,
        output_path: Path,
    ):
        mag = np.linalg.norm(components, axis=1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, components[:, 0], label=labels[0])
        ax.plot(t, components[:, 1], label=labels[1])
        ax.plot(t, components[:, 2], label=labels[2])
        ax.plot(t, mag, "k--", linewidth=1.5, label="magnitude")
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _save_diagnostic_plots(self, arrs: dict[str, np.ndarray]):
        t = arrs["t"]

        wheel_speeds = np.column_stack([arrs["ws0"], arrs["ws1"], arrs["ws2"], arrs["ws3"]])
        wheel_torques = np.column_stack([arrs["wt0"], arrs["wt1"], arrs["wt2"], arrs["wt3"]])

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for i in range(4):
            axs[0].plot(t, wheel_speeds[:, i], label=f"wheel {i}")
        axs[0].plot(t, np.linalg.norm(wheel_speeds, axis=1), "k--", linewidth=1.5, label="magnitude")
        axs[0].set_ylabel("Speed [rad/s]")
        axs[0].set_title("Reaction Wheel Speeds")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc="best")

        for i in range(4):
            axs[1].plot(t, wheel_torques[:, i], label=f"wheel {i}")
        axs[1].plot(t, np.linalg.norm(wheel_torques, axis=1), "k--", linewidth=1.5, label="magnitude")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Torque [N m]")
        axs[1].set_title("Reaction Wheel Torques")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="best")
        fig.tight_layout()
        fig.savefig(self._output_prefix.parent / "reaction_wheels.png", dpi=150)
        plt.close(fig)

        tau_cmd = np.column_stack([arrs["tau_cmd_x"], arrs["tau_cmd_y"], arrs["tau_cmd_z"]])
        q_err = np.column_stack([arrs["qerrx"], arrs["qerry"], arrs["qerrz"]])
        self._plot_components_with_magnitude(
            t,
            tau_cmd,
            ("tau_cmd_x", "tau_cmd_y", "tau_cmd_z"),
            "Controller Torque Command",
            "Torque [N m]",
            self._output_prefix.parent / "controller_torque_command.png",
        )
        self._plot_components_with_magnitude(
            t,
            q_err,
            ("qerr_x", "qerr_y", "qerr_z"),
            "Controller Pointing Error",
            "Quaternion Error [-]",
            self._output_prefix.parent / "controller_pointing_error.png",
        )

        g_err = np.column_stack([arrs["g_err_x"], arrs["g_err_y"], arrs["g_err_z"]])
        self._plot_components_with_magnitude(
            t,
            g_err,
            ("g_err_x", "g_err_y", "g_err_z"),
            "Gravity Model Error vs Keplerian",
            "Acceleration Error [m/s^2]",
            self._output_prefix.parent / "gravity_error.png",
        )

    def finalize(self, fault_history=None):
        if not self._rows:
            return

        self._output_prefix.parent.mkdir(parents=True, exist_ok=True)

        keys = list(self._rows[0].keys())
        csv_path = self._output_prefix.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in self._rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")

        npz_path = self._output_prefix.with_suffix(".npz")
        arrs = {k: np.array([row[k] for row in self._rows], dtype=float) for k in keys}
        np.savez(npz_path, **arrs)
        self._save_diagnostic_plots(arrs)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return ()


class NodeFrameCollectorInputs(NamedTuple):
    image: InputPort
    mask: InputPort


class NodeFrameCollector(Node):
    def __init__(self, output_path: Path, output_mask_path: Path | None = None, fps: float = 20, **kwargs):
        self._i = NodeFrameCollectorInputs(InputPort("image", self), InputPort("mask", self))
        self._output_path = Path(output_path)
        self._output_mask_path = Path(output_mask_path) if output_mask_path is not None else None
        self._fps = fps
        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        self._image_frames: list[np.ndarray] = []
        self._mask_frames: list[np.ndarray] = []

    def update(self, sim_time: float):
        image = self._i.image.read()
        mask = self._i.mask.read()
        if image is not None:
            self._image_frames.append(np.asarray(image, dtype=np.uint8))
        if mask is not None:
            self._mask_frames.append(np.asarray(mask, dtype=np.uint8))

    def finalize(self, fault_history=None):
        if self._image_frames:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                self._output_path,
                self._image_frames,
                duration=1.0 / max(self._fps, 1e-6),
                loop=0,
            )
        if self._mask_frames and self._output_mask_path is not None:
            self._output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                self._output_mask_path,
                self._mask_frames,
                duration=1.0 / max(self._fps, 1e-6),
                loop=0,
            )

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return ()


@dataclass
class FlybyArtifacts:
    output_dir: Path
    trajectory_animation: Path
    log_csv: Path
    log_npz: Path
    rendered_video: Path | None


def _parse_start_datetime(start_date_utc: str | None) -> datetime:
    if start_date_utc is None:
        return datetime.now(timezone.utc)
    start_dt = datetime.fromisoformat(start_date_utc)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    return start_dt


def _build_camera_node(cfg: FlybyRunConfig, start_dt: datetime, name: str = "camera") -> NodeAsteroidCamera:
    cam = NodeAsteroidCamera(
        asteroid=cfg.asteroid.asteroid,
        resolution_width=cfg.output.camera_width,
        resolution_height=cfg.output.camera_height,
        fov=cfg.output.camera_fov_deg,
        spp=cfg.output.spp,
        use_integrator_mask=cfg.output.use_integrator_mask,
        look_at_origin=False,
        date=start_dt,
        name=name,
    )
    cam.frequency = cfg.output.render_fps
    return cam


def _connect_allocator_to_wheels(allocator: NodeTorqueAllocator, wheels: list[NodeReactionWheel]) -> None:
    for wheel in wheels:
        allocator.o.tau_cmd_wheel >> wheel.i.tau_cmd


def _connect_wheel_aggregator(wheels: list[NodeReactionWheel], aggregator: NodeWheelAggregator) -> None:
    wheels[0].o.h_rw >> aggregator.i.h_rw_0
    wheels[0].o.tau_rw >> aggregator.i.tau_rw_0
    wheels[1].o.h_rw >> aggregator.i.h_rw_1
    wheels[1].o.tau_rw >> aggregator.i.tau_rw_1
    wheels[2].o.h_rw >> aggregator.i.h_rw_2
    wheels[2].o.tau_rw >> aggregator.i.tau_rw_2
    wheels[3].o.h_rw >> aggregator.i.h_rw_3
    wheels[3].o.tau_rw >> aggregator.i.tau_rw_3


def build_flyby_rl_system(cfg: FlybyRunConfig) -> tuple[NodeSystem, FlybyArtifacts]:
    output_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    trajectory_animation = output_dir / "trajectory_look.gif"
    log_prefix = output_dir / "timeseries"

    gravity = NodeAsteroidGravity(
        asteroid=cfg.asteroid.asteroid,
        lmax=cfg.asteroid.gravity_lmax,
        name="gravity",
    )

    mu = gravity._gravity_model.gm

    r0, v0 = _hyperbolic_state_from_params(
        mu=mu,
        periapsis_radius_m=cfg.flyby.periapsis_radius_m,
        external_angle=cfg.flyby.external_angle_deg,
        true_anomaly_deg=cfg.flyby.true_anomaly0_deg,
        inbound_ra_deg=cfg.flyby.inbound_ra_deg,
        inbound_dec_deg=cfg.flyby.inbound_dec_deg,
        bplane_angle_deg=cfg.flyby.bplane_angle_deg,
    )

    translational = NodeHyperbolicDynamics(np.concatenate([r0, v0]), mu=mu, name="translational")
    allocator = NodeTorqueAllocator(name="allocator")
    wheels = [
        NodeReactionWheel(
            i,
            cfg.rwa,
            TETRAHEDRAL_WHEEL_AXES[i],
            name=f"wheel_{i}"
        )
        for i in range(4)
    ]
    aggregator = NodeWheelAggregator(name="aggregator")

    attitude = NodeAttitudeDynamics(
        inertia_kgm2=cfg.spacecraft.inertia_kgm2,
        x0=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        name="attitude",
    )
    look = NodeLookVector(cfg.spacecraft.boresight_body, name="look")
    gyroscope = NodeGyroscope(name="gyroscope")

    cam = _build_camera_node(cfg, _parse_start_datetime(cfg.sim.start_date_utc), name="camera")

    rl_action = ExternalInputNode(initial_value=np.zeros(3, dtype=float), name="rl_action_input")
    gyro_output = ExternalOutputNode(name="rl_gyro_output")
    camera_image_output = ExternalOutputNode(name="rl_camera_image_output")
    camera_mask_output = ExternalOutputNode(name="rl_camera_mask_output")
    camera_visible_output = ExternalOutputNode(name="rl_camera_visible_output")
    attitude_output = ExternalOutputNode(name="rl_attitude_output")
    position_output = ExternalOutputNode(name="rl_position_output")

    system = NodeSystem()
    for n in [
        gravity,
        translational,
        allocator,
    ] + wheels + [
        aggregator,
        attitude,
        look,
        gyroscope,
        cam,
        rl_action,
        gyro_output,
        camera_image_output,
        camera_mask_output,
        camera_visible_output,
        attitude_output,
        position_output,
    ]:
        system.add_node(n)

    # Translational dynamics feedback loop
    translational.o.position >> gravity.i.position
    gravity.o.gravity_accel >> translational.i.gravity_accel

    # Attitude guidance is handled externally via RL action input
    action_input = rl_action.o.out
    action_input >> allocator.i.tau_cmd_body

    _connect_allocator_to_wheels(allocator, wheels)
    _connect_wheel_aggregator(wheels, aggregator)

    # Attitude dynamics from aggregated wheel outputs
    aggregator.o.tau_rw_total >> attitude.i.tau_body
    aggregator.o.h_rw_total >> attitude.i.h_rw_body

    # Gyroscope measurement from attitude dynamics
    attitude.o.w >> gyroscope.i.angular_velocity

    # Look vector and camera feed
    attitude.o.q >> look.i.q
    translational.o.position >> look.i.position
    translational.o.position >> cam.i.camera_position
    look.o.look_target >> cam.i.camera_target

    # External observation taps
    gyroscope.o.measurement >> gyro_output.i.inp
    cam.o.image >> camera_image_output.i.inp
    cam.o.asteroid_mask >> camera_mask_output.i.inp
    cam.o.asteroid_visible >> camera_visible_output.i.inp
    attitude.o.q >> attitude_output.i.inp
    translational.o.position >> position_output.i.inp

    artifacts = FlybyArtifacts(
        output_dir=output_dir,
        trajectory_animation=trajectory_animation,
        log_csv=log_prefix.with_suffix(".csv"),
        log_npz=log_prefix.with_suffix(".npz"),
        rendered_video=None,
    )
    return system, artifacts


def build_flyby_system(cfg: FlybyRunConfig) -> tuple[NodeSystem, FlybyArtifacts]:
    output_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    trajectory_animation = output_dir / "trajectory_look.gif"
    log_prefix = output_dir / "timeseries"
    render_video = output_dir / "camera_render.gif"
    
    gravity = NodeAsteroidGravity(
        asteroid=cfg.asteroid.asteroid,
        lmax=cfg.asteroid.gravity_lmax,
        name="gravity",
    )

    mu = gravity._gravity_model.gm

    r0, v0 = _hyperbolic_state_from_params(
        mu=mu,
        periapsis_radius_m=cfg.flyby.periapsis_radius_m,
        external_angle=cfg.flyby.external_angle_deg,
        true_anomaly_deg=cfg.flyby.true_anomaly0_deg,
        inbound_ra_deg=cfg.flyby.inbound_ra_deg,
        inbound_dec_deg=cfg.flyby.inbound_dec_deg,
        bplane_angle_deg=cfg.flyby.bplane_angle_deg,
    )

    translational = NodeHyperbolicDynamics(np.concatenate([r0, v0]), mu=mu, name="translational")
    guidance = NodeCenterPointingGuidance(name="guidance")
    controller = NodeAttitudeController(
        kp=cfg.controller.kp,
        kd=cfg.controller.kd,
        ki=cfg.controller.ki,
        integral_limit=cfg.controller.integral_limit,
        inertia_kgm2=cfg.spacecraft.inertia_kgm2,
        name="controller",
    )
    
    # Create torque allocator and 4 reaction wheels in tetrahedral configuration
    allocator = NodeTorqueAllocator(name="allocator")
    wheels = [
        NodeReactionWheel(
            i,
            cfg.rwa,
            TETRAHEDRAL_WHEEL_AXES[i],
            name=f"wheel_{i}"
        )
        for i in range(4)
    ]
    aggregator = NodeWheelAggregator(name="aggregator")
    
    attitude = NodeAttitudeDynamics(
        inertia_kgm2=cfg.spacecraft.inertia_kgm2,
        x0=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        name="attitude",
    )
    look = NodeLookVector(cfg.spacecraft.boresight_body, name="look")

    gyroscope = NodeGyroscope(name="gyroscope")

    animator = NodeTrajectoryAnimator(
        output_path=trajectory_animation,
        fps=cfg.output.trajectory_fps,
        name="trajectory_animator",
    )
    animator.period = 100.0  # Update animation every 100 timesteps.
    recorder = NodeDataRecorder(output_prefix=log_prefix, name="recorder")

    system = NodeSystem()
    for n in [gravity, translational, guidance, controller, allocator] + wheels + [aggregator, attitude, look, gyroscope, animator, recorder]:
        system.add_node(n)

    # Translational dynamics feedback loop
    translational.o.position >> gravity.i.position
    gravity.o.gravity_accel >> translational.i.gravity_accel

    # Guidance loop
    translational.o.position >> guidance.i.position
    translational.o.velocity >> guidance.i.velocity

    # Attitude control loop
    guidance.o.q_cmd >> controller.i.q_cmd
    guidance.o.w_cmd >> controller.i.w_cmd
    attitude.o.q >> controller.i.q
    attitude.o.w >> controller.i.w
    aggregator.o.h_rw_total >> controller.i.h_rw

    # Allocation to wheels
    controller.o.tau_cmd_body >> allocator.i.tau_cmd_body

    _connect_allocator_to_wheels(allocator, wheels)
    _connect_wheel_aggregator(wheels, aggregator)

    # Attitude dynamics from aggregated wheel outputs
    aggregator.o.tau_rw_total >> attitude.i.tau_body
    aggregator.o.h_rw_total >> attitude.i.h_rw_body

    # Gyroscope measurement from attitude dynamics
    attitude.o.w >> gyroscope.i.angular_velocity

    # Look vector computation and animation
    attitude.o.q >> look.i.q
    translational.o.position >> look.i.position

    translational.o.position >> animator.i.position
    look.o.look_vec >> animator.i.look_vec

    # Data recording
    translational.o.position >> recorder.i.position
    translational.o.velocity >> recorder.i.velocity
    attitude.o.q >> recorder.i.q
    attitude.o.w >> recorder.i.w
    controller.o.q_err >> recorder.i.q_err
    controller.o.tau_cmd_body >> recorder.i.tau_cmd_body
    translational.o.gravity_error >> recorder.i.gravity_error
    wheels[0].o.omega >> recorder.i.wheel_speed_0
    wheels[1].o.omega >> recorder.i.wheel_speed_1
    wheels[2].o.omega >> recorder.i.wheel_speed_2
    wheels[3].o.omega >> recorder.i.wheel_speed_3
    wheels[0].o.wheel_torque >> recorder.i.wheel_torque_0
    wheels[1].o.wheel_torque >> recorder.i.wheel_torque_1
    wheels[2].o.wheel_torque >> recorder.i.wheel_torque_2
    wheels[3].o.wheel_torque >> recorder.i.wheel_torque_3

    rendered_video_path: Path | None = None
    if cfg.output.render_video:
        cam = _build_camera_node(cfg, _parse_start_datetime(cfg.sim.start_date_utc), name="camera")

        collector = NodeFrameCollector(
            output_path=render_video,
            output_mask_path=output_dir / "camera_mask_render.gif",
            fps=20,
            name="render_collector",
        )
        collector.frequency = cfg.output.render_fps

        system.add_node(cam)
        system.add_node(collector)

        translational.o.position >> cam.i.camera_position
        look.o.look_target >> cam.i.camera_target
        cam.o.image >> collector.i.image
        cam.o.asteroid_mask >> collector.i.mask

        rendered_video_path = render_video

    artifacts = FlybyArtifacts(
        output_dir=output_dir,
        trajectory_animation=trajectory_animation,
        log_csv=log_prefix.with_suffix(".csv"),
        log_npz=log_prefix.with_suffix(".npz"),
        rendered_video=rendered_video_path,
    )
    return system, artifacts


def run_flyby(cfg: FlybyRunConfig) -> FlybyArtifacts:
    system, artifacts = build_flyby_system(cfg)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)

    system.simulate(
        t_f=cfg.sim.duration_s,
        dt=cfg.sim.dt_s,
        save_dir=str(artifacts.output_dir),
        sim_name=cfg.output.run_name,
    )
    return artifacts
