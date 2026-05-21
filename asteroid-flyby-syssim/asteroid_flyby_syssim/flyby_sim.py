"""Asteroid flyby simulation with syssim nodes.

This module builds a syssim graph for a spacecraft performing a hyperbolic
asteroid flyby while keeping a camera boresight pointed at asteroid center
using reaction-wheel attitude control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from syssim.core import NodeSystem
from syssim.nodes.io import ExternalInputNode, ExternalOutputNode

from .nodes import (
    NodeAttitudeController,
    NodeAttitudeDynamics,
    NodeAsteroidCamera,
    NodeAsteroidGravity,
    NodeCenterPointingGuidance,
    NodeDataRecorder,
    NodeFrameCollector,
    NodeGyroscope,
    NodeHyperbolicDynamics,
    NodeLookVector,
    NodeReactionWheel,
    NodeTorqueAllocator,
    NodeTrajectoryAnimator,
    NodeWheelAggregator,
    TETRAHEDRAL_WHEEL_AXES,
)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _normalize_quaternion_wxyz(q_wxyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a unit quaternion; fall back to identity for invalid inputs."""
    q = np.asarray(q_wxyz, dtype=float)
    if q.shape != (4,):
        q = q.reshape(4)
    if not np.all(np.isfinite(q)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=float)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=float)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _rotation_from_wxyz(q_wxyz: np.ndarray) -> Rotation:
    q_wxyz = _normalize_quaternion_wxyz(q_wxyz)
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
    if prev_q_cmd_bi_wxyz is not None:
        prev_rot = _rotation_from_wxyz(prev_q_cmd_bi_wxyz)
        cmd_rot = _rotation_from_wxyz(q_cmd)
        delta = cmd_rot * prev_rot.inv()
        max_step = np.deg2rad(55.0)
        angle = delta.magnitude()
        if angle > max_step:
            limited_rot = Rotation.from_rotvec(delta.as_rotvec() * (max_step / angle)) * prev_rot
            q_cmd = _wxyz_from_rotation(limited_rot)
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
    """Total simulation duration [s]."""
    dt_s: float = 0.5
    """Base simulation step size [s]."""
    start_date_utc: str = "2028-01-01T00:00:00+00:00"
    """UTC start date used for ephemeris-derived rendering context."""


@dataclass
class AsteroidConfig:
    asteroid: str = "Ceres"
    """Asteroid model name: Ceres, Vesta, or Eros."""
    gravity_lmax: int | None = 12
    """Maximum spherical-harmonic gravity degree, or None for dataset default."""


@dataclass
class FlybyConfig:
    periapsis_radius_m: float = 8.0e5
    """Flyby periapsis radius from asteroid center [m]."""
    external_angle_deg: float = 140.0
    """Hyperbolic external angle [deg]."""
    true_anomaly0_deg: float = -90.0
    """Initial true anomaly [deg]."""
    inbound_ra_deg: float = 10.0
    """Inbound asymptote right ascension [deg]."""
    inbound_dec_deg: float = 10.0
    """Inbound asymptote declination [deg]."""
    bplane_angle_deg: float = 1.0
    """B-plane orientation angle [deg]."""


@dataclass
class SpacecraftConfig:
    inertia_kgm2: tuple[float, float, float] = (70.0, 60.0, 45.0)
    """Principal spacecraft moments of inertia [kg m^2]."""
    boresight_body: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Camera boresight unit vector in body coordinates."""


@dataclass
class ControllerConfig:
    kp: float = 0.037
    """Scalar proportional attitude-control gain."""
    kd: float = 1.2
    """Scalar derivative attitude-control gain."""
    ki: float = 0.0
    """Scalar integral attitude-control gain."""
    integral_limit: float = 0.2
    """Absolute clamp for each integrated attitude-error component."""


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
    """Wheel spin-axis inertias [kg m^2]."""
    torque_max_nm: tuple[float, float, float] = (0.12, 0.12, 0.12)
    """Maximum motor torque per modeled wheel axis [N m]."""
    wheel_speed_max_rads: tuple[float, float, float] = (6000.0 * 2.0 * np.pi / 60.0,) * 3
    """Maximum wheel angular speed [rad/s]."""
    momentum_max_nms: tuple[float, float, float] = (2.0, 2.0, 2.0)
    """Maximum stored wheel angular momentum [N m s]."""
    command_lag_tau_s: float = 0.05
    """First-order motor command lag time constant [s]."""
    viscous_friction_nms: float = 1.0e-5
    """Viscous bearing friction coefficient [N m s]."""
    coulomb_friction_nm: float = 2.0e-5
    """Coulomb bearing friction torque [N m]."""
    jitter_std_nm: float = 1.0e-5
    """Standard deviation of wheel jitter disturbance torque [N m]."""
    # 250 g-mm^2 = 2.5e-7 kg*m^2, used as a conservative imbalance torque coefficient upper bound.
    imbalance_coeff_nm_per_rads2: float = 2.5e-7
    """Wheel imbalance disturbance coefficient [N m / (rad/s)^2]."""
    imbalance_freq_hz: float = 37.0
    """Wheel imbalance disturbance frequency [Hz]."""


@dataclass
class OutputConfig:
    output_dir: str = "./outputs"
    """Directory where flyby artifacts are written."""
    run_name: str = "asteroid_flyby"
    """Subdirectory/run label for generated artifacts."""
    render_video: bool = True
    """Whether to render camera GIF output."""
    render_fps: float = 1/500
    """Rendered camera sampling frequency [Hz]."""
    trajectory_fps: float = 20.0
    """Playback frame rate for the trajectory animation [Hz]."""
    camera_width: int = 512
    """Rendered camera image width [px]."""
    camera_height: int = 512
    """Rendered camera image height [px]."""
    camera_fov_deg: float = 42.0
    """Rendered camera field of view [deg]."""
    spp: int = 16
    """Mitsuba samples per pixel."""
    use_integrator_mask: bool = False
    """Whether to render masks with the Mitsuba visibility integrator."""


@dataclass
class FlybyRunConfig:
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    """Simulation timing and epoch configuration."""
    asteroid: AsteroidConfig = field(default_factory=AsteroidConfig)
    """Asteroid body and gravity configuration."""
    flyby: FlybyConfig = field(default_factory=FlybyConfig)
    """Hyperbolic flyby geometry configuration."""
    spacecraft: SpacecraftConfig = field(default_factory=SpacecraftConfig)
    """Spacecraft inertia and camera boresight configuration."""
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    """Attitude controller gain configuration."""
    rwa: ReactionWheelConfig = field(default_factory=ReactionWheelConfig)
    """Reaction wheel actuator configuration."""
    output: OutputConfig = field(default_factory=OutputConfig)
    """Output artifact and rendering configuration."""


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
