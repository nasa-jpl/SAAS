"""Generate a GIF of an asteroid orbit as seen by a camera node."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from syssim.core import NodeSystem

from .nodes import NodeAsteroidCamera, NodeAsteroidGravity, NodeOrbitDynamics, NodeOrbitFrameCollector


@dataclass
class OrbitConfig:
	asteroid: str = "Ceres"
	"""Asteroid model name to render."""
	camera_fps: float = 0.01  # Render 1 frame every 100 s.
	"""Camera render sampling frequency [Hz]."""
	sim_dt: float = 1.0
	"""Simulation step size [s]."""
	radius_scale: float = 2.0
	"""Circular orbit radius as a multiple of asteroid reference radius."""
	camera_fov_deg: float = 70.0
	"""Camera field of view [deg]."""
	spp: int = 8
	"""Mitsuba samples per pixel."""
	resolution_width: int = 384
	"""Rendered image width [px]."""
	resolution_height: int = 384
	"""Rendered image height [px]."""


def build_system(config: OrbitConfig) -> NodeSystem:
	system = NodeSystem()

	gravity_node = NodeAsteroidGravity(asteroid=config.asteroid, name="asteroid_gravity")
	mu = gravity_node._gravity_model.gm
	r_ref = gravity_node._gravity_model.r0

	orbit_radius = config.radius_scale * r_ref
	orbit_period = 2.0 * np.pi * np.sqrt(orbit_radius**3 / mu)

	position_0 = np.array([orbit_radius, 0.0, 0.0], dtype=float)
	velocity_0 = np.array([0.0, np.sqrt(mu / orbit_radius), 0.0], dtype=float)
	x0 = np.concatenate([position_0, velocity_0])

	orbit_node = NodeOrbitDynamics(
		x0=x0,
		name="orbit_dynamics",
	)

	camera_node = NodeAsteroidCamera(
		asteroid=config.asteroid,
		resolution_width=config.resolution_width,
		resolution_height=config.resolution_height,
		fov=config.camera_fov_deg,
		spp=config.spp,
		look_at_origin=True,
		name="asteroid_camera",
	)
	camera_node.frequency = config.camera_fps

	output_dir = Path("/tmp")
	collector_node = NodeOrbitFrameCollector(
		output_dir=output_dir,
		frame_hz=config.camera_fps,
		filename_prefix=f"orbit-{config.asteroid.lower()}",
		name="frame_collector",
	)

	system.add_node(orbit_node)
	system.add_node(gravity_node)
	system.add_node(camera_node)
	system.add_node(collector_node)

	orbit_node.o.position >> gravity_node.i.position
	gravity_node.o.gravity_accel >> orbit_node.i.gravity_accel
	orbit_node.o.position >> camera_node.i.camera_position
	camera_node.o.image >> collector_node.i.image

	system._orbit_period = orbit_period
	return system


def main():
	config = OrbitConfig()
	system = build_system(config)

	t_f = system._orbit_period / 4
	print(f"Simulating one orbit: {t_f:.1f} s")
	print(f"Camera frame rate: {config.camera_fps:.3f} Hz")
	print(f"Expected frames: {int(np.ceil(t_f * config.camera_fps))}")
	system.simulate(t_f=t_f, dt=config.sim_dt, sim_name="orbit_video")


if __name__ == "__main__":
	main()
