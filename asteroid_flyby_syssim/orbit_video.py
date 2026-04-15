"""Generate a GIF of an asteroid orbit as seen by a camera node."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import imageio.v2 as imageio
import numpy as np

from syssim.core import Node, NodeDifferential, NodeSystem, InputPort, OutputPort

from asteroid_camera import NodeAsteroidCamera
from asteroid_gravity import NodeAsteroidGravity


@dataclass
class OrbitConfig:
	asteroid: str = "Ceres"
	camera_fps: float = 0.01  # Render 1 frame every 100 s.
	sim_dt: float = 1.0
	radius_scale: float = 2.0
	camera_fov_deg: float = 70.0
	spp: int = 8
	resolution_width: int = 384
	resolution_height: int = 384


class NodeOrbitDynamicsInputs(NamedTuple):
	gravity_accel: InputPort


class NodeOrbitDynamicsOutputs(NamedTuple):
	position: OutputPort
	velocity: OutputPort


class NodeOrbitDynamics(NodeDifferential):
	"""Integrate orbital dynamics using Newtonian motion."""

	def __init__(self, x0: np.ndarray, **kwargs):
		gravity_accel = InputPort("gravity_accel", self)
		position = OutputPort("position", self)
		velocity = OutputPort("velocity", self)

		self._i = NodeOrbitDynamicsInputs(gravity_accel)
		self._o = NodeOrbitDynamicsOutputs(position, velocity)

		super().__init__(x0, self._i, self._o, **kwargs)

	def initialize(self):
		self._t = 0.0

	def update(self, sim_time: float):
		accel = self._i.gravity_accel.read()
		if accel is None or np.any(np.isnan(accel)):
			accel = np.zeros(3, dtype=float)

		dt = sim_time - self._t
		if dt <= 0.0:
			return

		r = self._x[0:3]
		v = self._x[3:6]

		v_new = v + accel * dt
		r_new = r + v_new * dt

		self._x = np.concatenate([r_new, v_new])
		self._t = sim_time

		self._o.position.shift_out(r_new)
		self._o.velocity.shift_out(v_new)

	@property
	def i(self):
		return self._i

	@property
	def o(self):
		return self._o


class NodeFrameCollector(Node):
	"""Collect frames and write a GIF on finalize."""

	class Inputs(NamedTuple):
		image: InputPort

	def __init__(self, output_dir: Path, frame_hz: float, filename_prefix: str, **kwargs):
		self._i = self.Inputs(InputPort("image", self))
		self._frames: list[np.ndarray] = []
		self._output_dir = Path(output_dir)
		self._frame_hz = float(frame_hz)
		self._filename_prefix = filename_prefix
		super().__init__(self._i, (), **kwargs)

	def update(self, sim_time: float):
		frame = self._i.image.read()
		if frame is None:
			return
		self._frames.append(self._to_uint8(frame))

	def finalize(self, fault_history=None):
		super().finalize(fault_history=fault_history)
		if not self._frames:
			print("No frames captured. GIF not created.")
			return
		self._output_dir.mkdir(parents=True, exist_ok=True)
		timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		gif_path = self._output_dir / f"{self._filename_prefix}-{timestamp}.gif"
		duration = 1.0 / max(self._frame_hz, 1e-9)
		imageio.mimsave(gif_path, self._frames, duration=3.0)
		print(f"GIF saved to {gif_path}")

	@staticmethod
	def _to_uint8(frame: np.ndarray) -> np.ndarray:
		arr = np.asarray(frame)
		if arr.dtype == np.uint8:
			return arr
		if np.issubdtype(arr.dtype, np.floating):
			max_val = float(np.nanmax(arr)) if arr.size else 1.0
			if max_val <= 1.0:
				arr = arr * 255.0
			else:
				arr = arr * (255.0 / max_val)
		arr = np.clip(arr, 0, 255)
		return arr.astype(np.uint8)

	@property
	def i(self):
		return self._i

	@property
	def o(self):
		return ()


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
	collector_node = NodeFrameCollector(
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
