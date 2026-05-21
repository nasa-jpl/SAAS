"""Orbit-video support nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from syssim.core import EmptySpec, InputPort, Node, NodeDifferential, OutputPort, input_port, output_port


@dataclass
class NodeOrbitDynamicsInputs:
    """Input ports for simple orbit dynamics."""

    gravity_accel: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Asteroid gravity acceleration [m/s^2]."""


@dataclass
class NodeOrbitDynamicsOutputs:
    """Output ports for simple orbit dynamics."""

    position: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Orbiting camera position relative to asteroid center [m]."""
    velocity: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Orbiting camera velocity relative to asteroid center [m/s]."""


class NodeOrbitDynamics(NodeDifferential[np.ndarray, NodeOrbitDynamicsInputs, NodeOrbitDynamicsOutputs, EmptySpec, EmptySpec]):
    """Integrate simple Newtonian orbital dynamics."""

    Inputs = NodeOrbitDynamicsInputs
    Outputs = NodeOrbitDynamicsOutputs

    def __init__(self, x0: np.ndarray, **kwargs):
        super().__init__(np.asarray(x0, dtype=float), **kwargs)
        self._i = self.i
        self._o = self.o

    def initialize(self):
        self._t = 0.0
        self.reset_state()

    def update(self, sim_time: float):
        accel = self.i.gravity_accel.read().value
        if accel is None or np.any(np.isnan(accel)):
            accel = np.zeros(3, dtype=float)

        dt = sim_time - self._t
        if dt <= 0.0:
            return

        r = self.state[0:3]
        v = self.state[3:6]
        v_new = v + accel * dt
        r_new = r + v_new * dt
        self.state = np.concatenate([r_new, v_new])
        self._t = sim_time

        self.o.position.write(r_new, sim_time)
        self.o.velocity.write(v_new, sim_time)


@dataclass
class NodeOrbitFrameCollectorInputs:
    """Input ports for orbit GIF frame collection."""

    image: InputPort[np.ndarray] = input_port(np.ndarray)
    """Rendered RGB camera frame."""


class NodeOrbitFrameCollector(Node[NodeOrbitFrameCollectorInputs, EmptySpec, EmptySpec, EmptySpec]):
    """Collect rendered orbit frames and write a GIF on finalize."""

    Inputs = NodeOrbitFrameCollectorInputs

    def __init__(self, output_dir: Path, frame_hz: float, filename_prefix: str, **kwargs):
        self._frames: list[np.ndarray] = []
        self._output_dir = Path(output_dir)
        self._frame_hz = float(frame_hz)
        self._filename_prefix = filename_prefix
        super().__init__(**kwargs)
        self._i = self.i

    def update(self, sim_time: float):
        del sim_time
        frame = self.i.image.read().value
        if frame is not None:
            self._frames.append(self._to_uint8(frame))

    def finalize(self, fault_history=None):
        super().finalize(fault_history=fault_history)
        if not self._frames:
            print("No frames captured. GIF not created.")
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        gif_path = self._output_dir / f"{self._filename_prefix}-{timestamp}.gif"
        imageio.mimsave(gif_path, self._frames, duration=1.0 / max(self._frame_hz, 1e-9))
        print(f"GIF saved to {gif_path}")

    @staticmethod
    def _to_uint8(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.dtype == np.uint8:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.nanmax(arr)) if arr.size else 1.0
            arr = arr * 255.0 if max_val <= 1.0 else arr * (255.0 / max_val)
        return np.clip(arr, 0, 255).astype(np.uint8)
