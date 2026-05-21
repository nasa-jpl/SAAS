"""Guidance and camera look-vector nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from syssim.core import EmptySpec, InputPort, Node, OutputPort, input_port, output_port

from .common import center_pointing_quaternion_wxyz, normalize, rotation_from_wxyz


@dataclass
class NodeCenterPointingGuidanceInputs:
    """Input ports for asteroid center-pointing guidance."""

    position: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft position relative to asteroid center [m]."""
    velocity: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft velocity relative to asteroid center [m/s]."""


@dataclass
class NodeCenterPointingGuidanceOutputs:
    """Output ports for center-pointing guidance commands."""

    q_cmd: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(4,))
    """Commanded body-to-inertial quaternion in scalar-first order [w, x, y, z]."""
    w_cmd: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Commanded body angular velocity [rad/s]."""
    look_dir_cmd: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Commanded inertial look direction from spacecraft toward asteroid center."""


class NodeCenterPointingGuidance(
    Node[NodeCenterPointingGuidanceInputs, NodeCenterPointingGuidanceOutputs, EmptySpec, EmptySpec]
):
    """Generate attitude commands that keep the body boresight on asteroid center."""

    Inputs = NodeCenterPointingGuidanceInputs
    Outputs = NodeCenterPointingGuidanceOutputs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._i = self.i
        self._o = self.o

    def initialize(self):
        self._q_cmd_prev_bi = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def update(self, sim_time: float):
        position_sc_i_m = self.i.position.read().value
        velocity_sc_i_mps = self.i.velocity.read().value

        if position_sc_i_m is None or np.linalg.norm(position_sc_i_m) < 1e-8:
            look_dir_i = np.array([1.0, 0.0, 0.0], dtype=float)
            q_cmd_bi = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            look_dir_i = normalize(-position_sc_i_m)
            q_cmd_bi = center_pointing_quaternion_wxyz(
                position_sc_i_m=position_sc_i_m,
                velocity_sc_i_mps=velocity_sc_i_mps,
                prev_q_cmd_bi_wxyz=self._q_cmd_prev_bi,
            )
        self._q_cmd_prev_bi = q_cmd_bi

        self.o.q_cmd.write(q_cmd_bi, sim_time)
        self.o.w_cmd.write(np.zeros(3, dtype=float), sim_time)
        self.o.look_dir_cmd.write(look_dir_i, sim_time)


@dataclass
class NodeLookVectorInputs:
    """Input ports for converting attitude to camera look geometry."""

    q: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(4,))
    """Body-to-inertial attitude quaternion in scalar-first order [w, x, y, z]."""
    position: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft position relative to asteroid center [m]."""


@dataclass
class NodeLookVectorOutputs:
    """Output ports for camera pointing geometry."""

    look_vec: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Unit camera boresight vector expressed in inertial coordinates."""
    look_target: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Point in inertial/body-aligned coordinates that the camera should look at."""


class NodeLookVector(Node[NodeLookVectorInputs, NodeLookVectorOutputs, EmptySpec, EmptySpec]):
    """Compute inertial camera look vector and target from spacecraft attitude."""

    Inputs = NodeLookVectorInputs
    Outputs = NodeLookVectorOutputs

    def __init__(self, boresight_body: tuple[float, float, float], **kwargs):
        self._boresight_body = normalize(np.array(boresight_body, dtype=float))
        super().__init__(**kwargs)
        self._i = self.i
        self._o = self.o

    def update(self, sim_time: float):
        q = self.i.q.read().value
        pos = self.i.position.read().value
        if q is None:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if pos is None:
            pos = np.zeros(3, dtype=float)

        look_vec = normalize(rotation_from_wxyz(q).apply(self._boresight_body))
        look_target = pos + max(np.linalg.norm(pos), 1.0) * look_vec

        self.o.look_vec.write(look_vec, sim_time)
        self.o.look_target.write(look_target, sim_time)
