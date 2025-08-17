import numpy as np

from syssim import NodeDifferential, InputPort, OutputPort
from typing import NamedTuple

from syssim.core.node import NodeParameter


class NodeRWASimpleInputs(NamedTuple):
    tau_cmd: InputPort
    """Commanded torque input for this wheel (scalar)"""


class NodeRWASimpleOutputs(NamedTuple):
    rw_mtm: OutputPort
    """Reaction wheel angular momentum vector in body frame"""
    rw_speed: OutputPort
    """Output reaction wheel speed vector in body frame"""


class NodeRWASimple(NodeDifferential):

    class Parameters(NamedTuple):
        body_axis: NodeParameter

    def __init__(self, x0: float = 0, body_axis: list = [1.0, 0.0, 0.0], **kwargs):
        """A model of a simple reaction wheel assembly. Keeps track of internal angular momentum vector and just passes through the comanded torque. No saturation or noise.

        Args:
            x0 (np.array): Initial state. Represents the reaction wheel angular momentum vector.

        Ports:
            tau_cmd (np.array): the commanded torque input. 3x1 [Nm]
            rwa_mtm (np.array): the output reaction wheel angular momentum vector. 3x1 [Nms]
            rwa_tau (np.array): the output torque. 3x1 [Nm]

        Configs:
            rwa_inertia: diagonal of the reaction wheel assembly inertia moment. 3x1 [kg m^2]
        """
        self._i = NodeRWASimpleInputs(InputPort("tau_cmd", self))
        self._o = NodeRWASimpleOutputs(OutputPort("rw_mtm", self), OutputPort("rw_speed", self))
        body_unit_vector = np.array(body_axis, dtype=float)
        body_unit_vector /= np.linalg.norm(body_unit_vector)
        self._p = self.Parameters(NodeParameter("body_vector", body_unit_vector))

        super().__init__(x0, self._i, self._o, self._p, **kwargs)

    def initialize(self):
        self._inertia = float(self._config["rwa_inertia"])

    def update(self, sim_time: float):
        tau_cmd = self._i.tau_cmd.read()
        if np.any(tau_cmd) == None:
            tau_cmd = 0
        else:
            tau_cmd = np.dot(tau_cmd, self.p.body_axis.value)

        # Update the differential state
        dt = self.period
        delta_speed = -tau_cmd / self._inertia
        self._x += delta_speed * dt

        self._o.rw_mtm.shift_out(self._x * self._inertia * self.p.body_axis.value)
        self._o.rw_speed.shift_out(self._x)


    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o

    @property
    def p(self):
        return self._p