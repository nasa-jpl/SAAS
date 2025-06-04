import numpy as np

from syssim import NodeDifferential, InputPort, OutputPort
from typing import NamedTuple


class NodeRWASimpleInputs(NamedTuple):
    tau_cmd: InputPort
    """Commanded torque input"""


class NodeRWASimpleOutputs(NamedTuple):
    rw_mtm: OutputPort
    """Reaction wheel angular momentum vector"""
    rw_speed: OutputPort
    """Output reaction wheel speed vector"""
    rw_torque: OutputPort
    """Output torque vector"""


class NodeRWASimple(NodeDifferential):
    def __init__(self, x0: float = 0, **kwargs):
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
        self._o = NodeRWASimpleOutputs(OutputPort("rw_mtm", self), OutputPort("rw_speed", self), OutputPort("rw_torque", self))

        super().__init__(x0, self._i, self._o, **kwargs)

    def initialize(self):
        self._inertia = float(self._config["rwa_inertia"])
        self._body_unit_vector = np.array(self._config['body_vector'], dtype=float)
        self._body_unit_vector /= np.linalg.norm(self._body_unit_vector)
        self._t = 0

    def update(self, sim_time: float):
        tau_cmd = self._i.tau_cmd.read()
        if np.any(tau_cmd) == None:
            tau_cmd = np.zeros((3,))

        dt = sim_time - self._t

        delta_speed = -tau_cmd / self._inertia
        self._x += delta_speed * (sim_time - self._t)

        self._t = sim_time

        self._o.rw_mtm.shift_out(self._x * self._inertia * self._body_unit_vector)
        self._o.rwa_tau.rw_torque(tau_cmd)
        self._o.rw_speed.shift_out(self._x)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o
