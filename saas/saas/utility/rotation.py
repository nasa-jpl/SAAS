from scipy.spatial.transform import Rotation
from typing import NamedTuple

from syssim import Node, InputPort, OutputPort
from syssim.core.port import InputPort, OutputPort

class NodeQuatRotationInputs(NamedTuple):
    in_vec: InputPort
    in_quat: InputPort

class NodeQuatRotationOutputs(NamedTuple):
    out_vec: OutputPort

class NodeQuatRotation(Node):
    def __init__(self, **kwargs):
        self._i = NodeQuatRotationInputs(
            InputPort("in_vec", self),
            InputPort("in_quat", self)
        )
        self._o = NodeQuatRotationOutputs(
            OutputPort("out_vec", self)
        )

        super().__init__(self._i, self._o, **kwargs)

    def update(self, sim_time: float):
        v = self._i.in_vec.read()
        q = self._i.in_quat.read()

        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        v_rotated = r.apply(v)

        self._o.out_vec.shift_out(v_rotated)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o
