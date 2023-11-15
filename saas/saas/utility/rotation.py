from scipy.spatial.transform import Rotation

from syssim import Node, InputPort, OutputPort
from syssim.core.port import InputPort, OutputPort


class NodeQuatRotation(Node):
    def __init__(self, **kwargs):
        in_vec = InputPort("in_vec", self)
        in_quat = InputPort("in_quat", self)

        out_vec = OutputPort("out_vec", self)

        ports = {
            in_vec.name: in_vec,
            in_quat.name: in_quat,
            out_vec.name: out_vec,
        }
        super().__init__(ports, **kwargs)

    def update(self, sim_time: float):
        v = self._ports["in_vec"].read()
        q = self._ports["in_quat"].read()

        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        v_rotated = r.apply(v)

        self._ports["out_vec"].shift_out(v_rotated)
