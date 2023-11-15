from typing import Dict, Union
import numpy as np

from syssim.core import Node
from syssim.core import OutputPort
from syssim.core.port import InputPort, OutputPort


class NodeConstant(Node):
    def __init__(self, value: np.array, **kwargs):
        """Node implementing a constant value output.

        Args:
            value (np.array): The constant value to output.

        Ports:
            output (np.array): output port for constant value
        """
        ports = {"output": OutputPort("output", self)}
        self._v = value
        super().__init__(ports, **kwargs)

    def update(self, sim_time: float):
        self._ports["output"].shift_out(self._v)
