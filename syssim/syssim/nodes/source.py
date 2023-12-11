from typing import Dict, Union, NamedTuple, Annotated
import numpy as np
from collections import namedtuple
from dataclasses import dataclass

from syssim.core import Node
from syssim.core import OutputPort
from syssim.core.port import InputPort, OutputPort

class NodeConstantOutputs(NamedTuple):

    constant_out: OutputPort
    """Constant output for node"""

class NodeConstant(Node):
    def __init__(self, value: np.array, **kwargs):
        """Node implementing a constant value output.

        Args:
            value (np.array): The constant value to output.

        Ports:
            output (np.array): output port for constant value
        """
        # ports = {"output": OutputPort("output", self)}
        self._v = value
        self._i = None
        self._o = NodeConstantOutputs(OutputPort("output_port", self))
        super().__init__((), self._o, **kwargs)

    def update(self, sim_time: float):
        self._o.constant_out.shift_out(self._v)

    @property
    def i(self):
        return ()

    @property
    def o(self):
        return self._o
