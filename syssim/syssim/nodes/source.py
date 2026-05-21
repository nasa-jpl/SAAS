from dataclasses import dataclass
from typing import Any

import numpy as np

from syssim.core import EmptySpec, Node, OutputPort, output_port


@dataclass
class NodeConstantOutputs:
    constant_out: OutputPort[Any] = output_port(Any)
    """Output port emitting the configured constant value on every update."""


class NodeConstant(Node[EmptySpec, NodeConstantOutputs, EmptySpec, EmptySpec]):
    Outputs = NodeConstantOutputs

    def __init__(self, value, **kwargs):
        """Node producing a constant value each update.

        Parameters
        ----------
        value : Any
            Constant payload to emit on every update.
        """
        self.value = np.asarray(value) if isinstance(value, (list, tuple)) else value
        super().__init__(**kwargs)

    def update(self, sim_time: float):
        self.o.constant_out.write(self.value, sim_time)
