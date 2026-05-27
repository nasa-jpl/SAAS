from dataclasses import dataclass
from typing import Any

import numpy as np

from syssim.core import EmptySpec, Node, OutputPort, output_port


@dataclass
class NodeConstantOutputs:
    """Output specification for ``NodeConstant``.

    Attributes
    ----------
    constant_out : OutputPort
        Output port that emits the configured constant value.
    """

    constant_out: OutputPort[Any] = output_port(Any)
    """Output port emitting the configured constant value on every update."""


class NodeConstant(Node[EmptySpec, NodeConstantOutputs, EmptySpec, EmptySpec]):
    """Node producing a constant value at each update.

    Parameters
    ----------
    value : Any
        Constant payload to emit. Lists and tuples are converted to NumPy
        arrays.
    **kwargs
        Arguments forwarded to ``Node``.
    """

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
        """Write the configured constant to the output port.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds.
        """
        self.o.constant_out.write(self.value, sim_time)
