import numpy as np
from syssim import Node, InputPort, OutputPort
from typing import NamedTuple

class AdderNodeOutputs(NamedTuple):
    sum: OutputPort
    # OutputPort for the sum

class AdderNode(Node):
    def __init__(self, n_inputs: int, **kwargs):
        """A simple adder node that sums n input values."""
        self._inputs_type = NamedTuple(
            "AdderNodeInputs",
            [(f"input_{i}", InputPort) for i in range(n_inputs)]
        )
        self._i = self._inputs_type(*[InputPort(f"input_{i}", self) for i in range(n_inputs)])
        self._o = AdderNodeOutputs(OutputPort("sum", self))
        super().__init__(self._i, self._o, **kwargs)

    def update(self, sim_time: float):
        values = np.stack([inp.read() for inp in self._i])
        total = np.sum(values, axis=0)
        self._o.sum.shift_out(total)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o


class ConcatNodeOutputs(NamedTuple):
    concat: OutputPort
    # OutputPort for the concatenated result

class ConcatNode(Node):
    def __init__(self, n_inputs: int, **kwargs):
        """A node that concatenates n input values along axis 0."""
        self._inputs_type = NamedTuple(
            "ConcatNodeInputs",
            [(f"input_{i}", InputPort) for i in range(n_inputs)]
        )
        self._i = self._inputs_type(*[InputPort(f"input_{i}", self) for i in range(n_inputs)])
        self._o = ConcatNodeOutputs(OutputPort("concat", self))
        super().__init__(self._i, self._o, **kwargs)

    def update(self, sim_time: float):
        values = [inp.read() for inp in self._i]
        values = [v if v.ndim > 0 else np.expand_dims(v, axis=0) for v in values]
        concatenated = np.concatenate(values, axis=0)
        self._o.concat.shift_out(concatenated)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o