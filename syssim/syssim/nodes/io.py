from dataclasses import dataclass
from typing import Any

from syssim.core import EmptySpec, InputPort, Node, OutputPort, input_port, output_port


@dataclass
class ExternalInputNodeOutputs:
    out: OutputPort[Any] = output_port(Any)
    """Output port emitting the externally injected value on every update."""


class ExternalInputNode(Node[EmptySpec, ExternalInputNodeOutputs, EmptySpec, EmptySpec]):
    """Node that exposes an output port for external value injection.""" 

    Outputs = ExternalInputNodeOutputs

    def __init__(self, initial_value=None, name: str = None, **kwargs):
        self._value = initial_value
        super().__init__(name=name, **kwargs)

    @property
    def value(self):
        """Current value that will be emitted on the output port."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def update(self, sim_time: float):
        self.o.out.write(self._value, sim_time)


@dataclass
class ExternalOutputNodeInputs:
    inp: InputPort[Any] = input_port(Any)
    """Input port whose value is captured and exposed via the ``value`` property after each step."""


class ExternalOutputNode(Node[ExternalOutputNodeInputs, EmptySpec, EmptySpec, EmptySpec]):
    """Node that exposes an input port for external observation after each step.""" 

    Inputs = ExternalOutputNodeInputs

    def __init__(self, name: str = None, **kwargs):
        self._last_value = None
        super().__init__(name=name, **kwargs)

    def update(self, sim_time: float):
        self._last_value = self.i.inp.read().value

    @property
    def value(self):
        """Last value observed at the input port after the most recent step."""
        return self._last_value
