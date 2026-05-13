from typing import NamedTuple, Optional

from syssim.core import Node, InputPort, OutputPort


class ExternalInputNodeOutputs(NamedTuple):
    out: OutputPort
    """Output port that forwards externally assigned values to connected nodes."""


class ExternalInputNode(Node):
    """Node that exposes an output port for external value injection.""" 

    def __init__(self, initial_value=None, name: str = None, **kwargs):
        self._value = initial_value
        self._o = ExternalInputNodeOutputs(OutputPort("out", self))
        super().__init__((), self._o, name=name, **kwargs)

    def initialize(self):
        self._value = self._value

    @property
    def value(self):
        """Current value that will be emitted on the output port."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def update(self, sim_time: float):
        self._o.out.shift_out(self._value, sim_time)

    @property
    def i(self):
        return ()

    @property
    def o(self):
        return self._o


class ExternalOutputNodeInputs(NamedTuple):
    inp: InputPort
    """Input port that receives values from the connected system port."""


class ExternalOutputNode(Node):
    """Node that exposes an input port for external observation after each step.""" 

    def __init__(self, name: str = None, **kwargs):
        self._last_value = None
        self._i = ExternalOutputNodeInputs(InputPort("in", self))
        super().__init__(self._i, (), name=name, **kwargs)

    def update(self, sim_time: float):
        self._last_value = self._i.inp.read()

    @property
    def value(self):
        """Last value observed at the input port after the most recent step."""
        return self._last_value

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return ()
