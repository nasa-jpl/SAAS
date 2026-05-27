from dataclasses import dataclass
from typing import Any

from syssim.core import EmptySpec, InputPort, Node, OutputPort, input_port, output_port


@dataclass
class ExternalInputNodeOutputs:
    """Output specification for ``ExternalInputNode``.

    Attributes
    ----------
    out : OutputPort
        Port that emits the externally assigned value.
    """

    out: OutputPort[Any] = output_port(Any)
    """Output port emitting the externally injected value on every update."""


class ExternalInputNode(Node[EmptySpec, ExternalInputNodeOutputs, EmptySpec, EmptySpec]):
    """Node that exposes an output port for external value injection.

    Parameters
    ----------
    initial_value : Any, optional
        Initial value emitted before the ``value`` property is changed.
    name : str, optional
        Node name used in systems, logs, and fully qualified port names.
    **kwargs
        Arguments forwarded to ``Node``.
    """

    Outputs = ExternalInputNodeOutputs

    def __init__(self, initial_value=None, name: str = None, **kwargs):
        self._value = initial_value
        super().__init__(name=name, **kwargs)

    @property
    def value(self):
        """Current value that will be emitted on the output port.

        Returns
        -------
        Any
            Current externally assigned value.
        """
        return self._value

    @value.setter
    def value(self, value):
        """Set the value emitted on the output port.

        Parameters
        ----------
        value : Any
            New externally assigned value.
        """
        self._value = value

    def update(self, sim_time: float):
        """Write the current external value to the output port.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds.
        """
        self.o.out.write(self._value, sim_time)


@dataclass
class ExternalOutputNodeInputs:
    """Input specification for ``ExternalOutputNode``.

    Attributes
    ----------
    inp : InputPort
        Port whose latest value is exposed through ``value``.
    """

    inp: InputPort[Any] = input_port(Any)
    """Input port whose value is captured and exposed via the ``value`` property after each step."""


class ExternalOutputNode(Node[ExternalOutputNodeInputs, EmptySpec, EmptySpec, EmptySpec]):
    """Node that exposes an input port for external observation.

    Parameters
    ----------
    name : str, optional
        Node name used in systems, logs, and fully qualified port names.
    **kwargs
        Arguments forwarded to ``Node``.
    """

    Inputs = ExternalOutputNodeInputs

    def __init__(self, name: str = None, **kwargs):
        self._last_value = None
        super().__init__(name=name, **kwargs)

    def update(self, sim_time: float):
        """Capture the latest input-port value.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds.
        """
        self._last_value = self.i.inp.read().value

    @property
    def value(self):
        """Last value observed at the input port.

        Returns
        -------
        Any
            Value captured during the most recent update.
        """
        return self._last_value
