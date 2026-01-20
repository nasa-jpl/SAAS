from typing import List, Union
from copy import deepcopy

class InputPort:
    def __init__(self, name: str, node: "Node"):
        """Input port owned by a node.

        Parameters
        ----------
        name : str
            Port name.
        node : Node
            Owning node.
        """
        self._name = name
        self._v = None
        self._node = node
        self._faults = []
        self._connected_out_port: "OutputPort" = None

    def read(self):
        """Return the stored value."""
        return self._v

    def add_fault(self, fault):
        """Attach a fault to the port."""
        self._faults.append(fault)

    def _write(self, val):
        """Write a value into the port applying active faults."""
        cval = deepcopy(val)
        for f in self._faults:
            if f.active:
                cval = f.action(cval)
        self._v = cval

    @property
    def name(self) -> str:
        """Port name."""
        return self._name

    @property
    def node(self) -> "Node":
        """Owning node."""
        return self._node

    @property
    def output_port(self) -> Union["OutputPort", None]:
        """Connected output port, if any."""
        return self._connected_out_port
    
    def __lshift__(lhs, rhs: "OutputPort"):
        # <<
        rhs.connect_input(lhs)


class OutputPort:
    def __init__(self, name: str, node: "Node"):
        """Output port owned by a node.

        Parameters
        ----------
        name : str
            Port name.
        node : Node
            Owning node.
        """
        self._name = name
        self._node = node
        self._input_ports: List[InputPort] = list()
        self._faults = []

    def shift_out(self, val):
        """Propagate a value to all connected inputs with fault mutation."""
        cval = deepcopy(val)
        for f in self._faults:
            if f.active:
                cval = f.action(val)

        for p in self._input_ports:
            p._write(cval)

    def connect_input(self, input_port: InputPort):
        """Connect this output port to an input port.

        Parameters
        ----------
        input_port : InputPort
            Input port to connect.

        Raises
        ------
        Exception
            If ``input_port`` is not an :class:`InputPort`.
        """
        if not isinstance(input_port, InputPort):
            raise Exception("Must connect OutputPort to InputPort")
        self._input_ports.append(input_port)
        input_port._connected_out_port = self

    def add_fault(self, fault):
        """Attach a fault to the port."""
        self._faults.append(fault)

    @property
    def name(self) -> str:
        """Port name."""
        return self._name

    @property
    def node(self) -> "Node":
        """Owning node."""
        return self._node
    
    def __rshift__(lhs, rhs: "InputPort"):
        # >>
        lhs.connect_input(rhs)

