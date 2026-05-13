from typing import List, Union
from copy import deepcopy


def _apply_fault_action(fault, value, timestamp):
    try:
        result = fault.action(value, timestamp)
    except TypeError:
        result = fault.action(value)

    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, timestamp

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
        self._t = None
        self._node = node
        self._faults = []
        self._connected_out_port: "OutputPort" = None

    def read(self):
        """Return the stored value."""
        return self._v

    def read_with_time(self):
        """Return the stored value and the simulation time it was produced."""
        return self._v, self._t

    def add_fault(self, fault):
        """Attach a fault to the port."""
        self._faults.append(fault)

    def _write(self, val, sim_time):
        """Write a value and time into the port applying active faults."""
        cval = deepcopy(val)
        ctime = sim_time
        for f in self._faults:
            if f.active:
                cval, ctime = _apply_fault_action(f, cval, ctime)
        self._v = cval
        self._t = ctime

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

    def shift_out(self, val, sim_time):
        """Propagate a value/time sample to all connected inputs with fault mutation."""
        cval = deepcopy(val)
        ctime = sim_time
        for f in self._faults:
            if f.active:
                cval, ctime = _apply_fault_action(f, cval, ctime)

        for p in self._input_ports:
            p._write(cval, ctime)

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

