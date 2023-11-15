from typing import List, Union
from copy import deepcopy


class InputPort:
    def __init__(self, name: str, node: "Node"):
        """Represents an input port into a node. Stores values during a simulation and allows those values to be updated by a connected output port.

        Args:
            name (str): name of the input port
            node (Node): a reference to the node to which the port belongs
        """
        self._name = name
        self._v = None
        self._node = node
        self._faults = []
        self._connected_out_port: "OutputPort" = None

    def read(self):
        """Get the value currently stored in this port

        Returns:
            any: the ports value
        """
        return self._v

    def add_fault(self, fault):
        """Register a fault onto this port.

        Args:
            fault (any): An object reprsenting the fault (i.e., FaultBasic())
        """
        self._faults.append(fault)

    def _write(self, val):
        """Write a new value to this port. May be overriden by an active fault.

        Args:
            val (any): The new value to write to the port.
        """
        cval = deepcopy(val)
        for f in self._faults:
            if f.is_active():
                f.action(cval)
        self._v = cval

    @property
    def name(self) -> str:
        """The name of the port.

        Returns:
            str: name
        """
        return self._name

    @property
    def node(self) -> "Node":
        """the node that owns this port

        Returns:
            Node: node
        """
        return self._node

    @property
    def output_port(self) -> Union["OutputPort", None]:
        """The output port connected to this input port

        Returns:
            Union[OutputPort, None]: the output port
        """
        return self._connected_out_port


class OutputPort:
    def __init__(self, name: str, node: "Node"):
        """Represents an output port for a node. Used to shift out updated values to connected input ports during a simulation.

        Args:
            name (str): name of the output port
            node (Node): the node that owns this output port
        """
        self._name = name
        self._node = node
        self._input_ports: List[InputPort] = list()
        self._faults = []

    def shift_out(self, val):
        """Write a new value out to all connected input ports. Value may be overriden by an active fault.

        Args:
            val (any): the value to shift out
        """
        for f in self._faults:
            if f.is_active():
                f.action(val)

        for p in self._input_ports:
            p._write(val)

    def connect_input(self, input_port: InputPort):
        """Connect this output port to an input port. One output may be connected to many inputs.

        Args:
            input_port (InputPort): port to connect

        Raises:
            Exception: Fails if you do not connect to another input port
        """
        if not isinstance(input_port, InputPort):
            raise Exception("Must connect OutputPort to InputPort")
        self._input_ports.append(input_port)
        input_port._connected_out_port = self

    def add_fault(self, fault):
        """Register a fault onto this port.

        Args:
            fault (any): An object reprsenting the fault (i.e., FaultBasic())
        """
        self._faults.append(fault)

    @property
    def name(self) -> str:
        """Name of this port

        Returns:
            str: the name
        """
        return self._name

    @property
    def node(self) -> "Node":
        """the node that owns this port

        Returns:
            Node: the node
        """
        return self._node
