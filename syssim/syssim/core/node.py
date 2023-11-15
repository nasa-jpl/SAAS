from typing import Union, Dict, List
from copy import deepcopy

import toml
from numpy import array

from syssim.core.port import InputPort, OutputPort


class Node:
    """A class representing a single unit of behavior in a simulation. This can be anything from a basic operation, like addition or multiplication, to a complicated model of a component, such as an inertial measurement unit, or physcial phenomenon or much more."""

    def __init__(
        self,
        ports: Dict[str, Union[InputPort, OutputPort]],
        config: str = None,
        sample_frequency=None,
        sample_period=None,
        name: str = None,
    ):
        """A class representing a single unit of behavior in a simulation. This can be anything from a basic operation, like addition or multiplication, to a complicated model of a component, such as an inertial measurement unit, or physcial phenomenon or much more.

        Args:
            ports (Dict[str, Union[InputPort, OutputPort]]): Dictionary of ports. Maps port names as a string to the port object itself.
            config (str, optional): Path to a TOML file containing the configuration values for this node. Defaults to None.
            sample_frequency (float, optional): Frequency at which this node is updated in the simulation. Superceeds setting sample_period through the constructor. Defaults to None.
            sample_period (float, optional): Period between node updates in the simulation. Defaults to None.
            name (str, optional): Name of the node. Used to map configuration values and faults to this node. Defaults to None.
        """
        self._ports = ports
        if config != None:
            self._full_config = toml.load(config)
        else:
            self._full_config = {}

        if sample_frequency != None:
            self._period = 1 / sample_frequency
        elif sample_period != None:
            self._period = sample_period
        else:
            self._period = None

        self._name = name

        if isinstance(self._name, str):
            try:
                self._config = self._full_config[self._name]
            except KeyError:
                self._config = {}
        else:
            self._config = {}

        self._system: "NodeSystem" = None

    def __getitem__(self, key: str) -> Union[InputPort, OutputPort]:
        """Get a port from this node by name.

        Args:
            key (str): Name of the port

        Returns:
            Union[InputPort, OutputPort]: Port with the given name
        """
        return self._ports[key]

    def __setitem__(self, key: str, port: InputPort):
        """Connect an input port to an output port in this node

        Args:
            key (str): name of the output port
            port (InputPort): the input port to connect to the named output port
        """
        self._ports[key].connect_input(port)

    def initialize(self):
        """Method stub for initializing the node at the start of a simulation. All initialization actions should be done here when subclassing the node. This ensures that the node will be properlly reset in simulations that run multiple batches."""
        pass

    def finalize(self):
        """Method stub for finalizing the node. This is a good place to perform one-shot tasks or analysis of data collected by the node. For instance, one could display a plot from this method."""
        pass

    def update(self, sim_time: float):
        """Method stub for updating a node during the simulation. At a schedule determined by the nodes frequency, this method is called to allow the node to process inputs and produce outputs.

        Args:
            sim_time (float): the current simulation time.
        """
        pass

    def depends(self) -> List["Node"]:
        """Enumerate the nodes on which this node depends. These nodes will be updated before this node.

        Returns:
            List[Node]: the nodes in the system on which this node depends
        """
        deps = list()
        for p in self._ports.values():
            if isinstance(p, InputPort):
                # TODO Issue a warning for unconnected ports?
                if p.output_port != None and p.output_port.node not in deps:
                    deps.append(p.output_port.node)
        return deps

    def ports(self) -> List[Union[InputPort, OutputPort]]:
        """Get this node's ports

        Returns:
            List[Union[InputPort, OutputPort]]: this node's ports
        """
        return self._ports

    @property
    def period(self) -> float:
        """Period at whcih this node is updated in [s]

        Returns:
            float: period
        """
        return self._period

    @period.setter
    def period(self, value: float):
        self._period = value

    @property
    def frequency(self) -> Union[float, None]:
        """The frequency at which this node is updated in [Hz]

        Returns:
            Union[float, None]: frequency
        """
        if self._period is None:
            return None
        else:
            return 1 / self._period

    @frequency.setter
    def frequency(self, value: float):
        self._period = 1 / value

    @property
    def n_inputs(self) -> int:
        """Number of input ports in this node

        Returns:
            int: num inputs
        """
        n = 0
        for p in self._ports.values():
            if isinstance(p, InputPort):
                n += 1
        return n

    @property
    def n_outputs(self) -> int:
        """number of output ports in this node

        Returns:
            int: num outputs
        """
        n = 0
        for p in self._ports.values():
            if isinstance(p, OutputPort):
                n += 1
        return n

    @property
    def name(self) -> str:
        """The name of this node

        Returns:
            str: name
        """
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
        if isinstance(name, str):
            try:
                self._config = self._full_config[name]
            except KeyError:
                self._config = {}


class NodeDifferential(Node):
    """A class representing any node that models a differential equation. These nodes are special in that they may have inputs, but are assumed to only depend on the input values from the simulation step before the one they are currently updating too. This means that they do not have dependencies for the purpose of solving for a node update order."""

    def __init__(
        self, x0: array, ports: Dict[str, Union[InputPort, OutputPort]], **kwargs
    ):
        """A class representing any node that models a differential equation. These nodes are special in that they may have inputs, but are assumed to only depend on the input values from the simulation step before the one they are currently updating too. This means that they do not have dependencies for the purpose of solving for a node update order.

        Args:
            x0 (array): The initial internal state of this node. This is the state used in the differential equation the node models.
            ports (Dict[str, Union[InputPort, OutputPort]]): Dictionary of ports. Maps port names as a string to the port object itself.
        """

        self._x = x0
        self._x0 = x0
        super().__init__(ports, **kwargs)

    def depends(self) -> List[Node]:
        # Differential blocks have no dependencies...
        return list()

    def finalize(self):
        self._x = deepcopy(self._x0)
        return super().finalize()
