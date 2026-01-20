from typing import Any, Union, Dict, List, Tuple, NamedTuple
from copy import deepcopy
from abc import ABC, abstractmethod

import toml
from numpy import array

from syssim.core.fault import Fault
from syssim.core.port import InputPort, OutputPort


class NodeParameter:
    """Node parameters are used to store values for a node which represent nominally parametric values of the model which the node implements. However, these parameters may be faulted and thus may be overridden by the fault logic in a similar way to ports."""

    def __init__(self, name: str, value: Any):
        """Create a parameter that can be faulted.

        Parameters
        ----------
        name : str
            Parameter name.
        value : Any
            Stored value prior to any fault mutation.
        """
        self._name = name
        self._value = value
        self._faults: List[Fault] = []

    def add_fault(self, fault):
        """Attach a fault to this parameter.

        Parameters
        ----------
        fault : Fault
            Fault to evaluate whenever the parameter is read.
        """
        self._faults.append(fault)

    # getter and setters
    @property
    def value(self) -> Any:
        """Return the parameter value after active faults are applied.

        Returns
        -------
        Any
            Possibly fault-mutated value.
        """
        cval = deepcopy(self._value)
        for f in self._faults:
            if f.active:
                cval = f.action(cval)
        return cval

    @property
    def name(self) -> str:
        """Name of the parameter."""
        return self._name

    @name.setter
    def name(self, name: str):
        """Set the parameter name."""
        if not isinstance(name, str):
            raise TypeError("Parameter name must be a string")
        self._name = name


class Node(ABC):
    """Abstract base class for simulation nodes.

    Nodes own ports, optional parameters, and implement simulation lifecycle
    hooks. Subclasses define behavior in :meth:`initialize`, :meth:`update`,
    and :meth:`finalize`.
    """

    def __init__(
        self,
        input_ports: Union[NamedTuple, Tuple],
        output_ports: Union[NamedTuple, Tuple],
        parameters: Union[NamedTuple, Tuple] = (),
        config: str = None,
        sample_frequency=None,
        sample_period=None,
        name: str = None,
    ):
        """Construct a node.

        Parameters
        ----------
        input_ports : NamedTuple or tuple
            Input ports owned by the node.
        output_ports : NamedTuple or tuple
            Output ports owned by the node.
        parameters : NamedTuple or tuple, optional
            Node parameters that can also be faulted.
        config : str, optional
            Path to a TOML file with per-node configuration keyed by node name.
        sample_frequency : float, optional
            Update frequency in Hz; overrides ``sample_period`` when provided.
        sample_period : float, optional
            Update period in seconds.
        name : str, optional
            Node name, also used to pull configuration from the TOML file.
        """

        self._i = input_ports
        self._o = output_ports
        self._p = parameters

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
        """Return a port by name.

        Parameters
        ----------
        key : str
            Port name.

        Returns
        -------
        InputPort or OutputPort
            Matching port.

        Raises
        ------
        KeyError
            If no port with ``key`` exists.
        """
        # Search the union of self._i and self._o for the port with this name
        for p in self._i + self._o:
            if p.name == key:
                return p
        raise KeyError(f"Port {key} not found in node {self._name}")

    def initialize(self):
        """Initialize node state prior to simulation batches."""
        pass

    def finalize(self, fault_history: Dict[float, Dict[str, bool]] = None):
        """Finalize after simulation completes.

        Parameters
        ----------
        fault_history : dict, optional
            Mapping from simulation time to fault activation status.
        """
        self._fault_history = fault_history or {}
        pass

    def update(self, sim_time: float):
        """Execute one update at the given simulation time.

        Parameters
        ----------
        sim_time : float
            Current simulation time.
        """
        pass

    def depends(self) -> List["Node"]:
        """List node dependencies based on connected input ports.

        Returns
        -------
        list of Node
            Nodes that must execute before this one.
        """
        deps = list()
        for p in self.i:
            # TODO Fail warning if p is not an input port
            # TODO Issue a warning for unconnected ports?
            if p.output_port != None and p.output_port.node not in deps:
                deps.append(p.output_port.node)
        return deps

    @property
    def i(self):
        """Input ports for this Node."""
        return ()

    @property
    def o(self):
        """Output ports for this Node."""
        return ()

    @property
    def p(self):
        """Parameters for this Node."""
        return ()

    @property
    def period(self) -> float:
        """Update period in seconds."""
        return self._period

    @period.setter
    def period(self, value: float):
        self._period = value

    @property
    def frequency(self) -> Union[float, None]:
        """Update frequency in Hz."""
        if self._period is None:
            return None
        else:
            return 1 / self._period

    @frequency.setter
    def frequency(self, value: float):
        self._period = 1 / value

    @property
    def n_inputs(self) -> int:
        """Number of input ports."""
        return len(self._i)

    @property
    def n_outputs(self) -> int:
        """Number of output ports."""
        return len(self._o)

    @property
    def name(self) -> str:
        """Name of the node."""
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
    """Base class for nodes that integrate differential equations.

    Differential nodes are assumed to depend only on inputs from the previous
    simulation step, so they declare no dependencies and help avoid algebraic
    loops in the execution order.
    """

    def __init__(
        self,
        x0: array,
        input_ports: List[str],
        output_ports: List[str],
        parameters: Union[NamedTuple, Tuple] = (),
        **kwargs,
    ):
        """Construct a differential node.

        Parameters
        ----------
        x0 : array
            Initial state for the modeled differential equation.
        input_ports : list-like
            Input ports for the node.
        output_ports : list-like
            Output ports for the node.
        parameters : NamedTuple or tuple, optional
            Parameters that may be faulted.
        **kwargs
            Forwarded to :class:`Node`.
        """

        self._x = x0
        self._x0 = x0
        super().__init__(input_ports, output_ports, parameters, **kwargs)

    def depends(self) -> List[Node]:
        # Differential blocks have no dependencies...
        return list()

    def finalize(self, fault_history: Dict[float, Dict[str, bool]] = None):
        self._x = deepcopy(self._x0)
        return super().finalize(fault_history)
