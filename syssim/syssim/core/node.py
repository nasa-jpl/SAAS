from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Generic, Mapping, TypeVar, cast

from syssim.core.port import InputPort, OutputPort, PortSample, validate_value

StateT = TypeVar("StateT")
ParamT = TypeVar("ParamT")
InputSpecT = TypeVar("InputSpecT")
OutputSpecT = TypeVar("OutputSpecT")
ParameterSpecT = TypeVar("ParameterSpecT")
ConfigSpecT = TypeVar("ConfigSpecT")


@dataclass
class EmptySpec:
    """Empty dataclass specification for nodes without a section."""

    pass


def input_port(
    value_type: Any = Any,
    *,
    name: str | None = None,
    dtype: Any = None,
    shape: tuple[int | None, ...] | None = None,
):
    """Declare an input port field on a node input spec.

    Parameters
    ----------
    value_type : Any, optional
        Runtime type contract for values written to the port.
    name : str, optional
        External port name. Defaults to the dataclass field name.
    dtype : Any, optional
        Required NumPy dtype when strict validation is enabled.
    shape : tuple of int or None, optional
        Required NumPy array shape. ``None`` entries match any size.

    Returns
    -------
    dataclasses.Field
        Dataclass field metadata consumed by ``Node`` construction.
    """
    return field(
        init=False,
        metadata={
            "syssim_kind": "input",
            "name": name,
            "value_type": value_type,
            "dtype": dtype,
            "shape": shape,
        },
    )


def output_port(
    value_type: Any = Any,
    *,
    name: str | None = None,
    dtype: Any = None,
    shape: tuple[int | None, ...] | None = None,
):
    """Declare an output port field on a node output spec.

    Parameters
    ----------
    value_type : Any, optional
        Runtime type contract for values written to the port.
    name : str, optional
        External port name. Defaults to the dataclass field name.
    dtype : Any, optional
        Required NumPy dtype when strict validation is enabled.
    shape : tuple of int or None, optional
        Required NumPy array shape. ``None`` entries match any size.

    Returns
    -------
    dataclasses.Field
        Dataclass field metadata consumed by ``Node`` construction.
    """
    return field(
        init=False,
        metadata={
            "syssim_kind": "output",
            "name": name,
            "value_type": value_type,
            "dtype": dtype,
            "shape": shape,
        },
    )


def parameter(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
    value_type: Any = Any,
    name: str | None = None,
    dtype: Any = None,
    shape: tuple[int | None, ...] | None = None,
):
    """Declare a faultable parameter field on a node parameter spec.

    Parameters
    ----------
    default : Any, optional
        Default nominal value for the parameter.
    default_factory : callable, optional
        Zero-argument callable used to create the nominal value.
    value_type : Any, optional
        Runtime type contract for parameter values.
    name : str, optional
        External parameter name. Defaults to the dataclass field name.
    dtype : Any, optional
        Required NumPy dtype when strict validation is enabled.
    shape : tuple of int or None, optional
        Required NumPy array shape. ``None`` entries match any size.

    Returns
    -------
    dataclasses.Field
        Dataclass field metadata consumed by ``Node`` construction.

    Raises
    ------
    ValueError
        If both ``default`` and ``default_factory`` are provided.
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("parameter cannot define both default and default_factory")
    return field(
        init=False,
        metadata={
            "syssim_kind": "parameter",
            "name": name,
            "default": default,
            "default_factory": default_factory,
            "value_type": value_type,
            "dtype": dtype,
            "shape": shape,
        },
    )


class NodeParameter(Generic[ParamT]):
    """Faultable, typed model coefficient owned by a node.

    Parameters
    ----------
    name : str
        External parameter name.
    value : ParamT
        Initial nominal value.
    node : Node
        Node that owns the parameter.
    attr_name : str, optional
        Attribute name used on the parameter spec dataclass.
    value_type : Any, optional
        Runtime type contract used when strict validation is enabled.
    dtype : Any, optional
        Required NumPy dtype for strict array validation.
    shape : tuple of int or None, optional
        Required NumPy shape for strict array validation.
    strict : bool, optional
        Whether validation is enforced when values are set.
    """

    def __init__(
        self,
        name: str,
        value: ParamT,
        node: "Node[Any, Any, Any, Any]",
        *,
        attr_name: str | None = None,
        value_type: Any = Any,
        dtype: Any = None,
        shape: tuple[int | None, ...] | None = None,
        strict: bool = False,
    ):
        self._name = name
        self._attr_name = attr_name or name
        self._node = node
        self._nominal_value = deepcopy(value)
        self._value = deepcopy(value)
        self._value_type = value_type
        self._dtype = dtype
        self._shape = shape
        self._strict = strict
        self._time = float("nan")

    def add_fault(self, fault):
        """Register this parameter as a mutable target for a fault.

        Parameters
        ----------
        fault : Fault
            Fault object that should be allowed to mutate this parameter.

        Returns
        -------
        Fault
            The same fault, enabling fluent construction.
        """
        fault.add_target(self)
        return fault

    def reset(self) -> None:
        """Restore the parameter to its nominal value."""
        self._value = deepcopy(self._nominal_value)
        self._time = float("nan")

    def set_nominal(self, value: ParamT) -> None:
        """Set both the current and nominal parameter values.

        Parameters
        ----------
        value : ParamT
            New nominal parameter value.
        """
        self.set(value)
        self._nominal_value = deepcopy(value)

    def set_contract(
        self,
        *,
        value_type: Any | None = None,
        dtype: Any = None,
        shape: tuple[int | None, ...] | None = None,
    ) -> None:
        """Update the runtime validation contract for this parameter.

        Parameters
        ----------
        value_type : Any, optional
            Python type or typing annotation accepted for future values.
        dtype : Any, optional
            Required NumPy dtype for future array values.
        shape : tuple of int or None, optional
            Required NumPy array shape. ``None`` entries match any size.
        """
        if value_type is not None:
            self._value_type = value_type
        if dtype is not None:
            self._dtype = dtype
        if shape is not None:
            self._shape = shape

    def set(self, value: ParamT, sim_time: float | None = None, *, strict: bool | None = None) -> None:
        """Set the current parameter value.

        Parameters
        ----------
        value : ParamT
            New parameter value.
        sim_time : float, optional
            Simulation time associated with the value change.
        strict : bool, optional
            Temporary validation override for this write.
        """
        old_strict = self._strict
        if strict is not None:
            self._strict = strict
        try:
            if self._strict:
                validate_value(
                    value,
                    self._value_type,
                    dtype=self._dtype,
                    shape=self._shape,
                    label=self.full_name,
                )
            self._value = deepcopy(value)
            if sim_time is not None:
                self._time = float(sim_time)
        finally:
            self._strict = old_strict

    @property
    def value(self) -> ParamT:
        """Current parameter value.

        Returns
        -------
        ParamT
            Current value stored by the parameter.
        """
        return self._value

    @value.setter
    def value(self, value: ParamT) -> None:
        """Set the current parameter value.

        Parameters
        ----------
        value : ParamT
            New parameter value.
        """
        self.set(value)

    @property
    def sample(self) -> PortSample[ParamT]:
        """Current parameter value and timestamp.

        Returns
        -------
        PortSample
            Sample containing the current value and last write time.
        """
        return PortSample(self._value, self._time)

    @property
    def name(self) -> str:
        """External parameter name.

        Returns
        -------
        str
            Name used in logs and lookup.
        """
        return self._name

    @property
    def attr_name(self) -> str:
        """Parameter spec attribute name.

        Returns
        -------
        str
            Dataclass attribute name for this parameter.
        """
        return self._attr_name

    @property
    def full_name(self) -> str:
        """Fully qualified parameter name.

        Returns
        -------
        str
            Name formatted as ``node.parameter``.
        """
        node_name = self._node.name or self._node.__class__.__name__
        return f"{node_name}.{self._attr_name}"

    @property
    def node(self) -> "Node[Any, Any, Any, Any]":
        """Node that owns this parameter.

        Returns
        -------
        Node
            Owning node.
        """
        return self._node

    @property
    def strict(self) -> bool:
        """Whether strict runtime validation is enabled.

        Returns
        -------
        bool
            ``True`` when writes enforce the parameter contract.
        """
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        """Set strict runtime validation.

        Parameters
        ----------
        value : bool
            Whether future writes should enforce the parameter contract.
        """
        self._strict = bool(value)


class Node(ABC, Generic[InputSpecT, OutputSpecT, ParameterSpecT, ConfigSpecT]):
    """Base class for dataclass-specified syssim nodes.

    Subclasses declare ``Inputs``, ``Outputs``, ``Parameters``, and ``Config``
    dataclasses. ``Node`` binds those declarations to ``i``, ``o``, ``p``, and
    ``config`` instances at construction time.

    Parameters
    ----------
    config : object, optional
        Dataclass instance matching the node's ``Config`` type.
    sample_frequency : float, optional
        Update frequency in hertz. Mutually exclusive in practice with
        ``sample_period``.
    sample_period : float, optional
        Update period in seconds.
    name : str, optional
        Node name used in systems, logs, and fully qualified port names.
    """

    Inputs: ClassVar[type[Any]] = EmptySpec
    Outputs: ClassVar[type[Any]] = EmptySpec
    Parameters: ClassVar[type[Any]] = EmptySpec
    Config: ClassVar[type[Any]] = EmptySpec

    i: InputSpecT
    o: OutputSpecT
    p: ParameterSpecT
    config: ConfigSpecT
    _config: ConfigSpecT

    def __init__(
        self,
        *,
        config: object | None = None,
        sample_frequency: float | None = None,
        sample_period: float | None = None,
        name: str | None = None,
    ):
        self._name = name
        self._period = 1.0 / sample_frequency if sample_frequency is not None else sample_period
        self._system = None
        self.i = cast(InputSpecT, self._build_ports(self.Inputs, InputPort, "input"))
        self.o = cast(OutputSpecT, self._build_ports(self.Outputs, OutputPort, "output"))
        self.p = cast(ParameterSpecT, self._build_parameters(self.Parameters))
        self.config = cast(ConfigSpecT, self._build_config(self.Config, config))
        self._config = self.config

    def __getitem__(self, key: str):
        """Return a port or parameter by name.

        Parameters
        ----------
        key : str
            External name or spec attribute name.

        Returns
        -------
        InputPort, OutputPort, or NodeParameter
            Matching item owned by this node.

        Raises
        ------
        KeyError
            If no item with ``key`` exists on this node.
        """
        for item in (*self.iter_ports(), *self.iter_parameters()):
            if item.name == key or item.attr_name == key:
                return item
        raise KeyError(f"{key!r} not found in node {self.name!r}")

    def initialize(self) -> None:
        """Prepare the node before a simulation run starts."""
        pass

    def finalize(self, fault_history: dict[float, dict[str, bool]] | None = None) -> None:
        """Clean up after a simulation run.

        Parameters
        ----------
        fault_history : dict, optional
            Mapping from simulation time to fault active states.
        """
        self._fault_history = fault_history or {}

    @abstractmethod
    def update(self, sim_time: float) -> None:
        """Advance node behavior at a simulation time.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds.
        """
        pass

    def depends(self) -> list["Node[Any, Any, Any, Any]"]:
        """Return nodes whose outputs feed this node's inputs.

        Returns
        -------
        list of Node
            Upstream nodes used by the scheduler for topological ordering.
        """
        deps = []
        for input_item in self.iter_input_ports():
            if input_item.source is not None and input_item.source.node not in deps:
                deps.append(input_item.source.node)
        return deps

    def run_step(self, sim_time: float, inputs: Mapping[str, Any] | None = None) -> dict[str, PortSample]:
        """Run one standalone update with optional input overrides.

        Parameters
        ----------
        sim_time : float
            Simulation time passed to ``update``.
        inputs : Mapping[str, Any], optional
            Values, ``PortSample`` objects, or callables keyed by input name.

        Returns
        -------
        dict
            Output samples keyed by output attribute name.
        """
        for name, input_value in (inputs or {}).items():
            port = getattr(self.i, name)
            value = input_value(sim_time) if callable(input_value) else input_value
            if isinstance(value, PortSample):
                port._write_sample(value)
            else:
                port.write(value, sim_time)
        self.update(sim_time)
        return {port.attr_name: port.read() for port in self.iter_output_ports()}

    def iter_input_ports(self):
        """Iterate over input ports.

        Returns
        -------
        tuple
            Input ports declared by the node's input spec.
        """
        return _iter_spec_values(self.i)

    def iter_output_ports(self):
        """Iterate over output ports.

        Returns
        -------
        tuple
            Output ports declared by the node's output spec.
        """
        return _iter_spec_values(self.o)

    def iter_ports(self):
        """Iterate over all input and output ports.

        Yields
        ------
        InputPort or OutputPort
            Ports declared by the node's input and output specs.
        """
        yield from self.iter_input_ports()
        yield from self.iter_output_ports()

    def iter_parameters(self):
        """Iterate over parameters.

        Returns
        -------
        tuple
            Parameters declared by the node's parameter spec.
        """
        return _iter_spec_values(self.p)

    def set_strict_types(self, strict: bool) -> None:
        """Enable or disable strict runtime validation on all ports and parameters.

        Parameters
        ----------
        strict : bool
            Whether future writes should enforce declared type contracts.
        """
        for item in (*self.iter_ports(), *self.iter_parameters()):
            item.strict = strict

    @property
    def period(self) -> float | None:
        """Node update period in seconds.

        Returns
        -------
        float or None
            Configured update period, or ``None`` to inherit system ``dt``.
        """
        return self._period

    @period.setter
    def period(self, value: float | None) -> None:
        """Set the node update period.

        Parameters
        ----------
        value : float or None
            Update period in seconds, or ``None`` to inherit system ``dt``.
        """
        self._period = None if value is None else float(value)

    @property
    def frequency(self) -> float | None:
        """Node update frequency in hertz.

        Returns
        -------
        float or None
            Reciprocal of ``period``, or ``None`` when period is unset.
        """
        return None if self._period is None else 1.0 / self._period

    @frequency.setter
    def frequency(self, value: float) -> None:
        """Set the node update frequency.

        Parameters
        ----------
        value : float
            Update frequency in hertz.
        """
        self._period = 1.0 / float(value)

    @property
    def n_inputs(self) -> int:
        """Number of input ports.

        Returns
        -------
        int
            Count of input ports declared by the node.
        """
        return len(tuple(self.iter_input_ports()))

    @property
    def n_outputs(self) -> int:
        """Number of output ports.

        Returns
        -------
        int
            Count of output ports declared by the node.
        """
        return len(tuple(self.iter_output_ports()))

    @property
    def name(self) -> str | None:
        """Node name.

        Returns
        -------
        str or None
            Name used in systems, logs, and fully qualified item names.
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        """Set the node name.

        Parameters
        ----------
        name : str or None
            New node name.
        """
        self._name = name

    def _build_ports(self, spec_cls, port_cls, kind: str):
        spec = _new_spec(spec_cls, kind)
        for spec_field in fields(spec):
            meta = spec_field.metadata
            if meta.get("syssim_kind", kind) != kind:
                raise TypeError(f"{type(self).__name__}.{kind} spec field {spec_field.name!r} has wrong kind")
            port_name = meta.get("name") or spec_field.name
            setattr(
                spec,
                spec_field.name,
                port_cls(
                    port_name,
                    self,
                    attr_name=spec_field.name,
                    value_type=meta.get("value_type", Any),
                    dtype=meta.get("dtype"),
                    shape=meta.get("shape"),
                ),
            )
        return spec

    def _build_parameters(self, spec_cls):
        spec = _new_spec(spec_cls, "parameter")
        for spec_field in fields(spec):
            meta = spec_field.metadata
            if meta.get("syssim_kind", "parameter") != "parameter":
                raise TypeError(f"{type(self).__name__}.Parameters field {spec_field.name!r} has wrong kind")
            default_factory = meta.get("default_factory", MISSING)
            if default_factory is not MISSING:
                default_value = default_factory()
            else:
                default_value = meta.get("default", None)
                if default_value is MISSING:
                    default_value = None
            parameter_name = meta.get("name") or spec_field.name
            setattr(
                spec,
                spec_field.name,
                NodeParameter(
                    parameter_name,
                    default_value,
                    self,
                    attr_name=spec_field.name,
                    value_type=meta.get("value_type", Any),
                    dtype=meta.get("dtype"),
                    shape=meta.get("shape"),
                ),
            )
        return spec

    def _build_config(self, spec_cls, config: object | None):
        spec_cls = _ensure_dataclass(spec_cls, "Config")
        if config is None:
            return spec_cls()
        if isinstance(config, spec_cls):
            return config
        raise TypeError(
            f"{type(self).__name__} config must be a {spec_cls.__name__} dataclass instance or None"
        )


class NodeDifferential(
    Node[InputSpecT, OutputSpecT, ParameterSpecT, ConfigSpecT],
    Generic[StateT, InputSpecT, OutputSpecT, ParameterSpecT, ConfigSpecT],
):
    """Node base for systems whose output comes from integrated state.

    Differential nodes are treated as having no same-step dependencies, which
    allows closed-loop graphs to be compiled when the state integration breaks
    the algebraic cycle.

    Parameters
    ----------
    initial_state : StateT
        Initial state copied into ``state`` before each run.
    **kwargs
        Arguments forwarded to ``Node``.
    """

    def __init__(self, initial_state: StateT, **kwargs):
        self.initial_state = deepcopy(initial_state)
        self.state = deepcopy(initial_state)
        super().__init__(**kwargs)

    def reset_state(self) -> None:
        """Reset ``state`` to ``initial_state``."""
        self.state = deepcopy(self.initial_state)

    def depends(self) -> list[Node[Any, Any, Any, Any]]:
        """Return no same-step dependencies for differential nodes.

        Returns
        -------
        list
            Always empty.
        """
        return []

    def finalize(self, fault_history: dict[float, dict[str, bool]] | None = None) -> None:
        """Reset state and finalize the node.

        Parameters
        ----------
        fault_history : dict, optional
            Mapping from simulation time to fault active states.
        """
        self.reset_state()
        super().finalize(fault_history)


def _ensure_dataclass(spec_cls, label: str):
    if spec_cls is None:
        return EmptySpec
    if not is_dataclass(spec_cls):
        raise TypeError(f"{label} must be a dataclass type")
    return spec_cls


def _new_spec(spec_cls, label: str):
    return _ensure_dataclass(spec_cls, label)()


def _iter_spec_values(spec):
    return tuple(getattr(spec, item.name) for item in fields(spec))
