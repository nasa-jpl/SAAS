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
    pass


def input_port(
    value_type: Any = Any,
    *,
    name: str | None = None,
    dtype: Any = None,
    shape: tuple[int | None, ...] | None = None,
):
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
    """Faultable, typed model coefficient owned by a node."""

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
        fault.add_target(self)
        return fault

    def reset(self) -> None:
        self._value = deepcopy(self._nominal_value)
        self._time = float("nan")

    def set_nominal(self, value: ParamT) -> None:
        self.set(value)
        self._nominal_value = deepcopy(value)

    def set_contract(
        self,
        *,
        value_type: Any | None = None,
        dtype: Any = None,
        shape: tuple[int | None, ...] | None = None,
    ) -> None:
        if value_type is not None:
            self._value_type = value_type
        if dtype is not None:
            self._dtype = dtype
        if shape is not None:
            self._shape = shape

    def set(self, value: ParamT, sim_time: float | None = None, *, strict: bool | None = None) -> None:
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
        return self._value

    @value.setter
    def value(self, value: ParamT) -> None:
        self.set(value)

    @property
    def sample(self) -> PortSample[ParamT]:
        return PortSample(self._value, self._time)

    @property
    def name(self) -> str:
        return self._name

    @property
    def attr_name(self) -> str:
        return self._attr_name

    @property
    def full_name(self) -> str:
        node_name = self._node.name or self._node.__class__.__name__
        return f"{node_name}.{self._attr_name}"

    @property
    def node(self) -> "Node[Any, Any, Any, Any]":
        return self._node

    @property
    def strict(self) -> bool:
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        self._strict = bool(value)


class Node(ABC, Generic[InputSpecT, OutputSpecT, ParameterSpecT, ConfigSpecT]):
    """Base class for dataclass-specified syssim nodes."""

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
        for item in (*self.iter_ports(), *self.iter_parameters()):
            if item.name == key or item.attr_name == key:
                return item
        raise KeyError(f"{key!r} not found in node {self.name!r}")

    def initialize(self) -> None:
        pass

    def finalize(self, fault_history: dict[float, dict[str, bool]] | None = None) -> None:
        self._fault_history = fault_history or {}

    @abstractmethod
    def update(self, sim_time: float) -> None:
        pass

    def depends(self) -> list["Node[Any, Any, Any, Any]"]:
        deps = []
        for input_item in self.iter_input_ports():
            if input_item.source is not None and input_item.source.node not in deps:
                deps.append(input_item.source.node)
        return deps

    def run_step(self, sim_time: float, inputs: Mapping[str, Any] | None = None) -> dict[str, PortSample]:
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
        return _iter_spec_values(self.i)

    def iter_output_ports(self):
        return _iter_spec_values(self.o)

    def iter_ports(self):
        yield from self.iter_input_ports()
        yield from self.iter_output_ports()

    def iter_parameters(self):
        return _iter_spec_values(self.p)

    def set_strict_types(self, strict: bool) -> None:
        for item in (*self.iter_ports(), *self.iter_parameters()):
            item.strict = strict

    @property
    def period(self) -> float | None:
        return self._period

    @period.setter
    def period(self, value: float | None) -> None:
        self._period = None if value is None else float(value)

    @property
    def frequency(self) -> float | None:
        return None if self._period is None else 1.0 / self._period

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._period = 1.0 / float(value)

    @property
    def n_inputs(self) -> int:
        return len(tuple(self.iter_input_ports()))

    @property
    def n_outputs(self) -> int:
        return len(tuple(self.iter_output_ports()))

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
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
    """Node base for systems whose output comes from integrated state."""

    def __init__(self, initial_state: StateT, **kwargs):
        self.initial_state = deepcopy(initial_state)
        self.state = deepcopy(initial_state)
        super().__init__(**kwargs)

    def reset_state(self) -> None:
        self.state = deepcopy(self.initial_state)

    def depends(self) -> list[Node[Any, Any, Any, Any]]:
        return []

    def finalize(self, fault_history: dict[float, dict[str, bool]] | None = None) -> None:
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
