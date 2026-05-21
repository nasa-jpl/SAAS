from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import UnionType
from typing import Any, Generic, TypeVar, get_args, get_origin

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class PortSample(Generic[T]):
    """A value carried by a port and the simulation time that produced it."""

    value: T
    time: float

    def __iter__(self):
        yield self.value
        yield self.time


def validate_value(
    value: Any,
    expected_type: Any = Any,
    *,
    dtype: Any = None,
    shape: tuple[int | None, ...] | None = None,
    label: str = "value",
) -> None:
    """Validate a runtime value against a compact syssim type contract."""
    if value is None or expected_type in (Any, object, None):
        return

    if not _matches_type(value, expected_type):
        raise TypeError(f"{label} must be {expected_type!r}; got {type(value)!r}")

    if dtype is not None or shape is not None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{label} must be a numpy.ndarray for dtype/shape checks")
        if dtype is not None and np.dtype(value.dtype) != np.dtype(dtype):
            raise TypeError(f"{label} dtype must be {np.dtype(dtype)}; got {value.dtype}")
        if shape is not None:
            if len(value.shape) != len(shape):
                raise TypeError(f"{label} shape must be {shape}; got {value.shape}")
            for actual, expected in zip(value.shape, shape):
                if expected is not None and actual != expected:
                    raise TypeError(f"{label} shape must be {shape}; got {value.shape}")


def _matches_type(value: Any, expected_type: Any) -> bool:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin in (UnionType, getattr(__import__("typing"), "Union")):
        return any(_matches_type(value, arg) for arg in args)
    if origin is not None:
        if origin is tuple and args and args[-1] is Ellipsis:
            return isinstance(value, tuple) and all(_matches_type(item, args[0]) for item in value)
        return isinstance(value, origin)
    if expected_type is type(None):
        return value is None
    if isinstance(expected_type, type):
        return isinstance(value, expected_type)
    return True


class _Port(Generic[T]):
    def __init__(
        self,
        name: str,
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
        self._value_type = value_type
        self._dtype = dtype
        self._shape = shape
        self._strict = strict
        self._sample: PortSample[T | None] = PortSample(None, float("nan"))

    def read(self) -> PortSample[T | None]:
        """Return the current value with its production timestamp."""
        return self._sample

    def read_with_time(self):
        """Return ``(value, time)`` for compatibility with older examples."""
        return self._sample.value, self._sample.time

    def add_fault(self, fault):
        """Register this port as a mutable target for a fault."""
        fault.add_target(self)
        return fault

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

    def _validate(self, value: Any) -> None:
        if self._strict:
            validate_value(
                value,
                self._value_type,
                dtype=self._dtype,
                shape=self._shape,
                label=self.full_name,
            )

    def _set_sample(self, sample: PortSample[T], *, strict: bool | None = None) -> None:
        old_strict = self._strict
        if strict is not None:
            self._strict = strict
        try:
            self._validate(sample.value)
            self._sample = PortSample(deepcopy(sample.value), float(sample.time))
        finally:
            self._strict = old_strict

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


class InputPort(_Port[T]):
    """Input boundary for a node."""

    def __init__(self, name: str, node: "Node", **kwargs):
        super().__init__(name, node, **kwargs)
        self._source: OutputPort[T] | None = None

    @property
    def output_port(self) -> OutputPort[T] | None:
        return self._source

    @property
    def source(self) -> OutputPort[T] | None:
        return self._source

    def _connect(self, output_port: "OutputPort[T]") -> None:
        if self._source is not None and self._source is not output_port:
            raise ValueError(f"Input port {self.full_name} is already connected")
        self._source = output_port

    def write(self, value: T, sim_time: float) -> None:
        self._set_sample(PortSample(value, sim_time))

    def _write_sample(self, sample: PortSample[T], *, strict: bool | None = None) -> None:
        self._set_sample(sample, strict=strict)

    def __lshift__(self, output_port: "OutputPort[T]"):
        output_port.connect(self)
        return self


class OutputPort(_Port[T]):
    """Output boundary for a node; one output can fan out to many inputs."""

    def __init__(self, name: str, node: "Node", **kwargs):
        super().__init__(name, node, **kwargs)
        self._inputs: list[InputPort[T]] = []

    @property
    def input_ports(self) -> tuple[InputPort[T], ...]:
        return tuple(self._inputs)

    def connect(self, input_port: InputPort[T]) -> None:
        if not isinstance(input_port, InputPort):
            raise TypeError("OutputPort can only connect to InputPort")
        input_port._connect(self)
        if input_port not in self._inputs:
            self._inputs.append(input_port)

    def connect_input(self, input_port: InputPort[T]) -> None:
        self.connect(input_port)

    def write(self, value: T, sim_time: float) -> None:
        sample = PortSample(value, sim_time)
        self._set_sample(sample)
        for input_port in self._inputs:
            input_port._write_sample(sample)

    def shift_out(self, value: T, sim_time: float) -> None:
        self.write(value, sim_time)

    def _write_sample(
        self,
        sample: PortSample[T],
        *,
        propagate: bool = True,
        strict: bool | None = None,
    ) -> None:
        self._set_sample(sample, strict=strict)
        if propagate:
            for input_port in self._inputs:
                input_port._write_sample(sample, strict=strict)

    def __rshift__(self, input_port: InputPort[T]):
        self.connect(input_port)
        return input_port

