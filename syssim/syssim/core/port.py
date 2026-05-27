from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import UnionType
from typing import Any, Generic, TypeVar, get_args, get_origin

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class PortSample(Generic[T]):
    """Value carried by a port and the time that produced it.

    Parameters
    ----------
    value : T
        Payload stored on the port.
    time : float
        Simulation time associated with ``value``.
    """

    value: T
    time: float

    def __iter__(self):
        """Iterate over ``(value, time)``.

        Yields
        ------
        T
            Stored payload.
        float
            Simulation time associated with the payload.
        """
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
    """Validate a runtime value against a compact syssim type contract.

    Parameters
    ----------
    value : Any
        Runtime value to validate.
    expected_type : Any, optional
        Python type or typing annotation accepted for ``value``.
    dtype : Any, optional
        Required NumPy dtype when ``value`` is an array.
    shape : tuple of int or None, optional
        Required NumPy array shape. ``None`` entries match any size.
    label : str, optional
        Human-readable name used in validation errors.

    Raises
    ------
    TypeError
        If ``value`` does not satisfy the requested contract.
    """
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
    """Base storage and validation behavior shared by ports.

    Parameters
    ----------
    name : str
        External port name.
    node : Node
        Node that owns the port.
    attr_name : str, optional
        Attribute name used on the node spec dataclass.
    value_type : Any, optional
        Runtime type contract used when strict validation is enabled.
    dtype : Any, optional
        Required NumPy dtype for strict array validation.
    shape : tuple of int or None, optional
        Required NumPy shape for strict array validation.
    strict : bool, optional
        Whether validation is enforced when values are written.
    """

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
        """Return the current value with its production timestamp.

        Returns
        -------
        PortSample
            Current port sample. Unwritten ports contain ``None`` and ``nan``.
        """
        return self._sample

    def read_with_time(self):
        """Return the current sample as a tuple.

        Returns
        -------
        tuple
            Pair ``(value, time)`` for compatibility with older examples.
        """
        return self._sample.value, self._sample.time

    def add_fault(self, fault):
        """Register this port as a mutable target for a fault.

        Parameters
        ----------
        fault : Fault
            Fault object that should be allowed to mutate this port.

        Returns
        -------
        Fault
            The same fault, enabling fluent construction.
        """
        fault.add_target(self)
        return fault

    def set_contract(
        self,
        *,
        value_type: Any | None = None,
        dtype: Any = None,
        shape: tuple[int | None, ...] | None = None,
    ) -> None:
        """Update the runtime validation contract for this port.

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
        """External port name.

        Returns
        -------
        str
            Name used in display and lookup.
        """
        return self._name

    @property
    def attr_name(self) -> str:
        """Port spec attribute name.

        Returns
        -------
        str
            Dataclass attribute name for this port.
        """
        return self._attr_name

    @property
    def full_name(self) -> str:
        """Fully qualified port name.

        Returns
        -------
        str
            Name formatted as ``node.port``.
        """
        node_name = self._node.name or self._node.__class__.__name__
        return f"{node_name}.{self._attr_name}"

    @property
    def node(self) -> "Node[Any, Any, Any, Any]":
        """Node that owns this port.

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
            ``True`` when writes enforce the port contract.
        """
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        """Set strict runtime validation.

        Parameters
        ----------
        value : bool
            Whether future writes should enforce the port contract.
        """
        self._strict = bool(value)


class InputPort(_Port[T]):
    """Input boundary for a node.

    Parameters
    ----------
    name : str
        External port name.
    node : Node
        Node that owns this input.
    **kwargs
        Additional validation metadata forwarded to ``_Port``.
    """

    def __init__(self, name: str, node: "Node", **kwargs):
        super().__init__(name, node, **kwargs)
        self._source: OutputPort[T] | None = None

    @property
    def output_port(self) -> OutputPort[T] | None:
        """Output port currently connected to this input.

        Returns
        -------
        OutputPort or None
            Upstream source, or ``None`` if the input is unconnected.
        """
        return self._source

    @property
    def source(self) -> OutputPort[T] | None:
        """Alias for ``output_port``.

        Returns
        -------
        OutputPort or None
            Upstream source, or ``None`` if the input is unconnected.
        """
        return self._source

    def _connect(self, output_port: "OutputPort[T]") -> None:
        if self._source is not None and self._source is not output_port:
            raise ValueError(f"Input port {self.full_name} is already connected")
        self._source = output_port

    def write(self, value: T, sim_time: float) -> None:
        """Write a value directly to the input port.

        Parameters
        ----------
        value : T
            Payload to store.
        sim_time : float
            Simulation time associated with ``value``.
        """
        self._set_sample(PortSample(value, sim_time))

    def _write_sample(self, sample: PortSample[T], *, strict: bool | None = None) -> None:
        self._set_sample(sample, strict=strict)

    def __lshift__(self, output_port: "OutputPort[T]"):
        """Connect an output using ``input_port << output_port``.

        Parameters
        ----------
        output_port : OutputPort
            Upstream output to connect.

        Returns
        -------
        InputPort
            This input port.
        """
        output_port.connect(self)
        return self


class OutputPort(_Port[T]):
    """Output boundary for a node.

    One output can fan out to many inputs.

    Parameters
    ----------
    name : str
        External port name.
    node : Node
        Node that owns this output.
    **kwargs
        Additional validation metadata forwarded to ``_Port``.
    """

    def __init__(self, name: str, node: "Node", **kwargs):
        super().__init__(name, node, **kwargs)
        self._inputs: list[InputPort[T]] = []

    @property
    def input_ports(self) -> tuple[InputPort[T], ...]:
        """Inputs currently connected to this output.

        Returns
        -------
        tuple of InputPort
            Downstream input ports receiving propagated samples.
        """
        return tuple(self._inputs)

    def connect(self, input_port: InputPort[T]) -> None:
        """Connect this output to an input port.

        Parameters
        ----------
        input_port : InputPort
            Downstream input to receive samples from this output.

        Raises
        ------
        TypeError
            If ``input_port`` is not an ``InputPort``.
        ValueError
            If the input is already connected to a different output.
        """
        if not isinstance(input_port, InputPort):
            raise TypeError("OutputPort can only connect to InputPort")
        input_port._connect(self)
        if input_port not in self._inputs:
            self._inputs.append(input_port)

    def connect_input(self, input_port: InputPort[T]) -> None:
        """Compatibility wrapper for ``connect``.

        Parameters
        ----------
        input_port : InputPort
            Downstream input to receive samples from this output.
        """
        self.connect(input_port)

    def write(self, value: T, sim_time: float) -> None:
        """Write a value and propagate it to connected inputs.

        Parameters
        ----------
        value : T
            Payload to store and propagate.
        sim_time : float
            Simulation time associated with ``value``.
        """
        sample = PortSample(value, sim_time)
        self._set_sample(sample)
        for input_port in self._inputs:
            input_port._write_sample(sample)

    def shift_out(self, value: T, sim_time: float) -> None:
        """Compatibility alias for ``write``.

        Parameters
        ----------
        value : T
            Payload to store and propagate.
        sim_time : float
            Simulation time associated with ``value``.
        """
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
        """Connect an input using ``output_port >> input_port``.

        Parameters
        ----------
        input_port : InputPort
            Downstream input to connect.

        Returns
        -------
        InputPort
            Connected input port.
        """
        self.connect(input_port)
        return input_port

