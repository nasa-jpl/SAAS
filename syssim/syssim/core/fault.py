from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from syssim.core.port import InputPort, OutputPort, PortSample


class FaultContext:
    """Read/write context passed into a fault at a simulation instant."""

    def __init__(self, system: "NodeSystem", fault: "Fault", time: float, targets: Iterable[Any]):
        self.system = system
        self.fault = fault
        self.time = float(time)
        self._targets = tuple(targets)

    @property
    def targets(self) -> tuple[Any, ...]:
        return self._targets

    @property
    def samples(self) -> dict[str, PortSample]:
        return self.system.samples()

    @property
    def parameters(self) -> dict[str, Any]:
        return self.system.parameter_values()

    def read(self, target: Any):
        if isinstance(target, (InputPort, OutputPort)):
            return target.read()
        if hasattr(target, "sample"):
            return target.sample
        raise TypeError(f"Cannot read unsupported fault target {target!r}")

    def write(self, target: Any, value: Any, *, sample_time: float | None = None) -> None:
        if target not in self._targets:
            raise PermissionError("Faults may only write to their registered targets")
        write_time = self.time if sample_time is None else sample_time
        sample = PortSample(value, write_time)
        if isinstance(target, OutputPort):
            target._write_sample(sample, propagate=True, strict=self.system.strict_types)
        elif isinstance(target, InputPort):
            target._write_sample(sample, strict=self.system.strict_types)
        elif hasattr(target, "set"):
            target.set(value, write_time, strict=self.system.strict_types)
        else:
            raise TypeError(f"Cannot write unsupported fault target {target!r}")


class Fault(ABC):
    """Base class for faults that mutate registered ports or parameters."""

    def __init__(self, name: str | None = None, targets: Iterable[Any] | None = None, *, enabled: bool = True):
        self._name = name
        self._targets: list[Any] = []
        self.enabled = bool(enabled)
        self._active = False
        self._triggered = False
        for target in targets or ():
            self.add_target(target)

    def initialize(self) -> None:
        self._active = False
        self._triggered = False

    def add_target(self, target: Any) -> "Fault":
        if target not in self._targets:
            self._targets.append(target)
        return self

    def evaluate(self, context: FaultContext) -> bool:
        self._active = self.enabled and bool(self.trigger(context))
        if self._active:
            self._triggered = True
        return self._active

    @abstractmethod
    def trigger(self, context: FaultContext) -> bool:
        pass

    @abstractmethod
    def mutate(self, context: FaultContext) -> None:
        pass

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        self._name = value

    @property
    def targets(self) -> tuple[Any, ...]:
        return tuple(self._targets)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def triggered(self) -> bool:
        return self._triggered