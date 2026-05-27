from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from syssim.core.port import InputPort, OutputPort, PortSample


class FaultContext:
    """Read/write context passed into a fault at a simulation instant.

    Parameters
    ----------
    system : NodeSystem
        System currently evaluating the fault.
    fault : Fault
        Fault that owns this context.
    time : float
        Current simulation time.
    targets : iterable
        Mutable targets available to this fault evaluation.
    """

    def __init__(self, system: "NodeSystem", fault: "Fault", time: float, targets: Iterable[Any]):
        self.system = system
        self.fault = fault
        self.time = float(time)
        self._targets = tuple(targets)

    @property
    def targets(self) -> tuple[Any, ...]:
        """Targets this context permits the fault to mutate.

        Returns
        -------
        tuple
            Registered targets selected for the current phase.
        """
        return self._targets

    @property
    def samples(self) -> dict[str, PortSample]:
        """Current system port samples.

        Returns
        -------
        dict
            Port samples keyed by fully qualified port name.
        """
        return self.system.samples()

    @property
    def parameters(self) -> dict[str, Any]:
        """Current system parameter values.

        Returns
        -------
        dict
            Parameter values keyed by fully qualified parameter name.
        """
        return self.system.parameter_values()

    def read(self, target: Any):
        """Read a port or parameter target as a sample.

        Parameters
        ----------
        target : Any
            Target port or parameter-like object with a ``sample`` attribute.

        Returns
        -------
        PortSample
            Current target sample.

        Raises
        ------
        TypeError
            If ``target`` is not readable by faults.
        """
        if isinstance(target, (InputPort, OutputPort)):
            return target.read()
        if hasattr(target, "sample"):
            return target.sample
        raise TypeError(f"Cannot read unsupported fault target {target!r}")

    def write(self, target: Any, value: Any, *, sample_time: float | None = None) -> None:
        """Write a value to a permitted target.

        Parameters
        ----------
        target : Any
            Registered target to mutate.
        value : Any
            Replacement value.
        sample_time : float, optional
            Timestamp to store with the value. Defaults to context time.

        Raises
        ------
        PermissionError
            If ``target`` is not available in this context.
        TypeError
            If ``target`` cannot be written by faults.
        """
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
    """Base class for faults that mutate registered ports or parameters.

    Parameters
    ----------
    name : str, optional
        Fault name used in logs and history.
    targets : iterable, optional
        Ports or parameters the fault may mutate.
    enabled : bool, optional
        Whether the fault is eligible to trigger.
    """

    def __init__(self, name: str | None = None, targets: Iterable[Any] | None = None, *, enabled: bool = True):
        self._name = name
        self._targets: list[Any] = []
        self.enabled = bool(enabled)
        self._active = False
        self._triggered = False
        for target in targets or ():
            self.add_target(target)

    def initialize(self) -> None:
        """Reset active and triggered state before a simulation run."""
        self._active = False
        self._triggered = False

    def add_target(self, target: Any) -> "Fault":
        """Register a mutable target for this fault.

        Parameters
        ----------
        target : Any
            Port or parameter-like object to mutate when active.

        Returns
        -------
        Fault
            This fault, enabling fluent construction.
        """
        if target not in self._targets:
            self._targets.append(target)
        return self

    def evaluate(self, context: FaultContext) -> bool:
        """Evaluate the trigger condition and update active state.

        Parameters
        ----------
        context : FaultContext
            Context for the current simulation instant.

        Returns
        -------
        bool
            ``True`` when the fault is enabled and triggered.
        """
        self._active = self.enabled and bool(self.trigger(context))
        if self._active:
            self._triggered = True
        return self._active

    @abstractmethod
    def trigger(self, context: FaultContext) -> bool:
        """Return whether the fault should be active.

        Parameters
        ----------
        context : FaultContext
            Context for the current simulation instant.

        Returns
        -------
        bool
            ``True`` when the fault should mutate its targets.
        """
        pass

    @abstractmethod
    def mutate(self, context: FaultContext) -> None:
        """Mutate registered targets while the fault is active.

        Parameters
        ----------
        context : FaultContext
            Context containing writable targets for the current phase.
        """
        pass

    @property
    def name(self) -> str | None:
        """Fault name.

        Returns
        -------
        str or None
            Name used in logs and history.
        """
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the fault name.

        Parameters
        ----------
        value : str or None
            New fault name.
        """
        self._name = value

    @property
    def targets(self) -> tuple[Any, ...]:
        """Registered mutation targets.

        Returns
        -------
        tuple
            Ports or parameters this fault may mutate.
        """
        return tuple(self._targets)

    @property
    def active(self) -> bool:
        """Whether the fault is active at the current simulation time.

        Returns
        -------
        bool
            Current active state.
        """
        return self._active

    @property
    def triggered(self) -> bool:
        """Whether the fault has ever become active in the current run.

        Returns
        -------
        bool
            ``True`` after the first active evaluation.
        """
        return self._triggered