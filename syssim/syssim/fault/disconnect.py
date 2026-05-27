from __future__ import annotations

from typing import Iterable

from syssim.core.fault import Fault, FaultContext
from syssim.fault.utility import nan_like, zero_like


class TimeTriggeredFault(Fault):
    """Fault base that becomes active after a trigger time.

    Parameters
    ----------
    name : str, optional
        Fault name used in logs and history.
    trigger_time : float, optional
        Simulation time at which the fault becomes active.
    targets : iterable, optional
        Ports or parameters that this fault may mutate.
    enabled : bool, optional
        Whether the fault is eligible to trigger.
    """

    def __init__(self, name: str | None = None, trigger_time: float = 0.0, targets: Iterable[object] | None = None, *, enabled: bool = True):
        super().__init__(name=name, targets=targets, enabled=enabled)
        self.trigger_time = float(trigger_time)

    def trigger(self, context: FaultContext) -> bool:
        """Return whether the context time is at or after ``trigger_time``.

        Parameters
        ----------
        context : FaultContext
            Context for the current simulation instant.

        Returns
        -------
        bool
            ``True`` when the fault should be active.
        """
        return context.time >= self.trigger_time


class DisconnectFault(TimeTriggeredFault):
    """Fault that replaces target values with NaNs after ``trigger_time``.

    Parameters
    ----------
    name : str, optional
        Fault name used in logs and history.
    trigger_time : float, optional
        Simulation time at which the fault becomes active.
    targets : iterable, optional
        Ports or parameters that this fault may mutate.
    enabled : bool, optional
        Whether the fault is eligible to trigger.
    """

    def mutate(self, context: FaultContext) -> None:
        """Write NaN-like values to each active target.

        Parameters
        ----------
        context : FaultContext
            Context containing writable targets for the current phase.
        """
        for target in context.targets:
            sample = context.read(target)
            context.write(target, nan_like(sample.value), sample_time=sample.time)


class ZeroFault(TimeTriggeredFault):
    """Fault that replaces target values with zeros after ``trigger_time``.

    Parameters
    ----------
    name : str, optional
        Fault name used in logs and history.
    trigger_time : float, optional
        Simulation time at which the fault becomes active.
    targets : iterable, optional
        Ports or parameters that this fault may mutate.
    enabled : bool, optional
        Whether the fault is eligible to trigger.
    """

    def mutate(self, context: FaultContext) -> None:
        """Write zero-like values to each active target.

        Parameters
        ----------
        context : FaultContext
            Context containing writable targets for the current phase.
        """
        for target in context.targets:
            sample = context.read(target)
            context.write(target, zero_like(sample.value), sample_time=sample.time)