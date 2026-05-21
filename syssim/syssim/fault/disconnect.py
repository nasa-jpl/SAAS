from __future__ import annotations

from typing import Iterable

from syssim.core.fault import Fault, FaultContext
from syssim.fault.utility import nan_like, zero_like


class TimeTriggeredFault(Fault):
    def __init__(self, name: str | None = None, trigger_time: float = 0.0, targets: Iterable[object] | None = None, *, enabled: bool = True):
        super().__init__(name=name, targets=targets, enabled=enabled)
        self.trigger_time = float(trigger_time)

    def trigger(self, context: FaultContext) -> bool:
        return context.time >= self.trigger_time


class DisconnectFault(TimeTriggeredFault):
    """Fault that replaces target values with NaNs after ``trigger_time``."""

    def mutate(self, context: FaultContext) -> None:
        for target in context.targets:
            sample = context.read(target)
            context.write(target, nan_like(sample.value), sample_time=sample.time)


class ZeroFault(TimeTriggeredFault):
    """Fault that replaces target values with zeros after ``trigger_time``."""

    def mutate(self, context: FaultContext) -> None:
        for target in context.targets:
            sample = context.read(target)
            context.write(target, zero_like(sample.value), sample_time=sample.time)