from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from syssim.core import Fault, FaultContext, InputPort, OutputPort


@dataclass
class FaultBasicConfig:
    """Configuration for :class:`FaultBasic`.

    ``start_time_distribution``, ``duration_distribution``, and
    ``value_distribution`` are optional zero-argument callables for stochastic
    runs. Fixed defaults keep the common deterministic case small.
    """

    name: str | None = None
    start_time: float = 0.0
    duration: float = float("inf")
    occurrence: float = 1.0
    action: str = "hold"
    value: Any = 0.0
    index: int | Sequence[int] | slice = 0
    start_time_distribution: Callable[[], float] | None = None
    duration_distribution: Callable[[], float] | None = None
    value_distribution: Callable[[], Any] | None = None


class FaultBasic(Fault):
    Config = FaultBasicConfig

    def __init__(
        self,
        config: FaultBasicConfig | None = None,
        port: InputPort | OutputPort | None = None,
        targets: Iterable[object] | None = None,
        *,
        enabled: bool = True,
    ):
        """Create a basic index/value fault from a dataclass config."""
        self.config = config or FaultBasicConfig()
        if not isinstance(self.config, FaultBasicConfig):
            raise TypeError("FaultBasic config must be a FaultBasicConfig dataclass instance or None")
        all_targets = list(targets or [])
        if port is not None:
            all_targets.append(port)
        super().__init__(name=self.config.name, targets=all_targets, enabled=enabled)
        self._port = port
        self._state = (None, None, None)

    def __repr__(self) -> str:
        if self._state[2] == True:
            return f"Basic Fault [{self.name}]: start={self._state[0]}, duration={self._state[1]}"
        else:
            return f"Basic Fault [{self.name}]: does not occur"

    def initialize(self):
        super().initialize()
        """Generate fault realization for the current batch."""
        self._state = (
            _draw(self.config.start_time, self.config.start_time_distribution),
            _draw(self.config.duration, self.config.duration_distribution),
            np.random.choice([True, False], p=[self.config.occurrence, 1 - self.config.occurrence]),
        )

    def start_time(self) -> float:
        """Start time for this realization."""
        return self._state[0]

    def duration(self) -> float:
        """Duration for this realization."""
        return self._state[1]

    def is_occuring(self) -> bool:
        """Whether the fault occurs in the current realization."""
        return self._state[2]

    def get_name(self) -> str:
        """Return the fault name."""
        return self.name

    def trigger(self, context: FaultContext) -> bool:
        if self._state[0] is None:
            self.initialize()
        start = self.start_time()
        end = start + self.duration()
        return bool(self.is_occuring() and start <= context.time and context.time < end - 1e-12)

    def mutate(self, context: FaultContext) -> None:
        for target in context.targets:
            sample = context.read(target)
            context.write(target, self._mutate_value(sample.value), sample_time=sample.time)

    def _mutate_value(self, value: Any):
        v = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        action_type = self.config.action
        index = _indices(self.config.index)
        if action_type == "random":
            for i in index:
                v[i] = _draw(self.config.value, self.config.value_distribution)

        elif action_type == "hold":
            if isinstance(self.config.value, list):
                vals = self.config.value
                if len(vals) != len(index):
                    raise Exception(
                        "For hold, must provide only one value or list of values the same length as index."
                    )
                for i, val in zip(index, vals):
                    v[i] = val
            else:
                for i in index:
                    v[i] = self.config.value
        elif action_type == "disconnect":
            for i in index:
                v[i] = np.nan
        else:
            raise Exception(f"Action type of {action_type} not known.")

        return v


def _draw(default: Any, distribution: Callable[[], Any] | None):
    return distribution() if distribution is not None else default


def _indices(index: int | Sequence[int] | slice) -> list[int]:
    if isinstance(index, slice):
        start = 0 if index.start is None else index.start
        if index.stop is None:
            raise ValueError("FaultBasic slice indices must define stop")
        step = 1 if index.step is None else index.step
        return list(range(start, index.stop, step))
    if isinstance(index, int):
        return [index]
    return list(index)
