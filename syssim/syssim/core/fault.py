from typing import Union, Any
from abc import ABC, abstractmethod

from syssim.core.port import InputPort, OutputPort

class Fault(ABC):
    """Abstract base class for faults in the system. All faults should inherit from this class and implement the required methods."""

    def __init__(self, name: str, trigger_time: float = 0.0):
        """Create a fault with a trigger time.

        Parameters
        ----------
        name : str
            Fault name.
        trigger_time : float, optional
            Time at which the fault becomes eligible to trigger.
        """
        assert isinstance(name, str), "Fault name must be a string."

        self._name = name
        self._port: InputPort | OutputPort | None = None
        self._active = True
        self._trigger_time = float(trigger_time)
        self._triggered = False

    @abstractmethod
    def action(self, value: Any) -> Any:
        """Return modified value based on the fault action at a given time."""
        pass

    def initialize(self):
        """Initialize fault state prior to a simulation batch."""
        pass

    def update(self, time: float):
        """Advance fault state based on simulation time."""
        if not self._triggered and time >= self._trigger_time:
            self._triggered = True

    @property
    def name(self) -> str:
        """Fault name."""
        return self._name
    
    @name.setter
    def name(self, new_name: str):
        """Set the fault name."""
        assert isinstance(new_name, str), "Fault name must be a string."
        self._name = new_name

    @property
    def port(self) -> Union[InputPort, OutputPort]:
        """Port associated with the fault."""
        return self._port
    
    @port.setter
    def port(self, new_port: Union[InputPort, OutputPort]):
        """Set the port associated with the fault."""
        assert isinstance(new_port, (InputPort, OutputPort)), "Port must be an InputPort or OutputPort."
        self._port = new_port

    @property
    def active(self) -> bool:
        """Whether the fault is currently active."""
        return self._active
    
    @active.setter
    def active(self, value: bool):
        """Set active state for the fault."""
        assert isinstance(value, bool), "Active state must be a boolean."
        self._active = value

    @property
    def triggered(self) -> bool:
        """Whether the fault has crossed its trigger time."""
        return self._triggered

    @property
    def trigger_time(self) -> float:
        """Trigger time for the fault."""
        return self._trigger_time