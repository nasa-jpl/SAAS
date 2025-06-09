from typing import Union, Any
from abc import ABC, abstractmethod

from syssim.core.port import InputPort, OutputPort

class Fault(ABC):
    """Abstract base class for faults in the system. All faults should inherit from this class and implement the required methods."""

    def __init__(self, name: str):
        """Initialize the fault with a name.

        Args:
            name (str): Name of the fault.
            port (Union[InputPort, OutputPort]): The port associated with the fault. This can be either an InputPort or an OutputPort.
        """
        assert isinstance(name, str), "Fault name must be a string."

        self._name = name
        self._port: InputPort | OutputPort | None = None
        self._active = True

    @abstractmethod
    def action(self, value: Any) -> Any:
        """Return modified value based on the fault action at a given time."""
        pass

    def initialize(self):
        """Initialize the fault state. This method can be overridden by subclasses to perform any necessary setup."""
        pass

    def update(self, time: float):
        """Update the fault state based on the current simulation time."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the fault.

        Returns:
            str: Name of the fault.
        """
        return self._name
    
    @name.setter
    def name(self, new_name: str):
        """Set a new name for the fault.

        Args:
            new_name (str): New name for the fault.
        """
        assert isinstance(new_name, str), "Fault name must be a string."
        self._name = new_name

    @property
    def port(self) -> Union[InputPort, OutputPort]:
        """Get the port associated with the fault.

        Returns:
            Union[InputPort, OutputPort]: The port associated with the fault.
        """
        return self._port
    
    @port.setter
    def port(self, new_port: Union[InputPort, OutputPort]):
        """Set a new port for the fault.

        Args:
            new_port (Union[InputPort, OutputPort]): New port for the fault.
        """
        assert isinstance(new_port, (InputPort, OutputPort)), "Port must be an InputPort or OutputPort."
        self._port = new_port

    @property
    def active(self) -> bool:
        """Check if the fault is currently active.

        Returns:
            bool: True if the fault is active, False otherwise.
        """
        return self._active
    
    @active.setter
    def active(self, value: bool):
        """Set the active state of the fault.

        Args:
            value (bool): True to activate the fault, False to deactivate it.
        """
        assert isinstance(value, bool), "Active state must be a boolean."
        self._active = value