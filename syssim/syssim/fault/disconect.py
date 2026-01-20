from typing import Union
from syssim.core.fault import Fault
import numpy as np

class DisconnectFault(Fault):
    """Fault that simulates a disconnection in the system."""

    def __init__(self, name: str, trigger_time: float = 0.0):
        """Create a disconnection fault.

        Parameters
        ----------
        name : str
            Fault name.
        trigger_time : float, optional
            Time when the disconnection begins.
        """
        super().__init__(name)
        self._trigger_time = trigger_time
        self._triggered = False

    def action(self, value: any):
        """Simulate disconnection by returning NaN-filled arrays."""
        
        # Check that the type is a NDArray
        assert isinstance(value, np.ndarray), "Value must be a numpy ndarray."
        if self._triggered:
            return np.full_like(value, np.nan)  # Return NaN to indicate disconnection
        return value  # Return the original value if not disconnected
    
    def update(self, time):
        """Update fault state based on the current simulation time."""
        if time >= self._trigger_time and not self._triggered:
            self._triggered = True
    
class ZeroFault(Fault):
    """Fault that sets the output to zero after a certain time."""
    def __init__(self, name: str, trigger_time: float = 0.0):
        """Create a zeroing fault.

        Parameters
        ----------
        name : str
            Fault name.
        trigger_time : float, optional
            Time when output is forced to zero.
        """
        super().__init__(name)
        self._trigger_time = trigger_time
        self._triggered = False

    def action(self, value: any):
        """Set the output to zero after the trigger time."""
        if self._triggered:
            if isinstance(value, np.ndarray):
                return np.zeros_like(value)
            elif isinstance(value, (float, int)):
                return 0.0
            else:
                raise TypeError("Value must be a numpy ndarray or a float/int.")
        return value
    
    def update(self, time):
        if time >= self._trigger_time and not self._triggered:
            self._triggered = True