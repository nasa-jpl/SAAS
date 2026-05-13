from typing import NamedTuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from syssim.core import NodeDifferential
from syssim.core import InputPort, OutputPort


class NodeStateSpaceInputs(NamedTuple):

    u: InputPort
    """Input vector for the SS system"""


class NodeStateSpaceOutputs(NamedTuple):

    y: OutputPort
    """Output vector for the SS system"""


class NodeStateSpace(NodeDifferential):

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, x0: np.ndarray, **kwargs):
        """Linear time-invariant state-space node without feedthrough.

        Parameters
        ----------
        a : np.ndarray
            State transition matrix.
        b : np.ndarray
            Input matrix.
        c : np.ndarray
            Output matrix.
        x0 : np.ndarray
            Initial state vector.

        Notes
        -----
        ``D`` is omitted because :class:`NodeDifferential` forbids dependence
        on the current-time input during output computation.
        """
        self._a = a
        self._b = b
        self._c = c

        self._i = NodeStateSpaceInputs(InputPort("input_u", self))
        self._o = NodeStateSpaceOutputs(OutputPort("output_y", self))

        super().__init__(x0, self._i, self._o, **kwargs)

    def initialize(self):
        self._t = 0

    def update(self, sim_time: float):
        u = self._i.u.read()
        def integrand(t, x): return self._dynamics(x, u)

        sol = solve_ivp(integrand, (self._t, sim_time), self._x)

        self._t = sim_time
        self._x = sol.y[:, -1]

        y = self._output(self._x, u)

        self._o.y.shift_out(y, sim_time)

        self._t = sim_time

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o

    def _dynamics(self, x, u):
        return self._a @ x + self._b @ u

    def _output(self, x: np.ndarray, u: np.ndarray):
        return self._c @ x
