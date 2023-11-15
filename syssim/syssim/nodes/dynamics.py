import numpy as np
from scipy.integrate import solve_ivp

from syssim.core import NodeDifferential
from syssim.core import InputPort, OutputPort


class NodeStateSpace(NodeDifferential):

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, x0: np.ndarray, **kwargs):
        """Implements an LTI system. Note that there is no D matrix because in the current system simulation framework, the output of a NodeDifferential cannot rely on the input at the current time step.

        Args:
            a (np.ndarray): System A matrix
            b (np.ndarray): System B matrix
            c (np.ndarray): System C matrix
            x0 (np.ndarray): Initial state
        Ports:
            input_u (np.array) input to the system
            output_y (np.array) output of the system            
        """
        self._a = a
        self._b = b
        self._c = c

        input_u = InputPort("input_u", self)
        output_y = OutputPort("output_y", self)

        ports = {input_u.name: input_u, output_y.name: output_y}

        super().__init__(x0, ports, **kwargs)

    def initialize(self):
        self._t = 0

    def update(self, sim_time: float):
        u = self._ports["input_u"].read()
        def integrand(t, x): return self._dynamics(x, u)

        sol = solve_ivp(integrand, (self._t, sim_time), self._x)

        self._t = sim_time
        self._x = sol.y[:, -1]

        y = self._output(self._x, u)

        self._ports["output_y"].shift_out(y)

        self._t = sim_time

    def _dynamics(self, x, u):
        return self._a @ x + self._b @ u

    def _output(self, x: np.ndarray, u: np.ndarray):
        return self._c @ x
