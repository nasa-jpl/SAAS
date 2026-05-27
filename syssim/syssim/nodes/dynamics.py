from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from syssim.core import EmptySpec, InputPort, NodeDifferential, NodeParameter, OutputPort, input_port, output_port, parameter


@dataclass
class NodeStateSpaceInputs:
    """Input specification for ``NodeStateSpace``.

    Attributes
    ----------
    u : InputPort
        Input vector with shape ``(m,)``.
    """

    u: InputPort[np.ndarray] = input_port(np.ndarray)
    """Input port for the state-space node. Expects a 1D array of shape (m,)."""

@dataclass
class NodeStateSpaceOutputs:
    """Output specification for ``NodeStateSpace``.

    Attributes
    ----------
    y : OutputPort
        Output vector with shape ``(p,)``.
    """

    y: OutputPort[np.ndarray] = output_port(np.ndarray)
    """Output port for the state-space node. Expects a 1D array of shape (p,)."""


@dataclass
class NodeStateSpaceParameters:
    """Parameter specification for ``NodeStateSpace``.

    Attributes
    ----------
    a : NodeParameter
        State transition matrix with shape ``(n, n)``.
    b : NodeParameter
        Input matrix with shape ``(n, m)``.
    c : NodeParameter
        Output matrix with shape ``(p, n)``.
    """

    a: NodeParameter[np.ndarray] = parameter(default_factory=lambda: np.empty((0, 0)), value_type=np.ndarray)
    """State transition matrix of shape (n, n)."""
    b: NodeParameter[np.ndarray] = parameter(default_factory=lambda: np.empty((0, 0)), value_type=np.ndarray)
    """Input matrix of shape (n, m)."""
    c: NodeParameter[np.ndarray] = parameter(default_factory=lambda: np.empty((0, 0)), value_type=np.ndarray)
    """Output matrix of shape (p, n)."""


class NodeStateSpace(
    NodeDifferential[
        np.ndarray,
        NodeStateSpaceInputs,
        NodeStateSpaceOutputs,
        NodeStateSpaceParameters,
        EmptySpec,
    ]
):
    """Continuous linear state-space node without feedthrough.

    The node integrates ``x_dot = A x + B u`` and emits ``y = C x``. Input
    ``u`` defaults to zeros until a sample is available.

    Parameters
    ----------
    a : np.ndarray
        State transition matrix with shape ``(n, n)``.
    b : np.ndarray
        Input matrix with shape ``(n, m)``.
    c : np.ndarray
        Output matrix with shape ``(p, n)``.
    x0 : np.ndarray
        Initial state vector with shape ``(n,)``.
    **kwargs
        Arguments forwarded to ``NodeDifferential``.
    """

    Inputs = NodeStateSpaceInputs
    Outputs = NodeStateSpaceOutputs
    Parameters = NodeStateSpaceParameters

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
        super().__init__(np.asarray(x0, dtype=float), **kwargs)
        self.p.a.set_nominal(np.asarray(a, dtype=float))
        self.p.b.set_nominal(np.asarray(b, dtype=float))
        self.p.c.set_nominal(np.asarray(c, dtype=float))
        self.p.a.set_contract(value_type=np.ndarray, dtype=float, shape=self.p.a.value.shape)
        self.p.b.set_contract(value_type=np.ndarray, dtype=float, shape=self.p.b.value.shape)
        self.p.c.set_contract(value_type=np.ndarray, dtype=float, shape=self.p.c.value.shape)
        self.i.u.set_contract(value_type=np.ndarray, dtype=float, shape=(self.p.b.value.shape[1],))
        self.o.y.set_contract(value_type=np.ndarray, dtype=float, shape=(self.p.c.value.shape[0],))

    def initialize(self):
        """Reset integration time and state before a run."""
        self._t = 0.0
        self.reset_state()

    def update(self, sim_time: float):
        """Integrate state to ``sim_time`` and write the output sample.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds.
        """
        u = self.i.u.read().value
        if u is None:
            u = np.zeros(self.p.b.value.shape[1], dtype=float)
        u = np.asarray(u, dtype=float)
        def integrand(t, x): return self._dynamics(x, u)

        if sim_time > self._t:
            sol = solve_ivp(integrand, (self._t, sim_time), self.state)
            self.state = sol.y[:, -1]

        y = self._output(self.state, u)

        self.o.y.write(y, sim_time)

        self._t = sim_time

    def _dynamics(self, x, u):
        return self.p.a.value @ x + self.p.b.value @ u

    def _output(self, x: np.ndarray, u: np.ndarray):
        return self.p.c.value @ x
