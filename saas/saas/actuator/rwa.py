import numpy as np
from scipy.integrate import solve_ivp

from syssim import NodeDifferential, InputPort, OutputPort


class NodeRWASimple(NodeDifferential):
    def __init__(self, x0: np.array, **kwargs):
        """A model of a simple reaction wheel assembly. Keeps track of internal angular momentum vector and just passes through the comanded torque. No saturation or noise.

        Args:
            x0 (np.array): Initial state. Represents the reaction wheel angular momentum vector.

        Ports:
            tau_cmd (np.array): the commanded torque input. 3x1 [Nm]
            rwa_mtm (np.array): the output reaction wheel angular momentum vector. 3x1 [Nms]
            rwa_tau (np.array): the output torque. 3x1 [Nm]

        Configs:
            rwa_inertia: diagonal of the reaction wheel assembly inertia moment. 3x1 [kg m^2]
        """
        input_tau_cmd = InputPort("tau_cmd", self)

        output_rwa_mtm = OutputPort("rwa_mtm", self)
        output_rwa_tau = OutputPort("rwa_tau", self)

        ports = {
            input_tau_cmd.name: input_tau_cmd,
            output_rwa_mtm.name: output_rwa_mtm,
            output_rwa_tau.name: output_rwa_tau,
        }

        # State is the reaction wheel angular momentum in the S/C frame
        super().__init__(x0, ports, **kwargs)

    def initialize(self):
        self._inertia = np.diag(self._config["rwa_inertia"])
        self._inertia_inv = np.linalg.inv(self._inertia)
        self._t = 0

    def update(self, sim_time: float):
        tau_cmd = self._ports["tau_cmd"].read()
        if np.any(tau_cmd) == None:
            tau_cmd = np.zeros((3,))

        dt = sim_time - self._t
        self._x = self._dynamics(tau_cmd) * dt

        self._t = sim_time

        self._ports["rwa_mtm"].shift_out(self._x)
        self._ports["rwa_tau"].shift_out(tau_cmd)

    def _dynamics(self, tau_cmd: np.ndarray):
        return -tau_cmd
