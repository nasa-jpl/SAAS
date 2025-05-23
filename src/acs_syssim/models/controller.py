import numpy as np

from syssim import Node, InputPort, OutputPort
from typing import NamedTuple


class NodeRateControlSimpleInputs(NamedTuple):
    input_w_cmd: InputPort
    input_w: InputPort
    input_mtm_int: InputPort
    input_sc_inertia_moment: InputPort


class NodeRateControlSimpleOutputs(NamedTuple):
    output_tau_cmd: OutputPort
    output_w_err: OutputPort


class NodeRateControlSimple(Node):
    def __init__(self, **kwargs):
        """A simple pointing controller based on quaternion feedback. Based on the reference [1] with added integral term.
        [1] B. Wie, H. Weiss, and A. Arapostathis, “Quarternion feedback regulator for spacecraft eigenaxis rotations,” Journal of Guidance, Control, and Dynamics, vol. 12, no. 3, pp. 375–380, May 1989, doi: 10.2514/3.20418.


        Args:
            x0 (np.ndarray): the initial value for the integral term.

        Ports:
            input_w_cmd (np.array): input angular rate comand. 3x1 [rad s^-1]
            input_w (np.array):  input measured angular rate. 3x1 [rad s^-1]
            input_mtm_int (np.array):  input measured internal angular momentum. 3x1 [Nms]
            output_tau_cmd (np.array): output torque command. 3x1 [Nm]
            output_w_err (np.array): output angular velocity error. 3x1 [rad s^-1]
        Configs:
            inertia_moment: spacecraft inertia moment diagonal. 3x1 [kg m^2]
            pointing_kp: proportional gain

        """
        self._i = NodeRateControlSimpleInputs(
            InputPort("input_w_cmd", self),
            InputPort("input_w", self),
            InputPort("input_mtm_int", self),
            InputPort("input_sc_inertia_moment", self)
        )

        self._o = NodeRateControlSimpleOutputs(
            OutputPort("output_tau_cmd", self),
            OutputPort("output_w_err", self)
        )

        super().__init__(self._i, self._o, **kwargs)

    def initialize(self):
        self._kp = self._config["pointing_kp"]

        self._t = 0

    def update(self, sim_time: float):
        self._inertia = self._i.input_sc_inertia_moment.read()

        w_cmd = self._i.input_w_cmd.read()
        if np.any(w_cmd) is None:
            w_cmd = np.zeros((3,))
        w = self._i.input_w.read()
        if np.any(w) is None:
            w = np.zeros((3,))

        w_e = w_cmd - w

        mtm_in = self._i.input_mtm_int.read()
        if np.any(mtm_in) == None:
            mtm_in = np.zeros((3,))

        u =  self._skew(w) @ (self._inertia @ w + mtm_in) + self._inertia @ (
            self._kp * w_e
        )

        self._t = sim_time

        self._o.output_tau_cmd.shift_out(u)
        self._o.output_w_err.shift_out(w_e)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o

    def _skew(self, x: np.ndarray):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
