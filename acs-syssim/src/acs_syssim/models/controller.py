import numpy as np

from syssim import Node, InputPort, OutputPort
from typing import NamedTuple


class NodePointingControlSimpleInputs(NamedTuple):
    input_q_cmd: InputPort
    input_q: InputPort
    input_w_cmd: InputPort
    input_w: InputPort
    input_mtm_int: InputPort


class NodePointingControlSimpleOutputs(NamedTuple):
    output_tau_cmd: OutputPort
    output_q_err: OutputPort
    output_w_err: OutputPort


class NodePointingControlSimple(Node):
    def __init__(self, **kwargs):
        """A simple pointing controller based on quaternion feedback. Based on the reference [1] with added integral term.
        [1] B. Wie, H. Weiss, and A. Arapostathis, “Quarternion feedback regulator for spacecraft eigenaxis rotations,” Journal of Guidance, Control, and Dynamics, vol. 12, no. 3, pp. 375–380, May 1989, doi: 10.2514/3.20418.


        Args:
            x0 (np.ndarray): the initial value for the integral term.

        Ports:
            input_q_cmd (np.array): input quaternion comand. 4x1 scalar first
            input_q (np.array): input measured quaternion. 4x1 scalar first
            input_w_cmd (np.array): input angular rate comand. 3x1 [rad s^-1]
            input_w (np.array):  input measured angular rate. 3x1 [rad s^-1]
            input_mtm_int (np.array):  input measured internal angular momentum. 3x1 [Nms]
            output_tau_cmd (np.array): output torque command. 3x1 [Nm]
            output_q_err (np.array): output quaternion error vector. 3x1
            output_w_err (np.array): output angular velocity error. 3x1 [rad s^-1]
        Configs:
            inertia_moment: spacecraft inertia moment diagonal. 3x1 [kg m^2]
            pointing_kp: proportional gain
            pointing_kd = derivative gain
            pointing_ki = integral gain
            i_mag_limit = limit on the magnitude of the integral term
            i_enable_limit = limit on the error under which to enable the integral term
        """
        self._i = NodePointingControlSimpleInputs(
            InputPort("input_q_cmd", self),
            InputPort("input_q", self),
            InputPort("input_w_cmd", self),
            InputPort("input_w", self),
            InputPort("input_mtm_int", self),
        )

        self._o = NodePointingControlSimpleOutputs(
            OutputPort("output_tau_cmd", self),
            OutputPort("output_q_err", self),
            OutputPort("output_w_err", self)
        )

        super().__init__(self._i, self._o, **kwargs)

    def initialize(self):
        self._kp = self._config["pointing_kp"]
        self._kd = self._config["pointing_kd"]
        if "inertia" in self._config:
            self._inertia = np.array(self._config["inertia"])
        else:
            self._inertia = np.eye(3) 
        self._t = 0

    def update(self, sim_time: float):
        q_cmd = self._i.input_q_cmd.read()
        # if np.any(q_cmd) is None:
        #     q_cmd = np.array([1, 0, 0, 0])
        q = self._i.input_q.read()
        # if np.any(q) is None:
        #     q = np.array([1, 0, 0, 0])

        q_cmd_w, q_cmd_x, q_cmd_y, q_cmd_z = q_cmd
        qw, qx, qy, qz = q

        qc = np.array(
            [
                [q_cmd_w, q_cmd_z, -q_cmd_y, -q_cmd_x],
                [-q_cmd_z, q_cmd_w, q_cmd_x, -q_cmd_y],
                [q_cmd_y, -q_cmd_x, q_cmd_w, -q_cmd_z],
                [q_cmd_x, q_cmd_y, q_cmd_z, q_cmd_w],
            ]
        )

        q_e_vec = (qc @ np.array([qx, qy, qz, qw]))[0:3]

        w_cmd = self._i.input_w_cmd.read()
        # if np.any(w_cmd) is None:
        #     w_cmd = np.zeros((3,))
        w = self._i.input_w.read()
        # if np.any(w) is None:
        #     w = np.zeros((3,))

        w_e = w - w_cmd

        mtm_in = self._i.input_mtm_int.read()
        if np.any(mtm_in) == None:
            mtm_in = np.zeros((3,))

        u = self._output(q_e_vec, w_e, mtm_in, w)

        self._o.output_tau_cmd.shift_out(u)
        self._o.output_q_err.shift_out(q_e_vec)
        self._o.output_w_err.shift_out(w_e)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o

    def _output(
        self,
        q_e: np.ndarray,
        w_e: np.ndarray,
        mtm_int: np.ndarray,
        w: np.ndarray,
    ):

        return -self._kp * q_e - self._kd * w_e + self._skew(w) @ (self._inertia @ w + mtm_int)

        # return self._skew(w) @ (self._inertia @ w + mtm_int) + self._inertia @ (
        #     self._kp * q_e + self._ki * integral + self._kd * w_e
        # )

    def _skew(self, x: np.ndarray):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
