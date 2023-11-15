import numpy as np

from syssim import NodeDifferential, InputPort, OutputPort


class NodePointingControlSimple(NodeDifferential):
    def __init__(self, x0: np.ndarray, **kwargs):
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
        input_q_cmd = InputPort("input_q_cmd", self)
        input_q = InputPort("input_q", self)
        input_w_cmd = InputPort("input_w_cmd", self)
        input_w = InputPort("input_w", self)
        input_mtm_internal = InputPort("input_mtm_int", self)

        output_tau_cmd = OutputPort("output_tau_cmd", self)
        output_q_err = OutputPort("output_q_err", self)
        output_w_err = OutputPort("output_w_err", self)

        ports = {
            input_q_cmd.name: input_q_cmd,
            input_q.name: input_q,
            input_w_cmd.name: input_w_cmd,
            input_w.name: input_w,
            input_mtm_internal.name: input_mtm_internal,
            output_tau_cmd.name: output_tau_cmd,
            output_q_err.name: output_q_err,
            output_w_err.name: output_w_err,
        }

        super().__init__(x0, ports, **kwargs)

    def initialize(self):
        self._kp = self._config["pointing_kp"]
        self._kd = self._config["pointing_kd"]
        self._ki = self._config["pointing_ki"]
        self._inertia = np.diag(self._config["inertia_moment"])

        self._t = 0

    def update(self, sim_time: float):
        q_cmd = self._ports["input_q_cmd"].read()
        if np.any(q_cmd) is None:
            q_cmd = np.array([1, 0, 0, 0])
        q = self._ports["input_q"].read()
        if np.any(q) is None:
            q = np.array([1, 0, 0, 0])

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

        w_cmd = self._ports["input_w_cmd"].read()
        if np.any(w_cmd) is None:
            w_cmd = np.zeros((3,))
        w = self._ports["input_w"].read()
        if np.any(w) is None:
            w = np.zeros((3,))

        w_e = w_cmd - w

        mtm_in = self._ports["input_mtm_int"].read()
        if np.any(mtm_in) == None:
            mtm_in = np.zeros((3,))

        if np.linalg.norm(q_e_vec) < self._config["i_enable_limit"]:
            dt = sim_time - self._t
            self._x = self._dynamics(q_e_vec) * dt

        self._t = sim_time

        u = self._output(self._x, q_e_vec, w_e, mtm_in, w)

        self._ports["output_tau_cmd"].shift_out(u)
        self._ports["output_q_err"].shift_out(q_e_vec)
        self._ports["output_w_err"].shift_out(w_e)

    def _dynamics(self, q_e_vec: np.ndarray):
        return q_e_vec

    def _output(
        self,
        x: np.ndarray,
        q_e: np.ndarray,
        w_e: np.ndarray,
        mtm_int: np.ndarray,
        w: np.ndarray,
    ):
        if np.linalg.norm(q_e) < self._config["i_enable_limit"]:
            integral = np.clip(
                self._x, -self._config["i_mag_limit"], self._config["i_mag_limit"]
            )
        else:
            integral = np.zeros((3,))

        return self._skew(w) @ (self._inertia @ w + mtm_int) + self._inertia @ (
            self._kp * q_e + self._ki * integral + self._kd * w_e
        )

    def _skew(self, x: np.ndarray):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
