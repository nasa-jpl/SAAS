import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from syssim import NodeDifferential, InputPort, OutputPort

from saas.utility import get_poliastro_body


class NodeSCOrbitalDynamics(NodeDifferential):
    def __init__(self, x0: np.ndarray, **kwargs):
        """Orbital dynamics implemented using numerical integration. Only environmental effect is a central body force.

        Args:
            x0 (np.ndarray): Initial position and velocity of the spacecraft. Stacked in a 6x1 [m], [m s^-1]

        Ports:
            input_m_ex (np.array): total amount of mass expended from the spacecraft. 1x1 [kg]
            input_f_ext (np.array): total external force. 3x1 [N]
            output_r_eci (np.array): output spacecraft position. 3x1 [m]
            output_v_eci (np.array): output spacecraft velocity. 3x1 [m s^-1]
            output_a_eci (np.array):  output spacecraft total acceleration. 3x1 [m s^-2]
            output_a_ff_eci (np.array): output spacecraft free-fall acceleration (acceleration without gravity). 3x1 [m s^-1]
        Configs:
            central_body_name (str): name of the central body (e.g., "mars")
            m0 (float): initial mass [kg]
        """
        input_mass = InputPort("input_mass", self)
        input_ext_forces = InputPort("input_f_ext", self)

        output_pos_eci = OutputPort("output_r_eci", self)
        output_vel_eci = OutputPort("output_v_eci", self)
        output_acc_eci = OutputPort("output_a_eci", self)
        output_acc_freefall_eci = OutputPort("output_a_ff_eci", self)

        ports = {
            input_mass.name: input_mass,
            input_ext_forces.name: input_ext_forces,
            output_pos_eci.name: output_pos_eci,
            output_vel_eci.name: output_vel_eci,
            output_acc_eci.name: output_acc_eci,
            output_acc_freefall_eci.name: output_acc_freefall_eci,
        }

        super().__init__(x0, ports, **kwargs)

    def initialize(self):
        self._t = 0

        self._mu = get_poliastro_body(self._config["central_body_name"]).k.value

    def update(self, sim_time: float):
        f_ext = self._ports["input_f_ext"].read()
        m = self._ports["input_mass"].read()

        f_ext = np.zeros((3,)) if f_ext == None else f_ext

        def integrand(t, x):
            return self._dynamics(x, f_ext, m)

        sol = solve_ivp(integrand, (self._t, sim_time), self._x)

        self._t = sim_time
        self._x = sol.y[:, -1]

        r_eci, v_eci, a_eci, a_ff_eci = self._output(self._x, f_ext, m)

        self._ports["output_r_eci"].shift_out(r_eci)
        self._ports["output_v_eci"].shift_out(v_eci)
        self._ports["output_a_eci"].shift_out(a_eci)
        self._ports["output_a_ff_eci"].shift_out(a_ff_eci)

    def _dynamics(self, x: np.ndarray, f_ext: np.ndarray, m: float):
        r = x[0:3]
        norm_r = np.linalg.norm(r)

        a_grav = -self._mu / norm_r**3 * r
        a_ext = f_ext / m

        a = a_grav + a_ext
        return np.concatenate([x[3:6], a])

    def _output(self, x: np.ndarray, f_ext: np.ndarray, m: float):
        r_eci = x[0:3]
        v_eci = x[3:6]

        norm_r = np.linalg.norm(r_eci)

        a_grav = -self._mu / norm_r**3 * r_eci
        a_ext = f_ext / m

        a_eci = a_grav + a_ext
        a_ff_eci = a_ext

        return r_eci, v_eci, a_eci, a_ff_eci


class NodeSCRigidBodyRotationDynamics(NodeDifferential):
    def __init__(self, x0: np.ndarray, **kwargs):
        """Spacecraft rigid body rotational dynamics. Implementation of Euler's equation for rigid bodies. Orientation represented as a quaternion.

        Args:
            x0 (np.ndarray): Initial orientation and angular velocity of the spacecraft. Angular velcoity given in the body frame. 7x1 stacked qaternion and angular velocity. Quaternion scalar first. Angular velocity [rad s^-1].

        Ports:
            input_mtm_internal_sc (np.array): input angular momentum that is internal to the spacecraft (like that from a reaction wheel assembly). In the body frame. 3x1 [Nms]
            input_tau_external_sc (np.array): input external torque. In the body frame. 3x1 [Nm]
            output_q_sc_to_eci (np.array):  output spacecraft orientation as a quaternion. 4x1 scalar first
            output_w_sc (np.array): output spacecraft angular velocity. In body frame. 3x1 [rad s^-1]

        Configs:
            inertia_moment: spacecraft inertia moment diagonal. 3x1 [kg m^2]
        """
        input_mtm_internal_sc = InputPort("input_mtm_internal_sc", self)
        input_tau_external_sc = InputPort("input_tau_external_sc", self)
        input_inertia_moment = InputPort("input_inertia_moment", self)

        output_q_sc_to_eci = OutputPort("output_q_sc_to_eci", self)
        output_w_sc = OutputPort("output_w_sc", self)

        ports = {
            input_mtm_internal_sc.name: input_mtm_internal_sc,
            input_tau_external_sc.name: input_tau_external_sc,
            input_inertia_moment.name: input_inertia_moment,
            output_q_sc_to_eci.name: output_q_sc_to_eci,
            output_w_sc.name: output_w_sc,
        }

        super().__init__(x0, ports, **kwargs)

    def initialize(self):
        self._t = 0

    def update(self, sim_time: float):
        tau_ext_sc = self._ports["input_tau_external_sc"].read()
        h_int_sc = self._ports["input_mtm_internal_sc"].read()
        j = self._ports["input_inertia_moment"].read()

        tau_ext_sc = np.zeros((3,)) if np.any(tau_ext_sc) == None else tau_ext_sc
        h_int_sc = np.zeros((3,)) if np.any(h_int_sc) == None else h_int_sc

        def integrand(t, x):
            return self._dynamics(x, h_int_sc, tau_ext_sc, j)

        sol = solve_ivp(integrand, (self._t, sim_time), self._x)

        self._t = sim_time
        x_tmp = sol.y[:, -1]
        # Quaternion regularization
        qx, qy, qz, qw = Rotation.from_quat(
            [x_tmp[1], x_tmp[2], x_tmp[3], x_tmp[0]]
        ).as_quat(canonical=True)
        x_tmp[0:4] = [qw, qx, qy, qz]
        self._x = x_tmp

        q_sc_to_eci, w_sc = self._output(self._x)

        self._ports["output_q_sc_to_eci"].shift_out(q_sc_to_eci)
        self._ports["output_w_sc"].shift_out(w_sc)

    def _dynamics(self, x: np.ndarray, mtm_int_sc: np.ndarray, tau_ext_sc: np.ndarray, j: np.ndarray):
        qw, qx, qy, qz = x[0:4]
        w_sc = x[4:7]

        # Quaternion derivative matrix
        G = np.array([[-qx, qw, qz, -qy], [-qy, -qz, qw, qx], [-qz, qy, -qx, qw]])

        angacc = np.linalg.inv(j) @ (
            tau_ext_sc - np.cross(w_sc, j @ w_sc + mtm_int_sc, axis=0)
        )

        dq = 1 / 2 * G.transpose() @ w_sc

        xsdot = np.array(
            [
                dq[0],
                dq[1],
                dq[2],
                dq[3],
                angacc[0],
                angacc[1],
                angacc[2],
            ]
        )
        return np.squeeze(xsdot)

    def _output(self, x: np.ndarray):
        q_sc_to_eci = x[0:4]
        w_sc = x[4:7]

        return q_sc_to_eci, w_sc
