import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from syssim import NodeDifferential, InputPort, OutputPort
from typing import NamedTuple

class NodeSCRigidBodyRotationDynamicsInputs(NamedTuple):
    input_mtm_internal_sc: InputPort
    input_tau_external_sc: InputPort
    input_inertia_moment: InputPort


class NodeSCRigidBodyRotationDynamicsOutputs(NamedTuple):
    output_q_sc_to_eci: OutputPort
    output_w_sc: OutputPort


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

        self._i = NodeSCRigidBodyRotationDynamicsInputs(input_mtm_internal_sc, input_tau_external_sc, input_inertia_moment)
        self._o = NodeSCRigidBodyRotationDynamicsOutputs(output_q_sc_to_eci, output_w_sc)

        super().__init__(x0, self._i, self._o, **kwargs)

    def initialize(self):
        self._t = 0

    def update(self, sim_time: float):
        tau_ext_sc = self._i.input_tau_external_sc.read()
        h_int_sc = self._i.input_mtm_internal_sc.read()
        j = self._i.input_inertia_moment.read()

        tau_ext_sc = np.zeros((3,)) if np.any(tau_ext_sc) == None else tau_ext_sc
        h_int_sc = np.zeros((3,)) if np.any(h_int_sc) == None else h_int_sc
        j = np.eye(3) if np.any(j) == None else j

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

        self._o.output_q_sc_to_eci.shift_out(q_sc_to_eci)
        self._o.output_w_sc.shift_out(w_sc)

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
    
    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o
