import numpy as np
from scipy.spatial.transform import Rotation
from typing import NamedTuple
from syssim.core import NodeDifferential, InputPort, OutputPort


class NodeKalmanEstimatorInputs(NamedTuple):
    imu1_rate: InputPort
    imu2_rate: InputPort
    sru1_q: InputPort
    sru2_q: InputPort
    torque_cmd: InputPort


class NodeKalmanEstimatorOutputs(NamedTuple):
    est_q: OutputPort
    est_w: OutputPort


class NodeKalmanEstimator(NodeDifferential):
    def __init__(self, x0: np.ndarray, **kwargs):
        self._i = NodeKalmanEstimatorInputs(
            InputPort("imu1_rate", self),
            InputPort("imu2_rate", self),
            InputPort("sru1_q", self),
            InputPort("sru2_q", self),
            InputPort("torque_cmd", self),
        )
        self._o = NodeKalmanEstimatorOutputs(
            OutputPort("est_q", self), OutputPort("est_w", self)
        )
        # Inertia can be passed as config or default to identity
        self._inertia = kwargs.get("inertia", np.eye(3))
        self._inertia_inv = np.linalg.inv(self._inertia)

        super().__init__(x0, self._i, self._o, **kwargs)

    def initialize(self):
        self._P = np.eye(7) * 0.01
        self._Q = np.eye(7) * 1e-5  # process noise
        self._R_imu = np.eye(6) * 1e-3
        self._R_sru = np.eye(8) * 1e-2
        if "inertia" in self._config:
            self._inertia = np.array(self._config["inertia"])
            self._inertia_inv = np.linalg.inv(self._inertia)
        self._H = np.zeros((14, 7))  # Measurement matrix
        self._H[:4, :4] = np.eye(4)  # Quaternion 1
        self._H[4:8, :4] = np.eye(4)  # Quaternion 2
        self._H[8:11, 4:] = np.eye(3)  # rate 1
        self._H[11:14, 4:] = np.eye(3)  # rate 2

    def update(self, sim_time: float):
        # Read inputs and set defaults if None
        imu1 = self._i.imu1_rate.read()
        imu2 = self._i.imu2_rate.read()
        if imu1 is None:
            imu1 = np.zeros(3)
        if imu2 is None:
            imu2 = np.zeros(3)
        sru1 = self._i.sru1_q.read()
        sru2 = self._i.sru2_q.read()
        if sru1 is None:
            sru1 = np.array([1.0, 0.0, 0.0, 0.0])
        if sru2 is None:
            sru2 = np.array([1.0, 0.0, 0.0, 0.0])
        torque_cmd = self._i.torque_cmd.read()
        if torque_cmd is None:
            torque_cmd = np.zeros(3)

        # Prediction step
        q = self._x[:4]
        w = self._x[4:]
        dt = self.period
        # Euler's equation: w_dot = I^-1 * (torque_cmd - w x (I w))
        w_cross_Iw = np.cross(w, self._inertia @ w)
        w_dot = self._inertia_inv @ (torque_cmd - w_cross_Iw)
        w_pred = w + w_dot * dt
        # Quaternion kinematics
        omega = np.zeros((4, 4))
        omega[0, 1:] = -w
        omega[1:, 0] = w
        omega[1:, 1:] = -self._skew(w)
        dq = 0.5 * omega @ q * dt
        q_pred = q + dq
        q_pred /= np.linalg.norm(q_pred)
        x_pred = np.concatenate([q_pred, w_pred])
        # Linearize F for EKF
        F = self._state_update_jacobian(x_pred, dt, self._inertia)
        P_pred = F @ self._P @ F.T + self._Q

        # Update with IMU rate measurements
        H_imu = self._H[8:14, :] # Section of measurement matrix for IMU rates
        y_imu = np.concatenate([imu1, imu2]) - np.concatenate([self._x[4:], self._x[4:]]) # Innovation pre-fit
        S_imu = H_imu @ P_pred @ H_imu.T + self._R_imu # Innovation covariance
        K_imu = P_pred @ H_imu.T @ np.linalg.inv(S_imu) # Kalman gain
        x_upd = x_pred + K_imu @ y_imu
        P_upd = (np.eye(7) - K_imu @ H_imu) @ P_pred

        # Measurement update (SRU)
        H_sru = self._H[:8, :]  # Section of measurement matrix for SRU quaternions
        y_sru = np.concatenate([sru1, sru2]) - np.concatenate([self._x[:4], self._x[:4]])  # Innovation pre-fit
        S_sru = H_sru @ P_upd @ H_sru.T + self._R_sru # Innovation covariance
        K_sru = P_upd @ H_sru.T @ np.linalg.inv(S_sru) # Kalman gain
        x_final = x_upd + K_sru @ y_sru 
        P_final = (np.eye(7) - K_sru @ H_sru) @ P_upd

        # Normalize quaternion
        x_final[:4] /= np.linalg.norm(x_final[:4])
        self._x = x_final
        self._P = P_final

        # Output the estimated state
        self._o.est_q.shift_out(self._x[:4])
        self._o.est_w.shift_out(self._x[4:])

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o

    def _skew(self, x):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    @staticmethod
    def _state_update_jacobian(x: np.ndarray, dt: float, J: np.ndarray) -> np.ndarray:
        [q0, q1, q2, q3, wx, wy, wz] = x
        [J11, J12, J13, J21, J22, J23, J31, J32, J33] = J.flatten()
        return np.array(
            [
                [
                    1,
                    -0.5 * dt * wx,
                    -0.5 * dt * wy,
                    -0.5 * dt * wz,
                    -0.5 * dt * q1,
                    -0.5 * dt * q2,
                    -0.5 * dt * q3,
                ],
                [
                    0.5 * dt * wx,
                    1,
                    0.5 * dt * wz,
                    -0.5 * dt * wy,
                    0.5 * dt * q0,
                    -0.5 * dt * q3,
                    0.5 * dt * q2,
                ],
                [
                    0.5 * dt * wy,
                    -0.5 * dt * wz,
                    1,
                    0.5 * dt * wx,
                    0.5 * dt * q3,
                    0.5 * dt * q0,
                    -0.5 * dt * q1,
                ],
                [
                    0.5 * dt * wz,
                    0.5 * dt * wy,
                    -0.5 * dt * wx,
                    1,
                    -0.5 * dt * q2,
                    0.5 * dt * q1,
                    0.5 * dt * q0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                        - dt
                        * (
                            (J12 * J23 - J13 * J22)
                            * (-J11 * wy + 2 * J21 * wx + J22 * wy + J23 * wz)
                            + (J12 * J33 - J13 * J32)
                            * (-J11 * wz + 2 * J31 * wx + J32 * wy + J33 * wz)
                            - (J21 * wz - J31 * wy) * (J22 * J33 - J23 * J32)
                        )
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                    dt
                    * (
                        (J12 * J23 - J13 * J22)
                        * (J11 * wx + 2 * J12 * wy + J13 * wz - J22 * wx)
                        + (J12 * J33 - J13 * J32) * (J12 * wz - J32 * wx)
                        - (J22 * J33 - J23 * J32)
                        * (-J22 * wz + J31 * wx + 2 * J32 * wy + J33 * wz)
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                    dt
                    * (
                        (J12 * J23 - J13 * J22) * (J13 * wy - J23 * wx)
                        + (J12 * J33 - J13 * J32)
                        * (J11 * wx + J12 * wy + 2 * J13 * wz - J33 * wx)
                        + (J22 * J33 - J23 * J32)
                        * (J21 * wx + J22 * wy + 2 * J23 * wz - J33 * wy)
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    dt
                    * (
                        (J11 * J23 - J13 * J21)
                        * (-J11 * wy + 2 * J21 * wx + J22 * wy + J23 * wz)
                        + (J11 * J33 - J13 * J31)
                        * (-J11 * wz + 2 * J31 * wx + J32 * wy + J33 * wz)
                        - (J21 * J33 - J23 * J31) * (J21 * wz - J31 * wy)
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                    (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                        - dt
                        * (
                            (J11 * J23 - J13 * J21)
                            * (J11 * wx + 2 * J12 * wy + J13 * wz - J22 * wx)
                            + (J11 * J33 - J13 * J31) * (J12 * wz - J32 * wx)
                            - (J21 * J33 - J23 * J31)
                            * (-J22 * wz + J31 * wx + 2 * J32 * wy + J33 * wz)
                        )
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                    -dt
                    * (
                        (J11 * J23 - J13 * J21) * (J13 * wy - J23 * wx)
                        + (J11 * J33 - J13 * J31)
                        * (J11 * wx + J12 * wy + 2 * J13 * wz - J33 * wx)
                        + (J21 * J33 - J23 * J31)
                        * (J21 * wx + J22 * wy + 2 * J23 * wz - J33 * wy)
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    -dt
                    * (
                        (J11 * J22 - J12 * J21)
                        * (-J11 * wy + 2 * J21 * wx + J22 * wy + J23 * wz)
                        + (J11 * J32 - J12 * J31)
                        * (-J11 * wz + 2 * J31 * wx + J32 * wy + J33 * wz)
                        - (J21 * J32 - J22 * J31) * (J21 * wz - J31 * wy)
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                    dt
                    * (
                        (J11 * J22 - J12 * J21)
                        * (J11 * wx + 2 * J12 * wy + J13 * wz - J22 * wx)
                        + (J11 * J32 - J12 * J31) * (J12 * wz - J32 * wx)
                        - (J21 * J32 - J22 * J31)
                        * (-J22 * wz + J31 * wx + 2 * J32 * wy + J33 * wz)
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                    (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                        + dt
                        * (
                            (J11 * J22 - J12 * J21) * (J13 * wy - J23 * wx)
                            + (J11 * J32 - J12 * J31)
                            * (J11 * wx + J12 * wy + 2 * J13 * wz - J33 * wx)
                            + (J21 * J32 - J22 * J31)
                            * (J21 * wx + J22 * wy + 2 * J23 * wz - J33 * wy)
                        )
                    )
                    / (
                        J11 * J22 * J33
                        - J11 * J23 * J32
                        - J12 * J21 * J33
                        + J12 * J23 * J31
                        + J13 * J21 * J32
                        - J13 * J22 * J31
                    ),
                ],
            ]
        )
