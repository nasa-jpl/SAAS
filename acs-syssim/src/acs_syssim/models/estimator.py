import numpy as np
from typing import NamedTuple, Sequence
from syssim.core import NodeDifferential, InputPort, OutputPort


class NodeKalmanEstimatorInputs(NamedTuple):
    imu1_rate: InputPort
    imu2_rate: InputPort
    sru1_q: InputPort
    sru2_q: InputPort
    torque_cmd: InputPort
    health: InputPort
    # Added encoder scalar inputs (8)
    encoder_rate_1: InputPort
    encoder_rate_2: InputPort
    encoder_rate_3: InputPort
    encoder_rate_4: InputPort
    encoder_rate_5: InputPort
    encoder_rate_6: InputPort
    encoder_rate_7: InputPort
    encoder_rate_8: InputPort


class NodeKalmanEstimatorOutputs(NamedTuple):
    est_q: OutputPort
    est_w: OutputPort
    # Added angular momentum vector outputs (8)
    est_angmom_1: OutputPort
    est_angmom_2: OutputPort
    est_angmom_3: OutputPort
    est_angmom_4: OutputPort
    est_angmom_5: OutputPort
    est_angmom_6: OutputPort
    est_angmom_7: OutputPort
    est_angmom_8: OutputPort


class NodeKalmanEstimator(NodeDifferential):
    # Modified constructor to accept wheel axes and wheel inertias
    def __init__(
        self,
        x0: np.ndarray,
        wheel_axes: Sequence[np.ndarray] = None,
        wheel_inertias: Sequence[float] = None,
        **kwargs,
    ):
        self._i = NodeKalmanEstimatorInputs(
            InputPort("imu1_rate", self),
            InputPort("imu2_rate", self),
            InputPort("sru1_q", self),
            InputPort("sru2_q", self),
            InputPort("torque_cmd", self),
            InputPort("health", self),
            # Encoder inputs
            InputPort("encoder_rate_1", self),
            InputPort("encoder_rate_2", self),
            InputPort("encoder_rate_3", self),
            InputPort("encoder_rate_4", self),
            InputPort("encoder_rate_5", self),
            InputPort("encoder_rate_6", self),
            InputPort("encoder_rate_7", self),
            InputPort("encoder_rate_8", self),
        )
        self._o = NodeKalmanEstimatorOutputs(
            OutputPort("est_q", self),
            OutputPort("est_w", self),
            # Angular momentum outputs
            OutputPort("est_angmom_1", self),
            OutputPort("est_angmom_2", self),
            OutputPort("est_angmom_3", self),
            OutputPort("est_angmom_4", self),
            OutputPort("est_angmom_5", self),
            OutputPort("est_angmom_6", self),
            OutputPort("est_angmom_7", self),
            OutputPort("est_angmom_8", self),
        )
        # Inertia can be passed as config or default to identity
        self._inertia = kwargs.get("inertia", np.eye(3))
        self._inertia_inv = np.linalg.inv(self._inertia)

        # Wheel axes and inertias (8 wheels)
        if wheel_axes is None:
            # default: 8 x-axis unit vectors
            self._wheel_axes = np.tile(np.array([1.0, 0.0, 0.0]), (8, 1))
        else:
            self._wheel_axes = np.asarray(wheel_axes, dtype=float)
        if wheel_inertias is None:
            self._wheel_inertias = np.ones(8) * 0.01
        else:
            self._wheel_inertias = np.asarray(wheel_inertias, dtype=float)

        # Basic validation (ensure shapes)
        if self._wheel_axes.shape != (8, 3):
            raise ValueError("wheel_axes must be sequence of 8 3-element vectors")
        if self._wheel_inertias.shape != (8,):
            raise ValueError("wheel_inertias must be sequence of 8 scalars")

        super().__init__(x0, self._i, self._o, **kwargs)

    def initialize(self):
        self._P = np.eye(7) * 0.01
        self._Q = np.eye(7) * 1e-5  # process noise
        self._R_imu = np.eye(6) * self._config.get("gyro_noise", 1.24e-4) ** 2
        self._R_sru = np.eye(8) * self._config.get("sru_noise", 0.000192) ** 2
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

        # Read encoder rates (scalars), default to 0.0
        enc = []
        enc.append(self._i.encoder_rate_1.read() or 0.0)
        enc.append(self._i.encoder_rate_2.read() or 0.0)
        enc.append(self._i.encoder_rate_3.read() or 0.0)
        enc.append(self._i.encoder_rate_4.read() or 0.0)
        enc.append(self._i.encoder_rate_5.read() or 0.0)
        enc.append(self._i.encoder_rate_6.read() or 0.0)
        enc.append(self._i.encoder_rate_7.read() or 0.0)
        enc.append(self._i.encoder_rate_8.read() or 0.0)
        enc = np.asarray(enc, dtype=float)

        # Read health status
        health = self._i.health.read()
        if health is None:
            health = {}

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

        # Sequential measurement updates: IMU1, IMU2, SRU1, SRU2
        x_pred_current = x_pred
        P_pred_current = P_pred

        # IMU1 update (3x measurement) - only if healthy
        if health.get("IMU_1", "Healthy") == "Healthy":  # Default to healthy if not specified
            H_imu1 = self._H[8:11, :]  # rows for imu1 rates
            z_imu1 = imu1
            y_imu1 = z_imu1 - (H_imu1 @ x_pred_current)
            R_imu1 = self._R_imu[:3, :3]
            S = H_imu1 @ P_pred_current @ H_imu1.T + R_imu1
            K = P_pred_current @ H_imu1.T @ np.linalg.inv(S)
            x_pred_current = x_pred_current + K @ y_imu1
            P_pred_current = (np.eye(7) - K @ H_imu1) @ P_pred_current

        # IMU2 update (3x measurement) - only if healthy
        if health.get("IMU_2", "Healthy") == "Healthy":  # Default to healthy if not specified
            H_imu2 = self._H[11:14, :]  # rows for imu2 rates
            z_imu2 = imu2
            y_imu2 = z_imu2 - (H_imu2 @ x_pred_current)
            R_imu2 = self._R_imu[3:6, 3:6]
            S = H_imu2 @ P_pred_current @ H_imu2.T + R_imu2
            K = P_pred_current @ H_imu2.T @ np.linalg.inv(S)
            x_pred_current = x_pred_current + K @ y_imu2
            P_pred_current = (np.eye(7) - K @ H_imu2) @ P_pred_current

        # SRU1 update (4x measurement) - only if healthy
        if health.get("SRU_1", "Healthy") == "Healthy":  # Default to healthy if not specified
            H_sru1 = self._H[0:4, :]  # rows for sru1 quaternion
            z_sru1 = sru1
            y_sru1 = z_sru1 - (H_sru1 @ x_pred_current)
            R_sru1 = self._R_sru[:4, :4]
            S = H_sru1 @ P_pred_current @ H_sru1.T + R_sru1
            K = P_pred_current @ H_sru1.T @ np.linalg.inv(S)
            x_pred_current = x_pred_current + K @ y_sru1
            P_pred_current = (np.eye(7) - K @ H_sru1) @ P_pred_current

        # SRU2 update (4x measurement) - only if healthy
        if health.get("SRU_2", "Healthy") == "Healthy":  # Default to healthy if not specified
            H_sru2 = self._H[4:8, :]  # rows for sru2 quaternion
            z_sru2 = sru2
            y_sru2 = z_sru2 - (H_sru2 @ x_pred_current)
            R_sru2 = self._R_sru[4:8, 4:8]
            S = H_sru2 @ P_pred_current @ H_sru2.T + R_sru2
            K = P_pred_current @ H_sru2.T @ np.linalg.inv(S)
            x_final = x_pred_current + K @ y_sru2
            P_final = (np.eye(7) - K @ H_sru2) @ P_pred_current
        else:
            x_final = x_pred_current
            P_final = P_pred_current

        # Normalize quaternion and make canonical
        x_final[:4] /= np.linalg.norm(x_final[:4])
        if x_final[0] < 0:
            x_final[:4] = -x_final[:4]
        self._x = x_final
        self._P = P_final

        # Output the estimated state
        self._o.est_q.shift_out(self._x[:4])
        self._o.est_w.shift_out(self._x[4:])

        # Compute and output angular momentum for each wheel: L = I_w * axis * omega_scalar
        for i in range(8):
            axis = self._wheel_axes[i]
            Iw = self._wheel_inertias[i]
            omega_scalar = enc[i]
            L = Iw * axis * omega_scalar  # 3-vector
            # dispatch to corresponding output port
            if i == 0:
                self._o.est_angmom_1.shift_out(L)
            elif i == 1:
                self._o.est_angmom_2.shift_out(L)
            elif i == 2:
                self._o.est_angmom_3.shift_out(L)
            elif i == 3:
                self._o.est_angmom_4.shift_out(L)
            elif i == 4:
                self._o.est_angmom_5.shift_out(L)
            elif i == 5:
                self._o.est_angmom_6.shift_out(L)
            elif i == 6:
                self._o.est_angmom_7.shift_out(L)
            elif i == 7:
                self._o.est_angmom_8.shift_out(L)

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
