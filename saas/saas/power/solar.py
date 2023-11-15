import numpy as np
from scipy.spatial.transform import Rotation

from syssim.core import NodeDifferential, InputPort, OutputPort


class NodeSimpleGimbalSolarPanel(NodeDifferential):
    def __init__(self, **kwargs):
        in_solar_constant = InputPort("in_solar_constant", self)
        in_sun_unit_sc = InputPort("in_sun_unit_sc", self)
        in_sp_gimbal_angle = InputPort("in_sp_gimbal_angle", self)
        in_is_occluded = InputPort("in_is_occluded", self)
        input_q_sc2eci = InputPort("input_q_sc2eci", self)

        out_power = OutputPort("out_power", self)
        out_gimbal = OutputPort("out_gimbal", self)
        out_gimbal_measure = OutputPort("out_gimbal_measure", self)

        ports = {
            in_solar_constant.name: in_solar_constant,
            in_sun_unit_sc.name: in_sun_unit_sc,
            in_sp_gimbal_angle.name: in_sp_gimbal_angle,
            in_is_occluded.name: in_is_occluded,
            input_q_sc2eci.name: input_q_sc2eci,
            out_power.name: out_power,
            out_gimbal.name: out_gimbal,
            out_gimbal_measure.name: out_gimbal_measure,
        }

        super().__init__(np.array([0.0]), ports, **kwargs)

    def initialize(self):
        self._area = self._config["area"]
        self._efficiency = self._config["efficiency"]
        self._sp_unit = np.array([0, 0, 1])
        self._gimbal_max_angle = np.radians(self._config["gimbal_max_angle"])
        self._panel_voltage = self._config["panel_voltage"]
        self._time_const = self._config["gimbal_time_constant"]
        self._encoder_noise = np.radians(self._config["encoder_noise_std"])
        self._t = 0

    def update(self, sim_time: float):
        in_sun_unit_sc = self._ports["in_sun_unit_sc"].read()
        q_sc2eci = self._ports["input_q_sc2eci"].read()
        if np.any(q_sc2eci) == None:
            r_sc2eci = Rotation.identity()
        else:
            r_sc2eci = Rotation.from_quat(
                [q_sc2eci[1], q_sc2eci[2], q_sc2eci[3], q_sc2eci[0]]
            )

        if np.any(in_sun_unit_sc) == None:
            in_sun_unit_sc = np.array([0, 0, 1])
        in_sp_gimbal_angle = np.clip(
            self._ports["in_sp_gimbal_angle"].read()
            if self._ports["in_sp_gimbal_angle"].read() != None
            else 0,
            -self._gimbal_max_angle,
            self._gimbal_max_angle,
        )
        solar_constant = self._ports["in_solar_constant"].read()
        is_occluded = self._ports["in_is_occluded"].read()

        if in_sp_gimbal_angle == None:
            in_sp_gimbal_angle = 0

        dt = sim_time - self._t
        self._x += self._dynamics(self._x, in_sp_gimbal_angle) * dt
        self._t = sim_time

        if is_occluded == False:
            r_sp2sc = r_sc2eci * Rotation.from_euler("y", self._x)

            sp_unit_sc = r_sp2sc.apply(self._sp_unit)[0, :]
            proj_area = np.clip(
                self._area * np.dot(in_sun_unit_sc, sp_unit_sc), 0, None
            )
            self._ports["out_power"].shift_out(
                proj_area * solar_constant * self._efficiency / self._panel_voltage
            )
        else:
            self._ports["out_power"].shift_out(0)

        self._ports["out_gimbal"].shift_out(self._x)
        self._ports["out_gimbal_measure"].shift_out(
            np.random.normal(self._x, self._encoder_noise)
        )
        # print(np.degrees(self._x))

    def _dynamics(self, angle: float, cmd_angle: float):
        return (cmd_angle - angle) / self._time_const
