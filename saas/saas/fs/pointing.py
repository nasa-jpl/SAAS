from typing import Dict, Union
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from math import degrees

from syssim import Node, InputPort, OutputPort
from syssim.core.port import InputPort, OutputPort


class NodePointingNadir(Node):
    def __init__(self, **kwargs):
        """Node for calculating the spacecraft comands in the nadir pointing mode. Ensures the spacecraft adaquately tracks the nadir direction.
        
        Ports:
            input_r_eci (np.array) 
            input_v_eci (np.array)
            input_s_b (np.array)
            output_q_cmd (np.array)
            output_w_cmd (np.array)
            output_gimbal_cmd (np.array)
            output_nadir (np.array)
        Configs:


        """
        input_pos_eci = InputPort("input_r_eci", self)
        input_v_eci = InputPort("input_v_eci", self)
        input_s_b = InputPort("input_s_b", self)

        output_q_cmd = OutputPort("output_q_cmd", self)
        output_w_cmd = OutputPort("output_w_cmd", self)
        output_gimbal_cmd = OutputPort("output_gimbal_cmd", self)
        output_nadir = OutputPort("output_nadir", self)

        ports = {
            input_pos_eci.name: input_pos_eci,
            input_v_eci.name: input_v_eci,
            input_s_b.name: input_s_b,
            output_q_cmd.name: output_q_cmd,
            output_w_cmd.name: output_w_cmd,
            output_gimbal_cmd.name: output_gimbal_cmd,
            output_nadir.name: output_nadir,
        }

        super().__init__(ports, **kwargs)

    # def initialize(self):
    #     self._prev_q = [0, 0, 0, 1]

    def update(self, sim_time: float):
        # if sim_time == 0:
        #     return  # skip first iteration

        # fs_state = self._ports["input_fs_state"].read()
        # if fs_state != SimpleFlightState.SCIENCE:
        #     return  # skip if in the wrong flight state

        # TODO should revisit if these default values make sense. We might want to just fail here.
        r_eci = self._ports["input_r_eci"].read()
        if np.any(r_eci) == None:
            r_eci = np.ones((3,))
        v_eci = self._ports["input_v_eci"].read()
        if np.any(v_eci) == None:
            v_eci = np.ones((3,))

        # If no secondary body axis, use y
        s_b = self._ports["input_s_b"].read()
        if np.any(s_b) == None:
            s_b = np.array([0, 1, 0])

        r_norm = np.linalg.norm(r_eci)
        v_norm = np.linalg.norm(v_eci)

        # Nadir vector
        nadir = -r_eci / r_norm  # n_icrs
        orbit_normal = np.cross(r_eci, v_eci) / (r_norm * v_norm)

        # ignore any x component in the body reference direction as we can't gaurentee that
        s_b[0] = 0
        s_b_unit = s_b / np.linalg.norm(s_b)

        # Rotation comand
        # TODO, this algorithm seems to chatter and get stuck sometimes.
        # Suspect this is because there are two rotations that will work
        # to align the vectors and the algorithm oscilates between them.
        #  Might want to find a way to smooth that out. Maybe by preserving
        # the sign on the quaternion. Seems form looking at it that the
        # vector part of the quaternion is getting reversed.
        r_cmd, rssd = Rotation.align_vectors(
            np.vstack([nadir, orbit_normal]),
            np.vstack([np.array([1, 0, 0]), s_b_unit]),
            weights=[1.0, 0.1],
        )

        q_cmd = r_cmd.as_quat(canonical=True)

        # Handle discontinuity
        # if np.dot(-q_cmd[0:3], self._prev_q[0:3]) > np.dot(q_cmd[0:3], self._prev_q[0:3]):
        #     q_cmd = -q_cmd

        # self._prev_q = q_cmd

        # Angular velocity comand feed forward
        w_cmd = v_norm / r_norm * s_b_unit

        self._ports["output_q_cmd"].shift_out(
            np.array([q_cmd[3], q_cmd[0], q_cmd[1], q_cmd[2]])
        )
        self._ports["output_w_cmd"].shift_out(w_cmd)
        self._ports["output_nadir"].shift_out(nadir)
        self._ports["output_gimbal_cmd"].shift_out(np.array([0.0]))


class NodePointingSunStuck(Node):
    def __init__(self, **kwargs):
        in_sun_normal = InputPort("in_sun_n", self)
        in_earth_normal = InputPort("in_earth_n", self)

        output_q_cmd = OutputPort("output_q_cmd", self)
        output_w_cmd = OutputPort("output_w_cmd", self)
        output_sp_gimbal = OutputPort("output_sp_gimbal", self)

        ports = {
            in_sun_normal.name: in_sun_normal,
            in_earth_normal.name: in_earth_normal,
            output_q_cmd.name: output_q_cmd,
            output_w_cmd.name: output_w_cmd,
            output_sp_gimbal.name: output_sp_gimbal,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._gimbal = 0  # Set up initial gimbal guess
        self._antenna_body = np.array(self._config["antenna_body"])
        self._antenna_body = self._antenna_body / np.linalg.norm(self._antenna_body)
        self._gimbal_max_angle = np.radians(self._config["gimbal_max_angle"])
        self._gimbal_sol_cache = 0

    def update(self, sim_time: float):
        if sim_time == 0:
            return  # skip first iteration

        sun_n = self._ports["in_sun_n"].read()
        earth_n = self._ports["in_earth_n"].read()

        r_sc2ss, _ = Rotation.align_vectors(
            [sun_n, earth_n],
            [
                np.array([0, 0, 1]),
                self._antenna_body,
            ],
            [1.0, 0.1],
        )

        antenna_ss = r_sc2ss.apply(self._antenna_body)
        gimbal_angle = np.array(
            [
                np.clip(
                    np.arccos(np.dot(antenna_ss, earth_n)),
                    -self._gimbal_max_angle,
                    self._gimbal_max_angle,
                )
            ]
        )

        r_sc2icrs = r_sc2ss * Rotation.from_euler("y", gimbal_angle[0])

        q_cmd = r_sc2icrs.as_quat(canonical=True)

        self._ports["output_q_cmd"].shift_out(
            np.array([q_cmd[3], q_cmd[0], q_cmd[1], q_cmd[2]])
        )
        self._ports["output_w_cmd"].shift_out(np.zeros((3,)))
        self._ports["output_sp_gimbal"].shift_out(-gimbal_angle)


class NodePointingCom(Node):
    def __init__(self, **kwargs):
        in_sun_normal = InputPort("in_sun_n", self)
        in_earth_normal = InputPort("in_earth_n", self)

        output_q_cmd = OutputPort("output_q_cmd", self)
        output_w_cmd = OutputPort("output_w_cmd", self)
        output_sp_gimbal = OutputPort("output_sp_gimbal", self)

        ports = {
            in_sun_normal.name: in_sun_normal,
            in_earth_normal.name: in_earth_normal,
            output_q_cmd.name: output_q_cmd,
            output_w_cmd.name: output_w_cmd,
            output_sp_gimbal.name: output_sp_gimbal,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._gimbal = 0  # Set up initial gimbal guess
        self._antenna_body = np.array(self._config["antenna_body"])
        self._antenna_body = self._antenna_body / np.linalg.norm(self._antenna_body)
        self._gimbal_max_angle = np.radians(self._config["gimbal_max_angle"])

    def update(self, sim_time: float):
        if sim_time == 0:
            return  # skip first iteration

        # fs_state = self._ports["input_fs_state"].read()
        # if fs_state != SimpleFlightState.EARTH_SUN_COM:
        #     return  # skip if in the wrong flight state

        sun_n = self._ports["in_sun_n"].read()
        earth_n = self._ports["in_earth_n"].read()

        r_sc2com, _ = Rotation.align_vectors(
            [earth_n, sun_n], [self._antenna_body, np.array([0, 0, 1])], [1.0, 0.1]
        )

        sp_com = r_sc2com.apply(np.array([0, 0, 1]))
        gimbal_angle = -np.array(
            [
                np.clip(
                    np.arccos(np.dot(sp_com, sun_n)),
                    -self._gimbal_max_angle,
                    self._gimbal_max_angle,
                )
            ]
        )

        q_cmd = r_sc2com.as_quat(canonical=True)

        self._ports["output_q_cmd"].shift_out(
            np.array([q_cmd[3], q_cmd[0], q_cmd[1], q_cmd[2]])
        )
        self._ports["output_w_cmd"].shift_out(np.zeros((3,)))
        self._ports["output_sp_gimbal"].shift_out(gimbal_angle)


class NodePointingCooling(Node):
    def __init__(self, **kwargs):
        in_sun_normal = InputPort("in_sun_n", self)

        output_q_cmd = OutputPort("output_q_cmd", self)
        output_w_cmd = OutputPort("output_w_cmd", self)
        output_sp_gimbal = OutputPort("output_sp_gimbal", self)

        ports = {
            in_sun_normal.name: in_sun_normal,
            output_q_cmd.name: output_q_cmd,
            output_w_cmd.name: output_w_cmd,
            output_sp_gimbal.name: output_sp_gimbal,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._gimbal = 0  # Set up initial gimbal guess
        self._gimbal_max_angle = np.radians(self._config["gimbal_max_angle"])
        self._min_area_normal = self._config["min_area_normal"]

    def update(self, sim_time: float):
        if sim_time == 0:
            return  # skip first iteration

        # fs_state = self._ports["input_fs_state"].read()
        # if fs_state != SimpleFlightState.COOLING:
        #     return  # skip if in the wrong flight state

        sun_n = self._ports["in_sun_n"].read()

        r_sc2cool, _ = Rotation.align_vectors(
            [sun_n, np.array([0, 0, 1])],
            [self._min_area_normal, np.array([0, 0, 1])],
            [1.0, 0.1],
        )

        sp_cool = r_sc2cool.apply(np.array([0, 0, 1]))
        gimbal_angle = np.array(
            [
                np.clip(
                    np.arccos(np.dot(sp_cool, sun_n)),
                    -self._gimbal_max_angle,
                    self._gimbal_max_angle,
                )
            ]
        )

        q_cmd = r_sc2cool.as_quat(canonical=True)

        self._ports["output_q_cmd"].shift_out(
            np.array([q_cmd[3], q_cmd[0], q_cmd[1], q_cmd[2]])
        )
        self._ports["output_w_cmd"].shift_out(np.zeros((3,)))
        self._ports["output_sp_gimbal"].shift_out(gimbal_angle)
