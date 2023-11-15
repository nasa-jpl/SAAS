import numpy as np
from astropy.coordinates import (
    get_body_barycentric,
    CartesianRepresentation,
)
from astropy.time import Time
from astropy import units as u

from syssim.core import Node, OutputPort, InputPort


class NodeSunPosition(Node):
    def __init__(
        self,
        **kwargs,
    ):
        """Node for getting the position of the Sun relative to a spacecraft given a time and date. Solutions given in the ICRS frame relative to a given central body position. Also provides the solar constant at the spacecrafts current position.

        Ports:
            in_sc_pos_icrs (np.array): input spacecraft position in the ICRS frame relative to central body. 3x1 [m]
            in_start_datetime (datetime): input time and date for the start of the simulation.
            out_sun_pos_icrs (np.array): output position of the Sun relative to the spacecraft in ICRS. 3x1 [m]
            out_sun_unit_icrs (np.array): ouptut unit vector from spaccraft to Sun in ICRS. 3x1 [m]
            out_solar_constant (float): the solar constant at the spacecrafts current position [W m^-2].
        Configs:
            central_body_name (str): name of the central body for the simulation (e.g., "mars")
        """
        in_sc_pos_icrs = InputPort("in_sc_pos_icrs", self)
        in_start_datetime = InputPort("in_start_datetime", self)

        out_sun_pos_icrs = OutputPort("out_sun_pos_icrs", self)
        out_sun_unit_icrs = OutputPort("out_sun_unit_icrs", self)
        out_solar_constant = OutputPort("out_solar_constant", self)

        ports = {
            in_sc_pos_icrs.name: in_sc_pos_icrs,
            in_start_datetime.name: in_start_datetime,
            out_sun_pos_icrs.name: out_sun_pos_icrs,
            out_sun_unit_icrs.name: out_sun_unit_icrs,
            out_solar_constant.name: out_solar_constant,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._solar_constant = 1.361e3  # W m^-2

    def update(self, sim_time: float):
        self._t0 = Time(self._ports["in_start_datetime"].read())

        sc_pos_cb = CartesianRepresentation(self._ports["in_sc_pos_icrs"].read() * u.m)

        t = self._t0 + (sim_time * u.s)
        sun_pos_ICRS = get_body_barycentric("sun", t)

        cb_pos_ICRS = get_body_barycentric(self._config["central_body_name"], t)
        sc_pos_ICRS = cb_pos_ICRS + sc_pos_cb

        sc_sun_ICRS = sun_pos_ICRS - sc_pos_ICRS
        sc_sun_au = np.linalg.norm(sc_sun_ICRS.xyz.to(u.au).value)

        self._ports["out_sun_pos_icrs"].shift_out(sc_sun_ICRS.get_xyz().si.value)
        self._ports["out_sun_unit_icrs"].shift_out(
            sc_sun_ICRS.get_xyz().si.value
            / np.linalg.norm(sc_sun_ICRS.get_xyz().si.value)
        )
        self._ports["out_solar_constant"].shift_out(
            self._solar_constant / sc_sun_au**2
        )
