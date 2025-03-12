import numpy as np
from astropy.coordinates import (
    get_body_barycentric,
    CartesianRepresentation,
)
from astropy.time import Time
from astropy import units as u
from typing import NamedTuple

from syssim import Node, OutputPort, InputPort

class NodeSunPositionInputs(NamedTuple):
    in_sc_pos_icrs: InputPort
    in_start_datetime: InputPort

class NodeSunPositionOutputs(NamedTuple):
    out_sun_pos_icrs: OutputPort
    out_sun_unit_icrs: OutputPort
    out_solar_constant: OutputPort

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
        self._i = NodeSunPositionInputs(
            InputPort("in_sc_pos_icrs", self),
            InputPort("in_start_datetime", self)
        )
        self._o = NodeSunPositionOutputs(
            OutputPort("out_sun_pos_icrs", self),
            OutputPort("out_sun_unit_icrs", self),
            OutputPort("out_solar_constant", self)
        )

        super().__init__(self._i, self._o, **kwargs)

    def initialize(self):
        self._solar_constant = 1.361e3  # W m^-2

    def update(self, sim_time: float):
        self._t0 = Time(self._i.in_start_datetime.read())

        sc_pos_cb = CartesianRepresentation(self._i.in_sc_pos_icrs.read() * u.m)

        t = self._t0 + (sim_time * u.s)
        sun_pos_ICRS = get_body_barycentric("sun", t)

        cb_pos_ICRS = get_body_barycentric(self._config["central_body_name"], t)
        sc_pos_ICRS = cb_pos_ICRS + sc_pos_cb

        sc_sun_ICRS = sun_pos_ICRS - sc_pos_ICRS
        sc_sun_au = np.linalg.norm(sc_sun_ICRS.xyz.to(u.au).value)

        self._o.out_sun_pos_icrs.shift_out(sc_sun_ICRS.get_xyz().si.value)
        self._o.out_sun_unit_icrs.shift_out(
            sc_sun_ICRS.get_xyz().si.value
            / np.linalg.norm(sc_sun_ICRS.get_xyz().si.value)
        )
        self._o.out_solar_constant.shift_out(
            self._solar_constant / sc_sun_au**2
        )
    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o