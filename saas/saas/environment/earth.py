import numpy as np
from astropy.coordinates import (
    get_body_barycentric,
    CartesianRepresentation,
)
from astropy.time import Time
from astropy import units as u
from typing import NamedTuple

from syssim.core import Node, OutputPort, InputPort


class NodeEarthPositionInputs(NamedTuple):
    in_sc_pos_icrs: InputPort
    in_start_datetime: InputPort

class NodeEarthPositionOutputs(NamedTuple):
    out_earth_pos_icrs: OutputPort
    out_earth_unit_icrs: OutputPort

class NodeEarthPosition(Node):
    def __init__(
        self,
        **kwargs,
    ):
        """Node for getting the position of the Earth relative to a spacecraft given a time and date. Solutions given in the ICRS frame relative to a given central body position.

        Ports:
            in_sc_pos_icrs (np.array): input spacecraft position in the ICRS frame relative to central body. 3x1 [m]
            in_start_datetime (datetime): input time and date for the start of the simulation.
            out_earth_pos_icrs (np.array): output position of the earth relative to the spacecraft in ICRS. 3x1 [m]
            out_earth_unit_icrs (np.array): ouptut unit vector from spaccraft to Earth in ICRS. 3x1 [m]
        Configs:
            central_body_name (str): name of the central body for the simulation (e.g., "mars")
        """
        self._i = NodeEarthPositionInputs(
            InputPort("in_sc_pos_icrs", self),
            InputPort("in_start_datetime", self)
        )
        self._o = NodeEarthPositionOutputs(
            OutputPort("out_earth_pos_icrs", self),
            OutputPort("out_earth_unit_icrs", self)
        )

        super().__init__(self._i, self._o, **kwargs)

    def update(self, sim_time: float):
        self._t0 = Time(self._i.in_start_datetime.read())

        sc_pos_cb = CartesianRepresentation(self._i.in_sc_pos_icrs.read() * u.m)

        t = self._t0 + (sim_time * u.s)
        earth_pos_ICRS = get_body_barycentric("earth", t)
        cb_pos_ICRS = get_body_barycentric(self._config["central_body_name"], t)
        sc_pos_ICRS = cb_pos_ICRS + sc_pos_cb

        sc_earth_icrs = earth_pos_ICRS - sc_pos_ICRS

        self._o.out_earth_pos_icrs.shift_out(sc_earth_icrs.get_xyz().si.value)
        self._o.out_earth_unit_icrs.shift_out(
            sc_earth_icrs.get_xyz().si.value
            / np.linalg.norm(sc_earth_icrs.get_xyz().si.value)
        )
    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o