import numpy as np
from astropy.coordinates import (
    get_body_barycentric,
    CartesianRepresentation,
)
from astropy.time import Time
from astropy import units as u
from poliastro.frames.util import _FRAME_MAPPING
from poliastro.frames.enums import Planes

from saas.utility.body import get_poliastro_body

from syssim.core import Node, InputPort, OutputPort


class NodeSunOcclusion(Node):
    def __init__(self, **kwargs):
        """Node which calculates if the Sun is occulded from the spacecraft's current position.

        Ports:
            in_sc_pos_icrs (np.array): input spacecraft position in the ICRS frame relative to central body. 3x1 [m]
            in_start_datetime (datetime): input time and date for the start of the simulation.
            out_is_occluded (bool): true if the sun is occluded. False otherwise
        Configs:
            central_body_name (str): name of the central body for the simulation (e.g., "mars")
        """
        in_sc_pos_icrs = InputPort("in_sc_pos_icrs", self)
        in_start_datetime = InputPort("in_start_datetime", self)

        out_is_occluded = OutputPort("out_is_occluded", self)

        ports = {
            in_sc_pos_icrs.name: in_sc_pos_icrs,
            in_start_datetime.name: in_start_datetime,
            out_is_occluded.name: out_is_occluded,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._central_body = get_poliastro_body(self._config["central_body_name"])
        self._central_body_ICRS = _FRAME_MAPPING[self._central_body][Planes.BODY_FIXED]

    def update(self, sim_time: float):
        self._t0 = Time(self._ports["in_start_datetime"].read())
        sc_pos_cb = CartesianRepresentation(self._ports["in_sc_pos_icrs"].read() * u.m)

        t = self._t0 + (sim_time * u.s)
        sun_pos_ICRS = get_body_barycentric("sun", t)

        cb_pos_ICRS = get_body_barycentric(self._config["central_body_name"], t)
        sc_pos_ICRS = cb_pos_ICRS + sc_pos_cb

        sc_sun_ICRS = sun_pos_ICRS - sc_pos_ICRS

        sun_sep_angle = np.arccos(
            (sc_sun_ICRS / sc_sun_ICRS.norm())
            .dot((-sc_pos_cb / sc_pos_cb.norm()))
            .value
        )

        apparent_size_of_planet = np.arctan(
            self._central_body.R.si.value / sc_pos_cb.norm().si.value
        )

        self._ports["out_is_occluded"].shift_out(
            sun_sep_angle < apparent_size_of_planet
        )


class NodeEarthOcclusion(Node):
    def __init__(self, **kwargs):
        """Node for calculating whether the Earth is occluded by the central body from the spacecraft's current position.

        Ports:
            in_sc_pos_icrs (np.array): input spacecraft position in the ICRS frame relative to central body. 3x1 [m]
            in_start_datetime (datetime): input time and date for the start of the simulation.
            out_is_occluded (bool): true if the Earth is occluded. False otherwise.
        Configs:
            central_body_name (str): name of the central body for the simulation (e.g., "mars")
        """

        in_sc_pos_icrs = InputPort("in_sc_pos_icrs", self)
        in_start_datetime = InputPort("in_start_datetime", self)

        out_is_occluded = OutputPort("out_is_occluded", self)

        ports = {
            in_sc_pos_icrs.name: in_sc_pos_icrs,
            in_start_datetime.name: in_start_datetime,
            out_is_occluded.name: out_is_occluded,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._central_body = get_poliastro_body(self._config["central_body_name"])
        self._central_body_ICRS = _FRAME_MAPPING[self._central_body][Planes.BODY_FIXED]

    def update(self, sim_time: float):
        self._t0 = Time(self._ports["in_start_datetime"].read())
        sc_pos_cb = CartesianRepresentation(self._ports["in_sc_pos_icrs"].read() * u.m)

        t = self._t0 + (sim_time * u.s)
        earth_pos_ICRS = get_body_barycentric("earth", t)
        cb_pos_ICRS = get_body_barycentric(self._config["central_body_name"], t)
        sc_pos_ICRS = cb_pos_ICRS + sc_pos_cb

        sc_earth_icrs = earth_pos_ICRS - sc_pos_ICRS

        earth_sep_angle = np.arccos(
            (sc_earth_icrs / sc_earth_icrs.norm())
            .dot((-sc_pos_cb / sc_pos_cb.norm()))
            .value
        )

        apparent_size_of_planet = np.arctan(
            self._central_body.R.si.value / sc_pos_cb.norm().si.value
        )

        self._ports["out_is_occluded"].shift_out(
            earth_sep_angle < apparent_size_of_planet
        )
