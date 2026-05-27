"""Asteroid gravitational force calculation using pyshtools spherical harmonics."""

from dataclasses import dataclass

import numpy as np
import pyshtools as pysh

from syssim.core import EmptySpec, InputPort, Node, OutputPort, input_port, output_port


# Process-level cache to avoid repeated dataset loading for parallel RL envs.
_GRAVITY_MODEL_CACHE: dict[tuple[str, int | None], object] = {}


@dataclass
class NodeAsteroidGravityInputs:
    """Input ports for asteroid gravity evaluation."""

    position: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Position vector [x, y, z] in asteroid body frame (meters)."""


@dataclass
class NodeAsteroidGravityOutputs:
    """Output ports for asteroid gravity evaluation."""

    gravity_accel: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(3,))
    """Gravitational acceleration vector [ax, ay, az] in asteroid body frame (m/s^2)."""


class NodeAsteroidGravity(Node[NodeAsteroidGravityInputs, NodeAsteroidGravityOutputs, EmptySpec, EmptySpec]):
    """Calculate gravitational acceleration from asteroid spherical harmonics.

    Supported asteroids: Ceres, Vesta, Eros.
    """

    Inputs = NodeAsteroidGravityInputs
    Outputs = NodeAsteroidGravityOutputs

    ASTEROID_DATASETS = {
        "Ceres": "CERES18D",  # JPL 18 degree gravity model
        "Vesta": "VESTA20H",  # JPL 20 degree gravity model
        "Eros": "JGE15A01",  # JPL 15 degree gravity model
    }

    def __init__(self, asteroid: str = "Ceres", lmax: int = None, **kwargs):
        """Initialize asteroid gravity model selection.

        Parameters
        ----------
        asteroid : str, optional
            Asteroid dataset name: ``"Ceres"``, ``"Vesta"``, or ``"Eros"``.
        lmax : int, optional
            Maximum spherical-harmonic degree to load.
        **kwargs
            Additional keyword arguments forwarded to ``Node``.

        Raises
        ------
        ValueError
            If ``asteroid`` is not one of the supported dataset names.
        """
        if asteroid not in self.ASTEROID_DATASETS:
            raise ValueError(
                f"Unknown asteroid '{asteroid}'. Choose from: {list(self.ASTEROID_DATASETS.keys())}"
            )

        self._asteroid = asteroid
        self._lmax = lmax
        self._setup_gravity_model()

        super().__init__(**kwargs)
        self._i = self.i
        self._o = self.o

    def _setup_gravity_model(self):
        """Load spherical harmonic gravity model from pyshtools datasets."""
        cache_key = (self._asteroid, self._lmax)
        cached_model = _GRAVITY_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            self._gravity_model = cached_model
            try:
                self._gravity_model.omega = 0.0
            except Exception:
                pass
            self._max_degree = self._gravity_model.lmax
            return

        if self._lmax is not None:
            if self._asteroid == "Ceres":
                model = pysh.datasets.Ceres.CERES18D(lmax=self._lmax)
            elif self._asteroid == "Vesta":
                model = pysh.datasets.Vesta.VESTA20H(lmax=self._lmax)
            elif self._asteroid == "Eros":
                model = pysh.datasets.Eros.JGE15A01(lmax=self._lmax)
        else:
            if self._asteroid == "Ceres":
                model = pysh.datasets.Ceres.CERES18D()
            elif self._asteroid == "Vesta":
                model = pysh.datasets.Vesta.VESTA20H()
            elif self._asteroid == "Eros":
                model = pysh.datasets.Eros.JGE15A01()

        self._gravity_model = model
        try:
            self._gravity_model.omega = 0.0
        except Exception:
            pass
        _GRAVITY_MODEL_CACHE[cache_key] = model

        self._max_degree = self._gravity_model.lmax

    def initialize(self):
        """Initialize node before simulation."""
        pass

    def update(self, sim_time: float):
        """Compute gravitational acceleration at current position."""
        position = self.i.position.read().value

        if position is None or np.any(np.isnan(position)):
            self.o.gravity_accel.write(np.array([0.0, 0.0, 0.0]), sim_time)
            return

        x, y, z = position
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1.0:
            self.o.gravity_accel.write(np.array([0.0, 0.0, 0.0]), sim_time)
            return

        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x))
        if lon < 0:
            lon += 360.0

        a_r, a_theta, a_phi = pysh.gravmag.MakeGravGridPoint(
            self._gravity_model.coeffs,
            self._gravity_model.gm,
            self._gravity_model.r0,
            r,
            lat,
            lon,
            lmax=self._max_degree,
        )

        theta = np.radians(90.0 - lat)
        phi = np.radians(lon)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        a_x = a_r * sin_theta * cos_phi + a_theta * cos_theta * cos_phi - a_phi * sin_phi
        a_y = a_r * sin_theta * sin_phi + a_theta * cos_theta * sin_phi + a_phi * cos_phi
        a_z = a_r * cos_theta - a_theta * sin_theta

        accel = np.array([a_x, a_y, a_z])
        self.o.gravity_accel.write(accel, sim_time)
