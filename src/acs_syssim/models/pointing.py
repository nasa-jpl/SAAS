import numpy as np
from typing import NamedTuple, Dict, Tuple
from syssim import Node, OutputPort
from scipy.spatial.transform import Rotation as R

"""The pointing node produces the command signals for a nadir pointing science mode and a safe-mode attitude.
The nadir pointing mode is determined from a circular orbit at the equator of a body defined by a gravitational parameter, planetary radius, and orbit altitude.
"""

class PointingOutputs(NamedTuple):
    w_nadir: OutputPort
    q_nadir: OutputPort
    w_safe: OutputPort
    q_safe: OutputPort

class Pointing(Node):
    
    def __init__(self, **kwargs):
        self._o = PointingOutputs(
            OutputPort("w_nadir", self),
            OutputPort("q_nadir", self),
            OutputPort("w_safe", self),
            OutputPort("q_safe", self),
        )
        # Attributes to expose last written outputs for simple unit testing
        self.latest_w_nadir = None
        self.latest_q_nadir = None
        self.latest_w_safe = None
        self.latest_q_safe = None
        super().__init__((), self._o, **kwargs)
    
    def initialize(self):
        self._mu = float(self._config.get("mu", 3.986e14))  # Earth's gravitational parameter in m^3/s^2
        self._R_planet = float(self._config.get("R_planet", 6371e3))  # Earth's radius in meters
        self._altitude = float(self._config.get("altitude", 500e3))  # Orbit altitude in meters
        self._orbit_radius = self._R_planet + self._altitude

        self._orbit_velocity = np.sqrt(self._mu / self._orbit_radius)

        # Calculate an orbital angular rate for a circular orbit
        self._nadir_angular_rate = self._orbit_velocity / self._orbit_radius
        self._w_nadir = np.array([0.0, self._nadir_angular_rate, 0.0])

        # Safe mode is assumed to be non-rotating in inertial frame
        self._w_safe = np.array([0.0, 0.0, 0.0])

    def update(self, t: float):
        orbit_angle = self._nadir_angular_rate * t

        xy_position_eci = self._orbit_radius * np.array([np.cos(orbit_angle), np.sin(orbit_angle), 0.0])

        z_body_eci = -xy_position_eci / np.linalg.norm(xy_position_eci)

        nadir_rotation = R.align_vectors([[0, 0, 1], [1, 0, 0]], [z_body_eci, [0, 0, 1]])[0].as_quat(canonical=True, scalar_first=True)
        safe_rotation = R.from_quat([1, 0, 0, 0]).as_quat(canonical=True, scalar_first=True)
        self._o.w_nadir.shift_out(self._w_nadir)
        self._o.q_nadir.shift_out(nadir_rotation)
        self._o.w_safe.shift_out(self._w_safe)
        self._o.q_safe.shift_out(safe_rotation)

    @property
    def o(self):
        return self._o


if __name__ == "__main__":
    from syssim import NodeSystem
    from syssim.nodes.viz import NodeScope

    system = NodeSystem()
    pointing = Pointing(name="pointing")
    scope_q_nadir = NodeScope(name="scope_q_nadir")
    scope_w_nadir = NodeScope(name="scope_w_nadir")
    scope_q_safe = NodeScope(name="scope_q_safe")
    scope_w_safe = NodeScope(name="scope_w_safe")
    [system.add_node(n) for n in [pointing, scope_q_nadir, scope_w_nadir, scope_q_safe, scope_w_safe]]

    pointing.o.q_nadir >> scope_q_nadir.i.scope
    pointing.o.w_nadir >> scope_w_nadir.i.scope
    pointing.o.q_safe >> scope_q_safe.i.scope
    pointing.o.w_safe >> scope_w_safe.i.scope
    system.simulate(3600, dt=1.0)  # simulate for one hour with 1 second timestep