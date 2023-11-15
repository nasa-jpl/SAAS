import numpy as np
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
from syssim.core import NodeDifferential, InputPort, OutputPort


# TODO Might consider renaming this module to thermal if that seems more appropriate.
# TODO Might also consider driving inertia moment calcs through here instead of through the config
class NodeSimpleRectanglePrismSCBus(NodeDifferential):
    def __init__(self, x0, **kwargs):
        in_sun_unit = InputPort("in_sun_unit", self)
        in_solar_constant = InputPort("in_solar_constant", self)
        in_q_sc2eci = InputPort("in_q_sc2eci", self)

        out_sc_temp = OutputPort("out_sc_temp", self)

        ports = {
            in_sun_unit.name: in_sun_unit,
            in_solar_constant.name: in_solar_constant,
            in_q_sc2eci.name: in_q_sc2eci,
            out_sc_temp.name: out_sc_temp,
        }

        super().__init__(x0, ports, **kwargs)

    def initialize(self):
        self._t = 0

        l = self._config["length"]
        w = self._config["width"]
        h = self._config["height"]

        ax_sc = np.array([[w * h, 0, 0]]).T
        ay_sc = np.array([[0, l * h, 0]]).T
        az_sc = np.array([[0, 0, l * w]]).T

        self._a_sc = np.hstack((ax_sc, ay_sc, az_sc, -ax_sc, -ay_sc, -az_sc))
        self._a_total = 2 * (l * w + l * h + w * h)
        self._c = self._config["heat_cap_spec"]
        self._eps = self._config["emmisivity"]
        self._sig = 5.670374419e-8

    def update(self, sim_time: float):
        q_sc2eci = self._ports["in_q_sc2eci"].read()
        sun_unit = self._ports["in_sun_unit"].read()
        solar_constant = self._ports["in_solar_constant"].read()

        if np.any(q_sc2eci) == None:
            q_sc2eci = Rotation.identity()
        if np.any(sun_unit) == None:
            sun_unit = np.array([1, 0, 0])
        if solar_constant == None:
            solar_constant = 1.361e3  # W m^-2

        r_sc2eci = Rotation.from_quat(
            [q_sc2eci[1], q_sc2eci[2], q_sc2eci[3], q_sc2eci[0]]
        )
        def integrand(t, x):
            return self._dynamics(x, r_sc2eci, sun_unit * solar_constant)

        sol = solve_ivp(integrand, (self._t, sim_time), self._x)

        self._t = sim_time
        self._x = sol.y[:, -1]

        self._ports["out_sc_temp"].shift_out(self._x)

    def _dynamics(self, x, r_sc2eci, sun_flux_eci):
        a_eci = r_sc2eci.apply(self._a_sc.T)

        # Compute the incident flux using the reveresed area vectors
        flux_sc = sun_flux_eci @ -a_eci.T
        flux_total = np.sum(np.clip(flux_sc, 0, None))  # Negative flux not counted

        alpha = self._config["absorbtivity"]
        m = self._config["mass"]

        p_absorbed = alpha * flux_total
        p_emmited = self._eps * self._sig * self._a_total * x**4
        return (p_absorbed - p_emmited) / (self._c * m)
