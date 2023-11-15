import numpy as np
from scipy.integrate import solve_ivp

from syssim.core import NodeDifferential, InputPort, OutputPort
from syssim.core.port import InputPort, OutputPort

import matplotlib.pyplot as plt

class NodeSimpleBattery(NodeDifferential):
    def __init__(self, x0, **kwargs):
        in_current = InputPort("in_current", self)
        in_draw = InputPort("in_current_draw", self)

        out_soc = OutputPort("out_soc", self)

        ports = {
            in_current.name: in_current,
            in_draw.name: in_draw,
            out_soc.name: out_soc,
        }


        super().__init__(x0, ports, **kwargs)

    def initialize(self):
        self._eff = self._config["efficiency"]
        self._ts = []
        self._inp = []
        self._t = 0


    # def finalize(self):
    #     plt.plot(self._ts, self._inp)
    #     plt.show()

    def update(self, sim_time: float):
        i_inp = self._ports["in_current"].read()
        i_out = self._ports["in_current_draw"].read()

        if i_inp == None:
            i_inp = 0
        if i_out == None:
            i_out = 0

        self._ts.append(sim_time)
        self._inp.append(i_inp)

        dt = sim_time - self._t
        # print(
        #     f"{(i_inp - i_out) * dt / 3600 / self._config['capacity_ah']} | {self._x}"
        # )
        # print(f'{i_inp}')
        self._x += (
            (self._eff * i_inp - (1 / self._eff) * i_out)
            * dt
            / 3600
            / self._config["capacity_ah"]
        )
        self._x = np.clip(self._x, 0.0, 1.0)

        self._t = sim_time

        self._ports["out_soc"].shift_out(self._x)
