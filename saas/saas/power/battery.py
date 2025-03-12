from typing import NamedTuple
import numpy as np
from scipy.integrate import solve_ivp

from syssim.core import NodeDifferential, InputPort, OutputPort
from syssim.core.port import InputPort, OutputPort

import matplotlib.pyplot as plt

class NodeBatteryInputs(NamedTuple):
    in_current: InputPort
    in_current_draw: InputPort

class NodeBatteryOutputs(NamedTuple):
    out_soc: OutputPort

class NodeSimpleBattery(NodeDifferential):
    def __init__(self, x0, **kwargs):
        self._i = NodeBatteryInputs(
            InputPort("in_current", self),
            InputPort("in_current_draw", self)
        )
        self._o = NodeBatteryOutputs(
            OutputPort("out_soc", self)
        )

        super().__init__(x0, self._i, self._o, **kwargs)

    def initialize(self):
        self._eff = self._config["efficiency"]
        self._ts = []
        self._inp = []
        self._t = 0

    def update(self, sim_time: float):
        i_inp = self._i.in_current.read()
        i_out = self._i.in_current_draw.read()

        if i_inp == None:
            i_inp = 0
        if i_out == None:
            i_out = 0

        self._ts.append(sim_time)
        self._inp.append(i_inp)

        dt = sim_time - self._t
        self._x += (
            (self._eff * i_inp - (1 / self._eff) * i_out)
            * dt
            / 3600
            / self._config["capacity_ah"]
        )
        self._x = np.clip(self._x, 0.0, 1.0)

        self._t = sim_time

        self._o.out_soc.shift_out(self._x)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o

