#! python
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

from syssim.core import Node, NodeSystem, InputPort, OutputPort
from syssim.nodes.dynamics import NodeStateSpace


class NodeSinSourceOutputs(NamedTuple):
    y: OutputPort


class NodeSinSource(Node):
    def __init__(self, omega: float = 1.0, amplitude: float = 1.0, **kwargs):
        self._omega = omega
        self._amp = amplitude
        self._o = NodeSinSourceOutputs(OutputPort("output_y", self))
        super().__init__((), self._o, **kwargs)

    def update(self, sim_time: float):
        y = np.array([self._amp * np.sin(self._omega * sim_time)])
        self._o.y.shift_out(y, sim_time)

    @property
    def o(self):
        return self._o


class IntegratorScopeInputs(NamedTuple):
    y1: InputPort
    y2: InputPort
    y3: InputPort


class IntegratorScope(Node):
    def __init__(self, **kwargs):
        self._i = IntegratorScopeInputs(
            InputPort("input_y1", self),
            InputPort("input_y2", self),
            InputPort("input_y3", self),
        )
        self._t = []
        self._y1 = []
        self._y2 = []
        self._y3 = []
        self._age_1 = []
        self._age_2 = []
        self._age_3 = []
        super().__init__(self._i, (), **kwargs)

    def update(self, sim_time: float):
        y1, t1 = self._i.y1.read_with_time()
        y2, t2 = self._i.y2.read_with_time()
        y3, t3 = self._i.y3.read_with_time()

        self._t.append(sim_time)
        self._y1.append(float(y1[0]))
        self._y2.append(float(y2[0]))
        self._y3.append(float(y3[0]))
        self._age_1.append(sim_time - t1)
        self._age_2.append(sim_time - t2)
        self._age_3.append(sim_time - t3)

    def finalize(self, fault_history=None):
        fig, (ax_out, ax_age) = plt.subplots(2, 1, sharex=True)

        ax_out.plot(self._t, self._y1, label="Integrator 1")
        ax_out.plot(self._t, self._y2, label="Integrator 2")
        ax_out.plot(self._t, self._y3, label="Integrator 3")
        ax_out.set_ylabel("Output")
        ax_out.set_title("Chained Integrators: Output Delay Through the Chain")
        ax_out.grid(True)
        ax_out.legend()

        ax_age.plot(self._t, self._age_1, label="Age y1")
        ax_age.plot(self._t, self._age_2, label="Age y2")
        ax_age.plot(self._t, self._age_3, label="Age y3")
        ax_age.set_xlabel("Simulation Time (s)")
        ax_age.set_ylabel("Sample Age (s)")
        ax_age.grid(True)
        ax_age.legend()

        plt.tight_layout()
        plt.show()

    @property
    def i(self):
        return self._i


def make_integrator(name: str, sample_period: float):
    return NodeStateSpace(
        a=np.array([[0.0]]),
        b=np.array([[1.0]]),
        c=np.array([[1.0]]),
        x0=np.array([0.0]),
        sample_period=sample_period,
        name=name,
    )


if __name__ == "__main__":
    dt = 0.01
    t_final = 20.0

    source = NodeSinSource(omega=1.0, amplitude=1.0, sample_period=dt, name="sin-source")
    int_1 = make_integrator("integrator-1", dt)
    int_2 = make_integrator("integrator-2", dt)
    int_3 = make_integrator("integrator-3", dt)
    scope = IntegratorScope(sample_period=dt, name="integrator-scope")

    system = NodeSystem()
    system.add_node(source)
    system.add_node(int_1)
    system.add_node(int_2)
    system.add_node(int_3)
    system.add_node(scope)

    source.o.y >> int_1.i.u
    int_1.o.y >> int_2.i.u
    int_2.o.y >> int_3.i.u

    int_1.o.y >> scope.i.y1
    int_2.o.y >> scope.i.y2
    int_3.o.y >> scope.i.y3

    system.simulate(t_f=t_final, dt=dt, sim_name="integrator-chain-delay")
