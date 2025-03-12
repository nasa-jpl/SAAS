import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

from syssim.core import Node, InputPort

from saas.utility.plotting import add_fault_vbars, add_fault_detect_vline


class NodeVizAttitudeErrorInputs(NamedTuple):
    in_q_e: InputPort
    in_w_e: InputPort

class NodeVizAttitudeError(Node):
    def __init__(self, **kwargs):
        self._i = NodeVizAttitudeErrorInputs(
            InputPort("in_q_e", self),
            InputPort("in_w_e", self)
        )

        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        self._t = []
        self._q_e = []
        self._w_e = []

    def update(self, sim_time: float):
        self._t.append(sim_time / 3600)
        self._q_e.append(self._i.in_q_e.read())
        self._w_e.append(self._i.in_w_e.read())

    def finalize(self):
        title = self._config["title"]
        if self._config["dark_mode"]:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        _, (q_ax, w_ax) = plt.subplots(2, 1, sharex=True)

        q_ax.plot(self._t, self._q_e, label=["quat_err_x", "quat_err_y", "quat_err_z"])
        w_ax.plot(self._t, self._w_e, label=["w_err_x", "w_err_y", "w_err_z"])
        q_ax.plot(self._t, [np.linalg.norm(v) for v in self._q_e], label="norm")
        w_ax.plot(self._t, [np.linalg.norm(v) for v in self._w_e], label="norm")

        q_ax.set_ylabel("error")
        w_ax.set_ylabel("error (rad/s)")

        q_ax.set_title(title)

        plt.xlim(self._t[0], self._t[-1])

        q_ax.legend()
        w_ax.legend()

        if self._config["show_faults"]:
            faults = self._system.get_faults()
            fault_detections = self._system.get_fault_detections()
            add_fault_vbars(faults, q_ax)
            add_fault_vbars(faults, w_ax)
            add_fault_detect_vline(fault_detections, q_ax)
            add_fault_detect_vline(fault_detections, w_ax)

        if self._config["show"]:
            plt.show()

        if self._config["save"] and self._system.get_output_dir() is not None:
            plt.savefig(
                self._system.get_output_dir() + f"{self.name}.png".replace(" ", "_")
            )

        plt.close(plt.gcf())

    @property
    def i(self):
        return self._i

    
    @property
    def o(self):
        return ()