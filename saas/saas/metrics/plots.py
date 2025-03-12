from typing import Dict, Union, NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from syssim.core import Node, InputPort
from saas.utility.plotting import add_fault_vbars, add_fault_detect_vline
from syssim.core.port import InputPort


class NodeMetricPointingErrorInputs(NamedTuple):
    in_cmd_vec: InputPort
    in_meas_vec: InputPort
    in_start_datetime: InputPort

class NodeMetricPointingError(Node):
    def __init__(self, **kwargs):
        self._i = NodeMetricPointingErrorInputs(
            InputPort("input_cmd_vec", self),
            InputPort("input_meas_vec", self),
            InputPort("in_start_datetime", self)
        )

        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        self._t = []
        self._error = []

    def update(self, sim_time: float):
        cmd_vec = self._i.in_cmd_vec.read()
        meas_vec = self._i.in_meas_vec.read()

        # Angle between the vectors
        error = np.arccos(
            np.dot(cmd_vec, meas_vec)
            / (np.linalg.norm(cmd_vec) * np.linalg.norm(meas_vec))
        )
        self._t.append(sim_time / 3600)
        self._error.append(error)

    def finalize(self):
        if self._config["dark_mode"]:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        t0 = self._i.in_start_datetime.read()

        title = self._config["title"]
        ylabel = self._config["ylabel"]

        plt.figure()
        plt.plot(self._t, self._error, label="error")
        plt.title(title)
        plt.xlabel(f"t + {t0.strftime('%Y-%m-%d %H:%M:%S')} (hr)")
        plt.ylabel(ylabel)
        plt.xlim(self._t[0], self._t[-1])

        avg_error = np.average(self._error)
        max_error = np.max(self._error)

        plt.hlines([avg_error], [0], [self._t[-1]], ["g"], label="avg")

        plt.hlines([max_error], [0], [self._t[-1]], ["r"], label="max")

        if self._config["show_faults"]:
            faults = self._system.get_faults()
            fault_detections = self._system.get_fault_detections()
            add_fault_detect_vline(fault_detections, plt.gca())
            add_fault_vbars(faults, plt.gca())

        plt.legend()

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
        return self._o


class NodeMetricSoCTemperatureInputs(NamedTuple):
    in_soc: InputPort
    in_temp: InputPort
    in_start_datetime: InputPort

class NodeMetricSoCTemperature(Node):
    def __init__(self, **kwargs):
        self._i = NodeMetricSoCTemperatureInputs(
            InputPort("in_soc", self),
            InputPort("in_temp", self),
            InputPort("in_start_datetime", self)
        )

        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        self._t = []
        self._soc = []
        self._k = []

    def update(self, sim_time: float):
        self._t.append(sim_time / 3600)
        self._soc.append(self._i.in_soc.read())
        self._k.append(self._i.in_temp.read())

    def finalize(self):
        if self._config["dark_mode"]:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        t0 = self._i.in_start_datetime.read()

        title = self._config["title"]

        _, soc_ax = plt.subplots()
        k_ax = soc_ax.twinx()

        soc_ax.set_title(title)
        soc_ax.set_ylabel("State of Charge (%)")
        k_ax.set_ylabel("S/C Temperature (K)")
        soc_ax.set_xlabel(f"t + {t0.strftime('%Y-%m-%d %H:%M:%S')} (hr)")
        plt.xlim(self._t[0], self._t[-1])

        line1 = soc_ax.plot(self._t, self._soc, label="SoC", color="green")
        line2 = k_ax.plot(self._t, self._k, label="S/C Temp", color="red")
        artists = [line1[0], line2[0]]

        if self._config["show_faults"]:
            faults = self._system.get_faults()
            fault_detections = self._system.get_fault_detections()
            artists += add_fault_vbars(faults, soc_ax)
            artists += add_fault_detect_vline(fault_detections, soc_ax)

        labels = [line.get_label() for line in artists]
        k_ax.legend(artists, labels, loc="upper left").set_zorder(10.0)

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
        return self._o


class NodeMetricScienceInputs(NamedTuple):
    in_sci: InputPort
    in_sci_dl: InputPort
    in_start_datetime: InputPort

class NodeMetricScience(Node):
    def __init__(self, **kwargs):
        self._i = NodeMetricScienceInputs(
            InputPort("in_science", self),
            InputPort("in_science_dl", self),
            InputPort("in_start_datetime", self)
        )

        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        self._t = []
        self._s = []
        self._d = []

    def update(self, sim_time: float):
        self._t.append(sim_time / 3600)
        self._s.append(self._i.in_sci.read())
        self._d.append(self._i.in_sci_dl.read())

    def finalize(self):
        if self._config["dark_mode"]:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
            
        t0 = self._i.in_start_datetime.read()

        title = self._config["title"]

        _, s_ax = plt.subplots()
        d_ax = s_ax.twinx()

        s_ax.set_title(title)
        s_ax.set_ylabel("Science Onboard S/C (Gb)")
        d_ax.set_ylabel("Science Downlinked (Gb)")
        s_ax.set_xlabel(f"t + {t0.strftime('%Y-%m-%d %H:%M:%S')} (hr)")
        plt.xlim(self._t[0], self._t[-1])

        line1 = s_ax.plot(self._t, self._s, label="Science Oboard", color="blue")
        line2 = d_ax.plot(self._t, self._d, label="Science Downlinked", color="green")

        if self._config["show_faults"]:
            faults = self._system.get_faults()
            fault_detections = self._system.get_fault_detections()
            add_fault_vbars(faults, s_ax)
            add_fault_detect_vline(fault_detections, s_ax)

        lines = [line1[0], line2[0]]
        labels = [line.get_label() for line in lines]
        d_ax.legend(lines, labels, loc="upper left").set_zorder(10.0)

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
        return self._o
