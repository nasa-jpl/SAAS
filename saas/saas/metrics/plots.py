from typing import Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from syssim.core import Node, InputPort
from saas.utility.plotting import add_fault_vbars, add_fault_detect_vline
from syssim.core.port import InputPort


class NodeMetricPointingError(Node):
    def __init__(self, **kwargs):
        in_cmd_vec = InputPort("input_cmd_vec", self)
        in_meas_vec = InputPort("input_meas_vec", self)
        in_start_datetime = InputPort("in_start_datetime", self)

        ports = {
            in_cmd_vec.name: in_cmd_vec,
            in_meas_vec.name: in_meas_vec,
            in_start_datetime.name: in_start_datetime,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._t = []
        self._error = []

    def update(self, sim_time: float):
        cmd_vec = self._ports["input_cmd_vec"].read()
        meas_vec = self._ports["input_meas_vec"].read()

        # Angle between the vectors
        error = np.arccos(
            np.dot(cmd_vec, meas_vec)
            / (np.linalg.norm(cmd_vec) * np.linalg.norm(meas_vec))
        )
        self._t.append(sim_time / 3600)
        self._error.append(error)

    def finalize(self):
        t0 = self._ports["in_start_datetime"].read()

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


class NodeMetricSoCTemperature(Node):
    def __init__(self, **kwargs):
        in_soc = InputPort("in_soc", self)
        in_temp = InputPort("in_temp", self)
        in_start_datetime = InputPort("in_start_datetime", self)

        ports = {
            in_soc.name: in_soc,
            in_temp.name: in_temp,
            in_start_datetime.name: in_start_datetime,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._t = []
        self._soc = []
        self._k = []

    def update(self, sim_time: float):
        self._t.append(sim_time / 3600)
        self._soc.append(self._ports["in_soc"].read())
        self._k.append(self._ports["in_temp"].read())

    def finalize(self):
        t0 = self._ports["in_start_datetime"].read()

        title = self._config["title"]

        _, soc_ax = plt.subplots()
        k_ax = soc_ax.twinx()

        soc_ax.set_title(title)
        soc_ax.set_ylabel("State of Charge (%)")
        k_ax.set_ylabel("S/C Temperature (K)")
        soc_ax.set_xlabel(f"t + {t0.strftime('%Y-%m-%d %H:%M:%S')} (hr)")
        plt.xlim(self._t[0], self._t[-1])

        # avg_error = np.average(self._error)
        # max_error = np.max(self._error)

        # plt.hlines([avg_error], [0], [self._t[-1]], ["g"], label="avg")

        # plt.hlines([max_error], [0], [self._t[-1]], ["r"], label="max")

        line1 = soc_ax.plot(self._t, self._soc, label="SoC", color="green")
        line2 = k_ax.plot(self._t, self._k, label="S/C Temp", color="red")
        artists = [line1[0], line2[0]]

        if self._config["show_faults"]:
            faults = self._system.get_faults()
            fault_detections = self._system.get_fault_detections()
            artists += add_fault_vbars(faults, soc_ax)
            artists += add_fault_detect_vline(fault_detections, soc_ax)

        labels = [line.get_label() for line in artists]
        soc_ax.legend(artists, labels, loc="upper left")

        if self._config["show"]:
            plt.show()

        if self._config["save"] and self._system.get_output_dir() is not None:
            plt.savefig(
                self._system.get_output_dir() + f"{self.name}.png".replace(" ", "_")
            )

        plt.close(plt.gcf())


class NodeMetricScience(Node):
    def __init__(self, **kwargs):
        in_sci = InputPort("in_science", self)
        in_sci_dl = InputPort("in_science_dl", self)
        in_start_datetime = InputPort("in_start_datetime", self)

        ports = {
            in_sci.name: in_sci,
            in_sci_dl.name: in_sci_dl,
            in_start_datetime.name: in_start_datetime,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._t = []
        self._s = []
        self._d = []

    def update(self, sim_time: float):
        self._t.append(sim_time / 3600)
        self._s.append(self._ports["in_science"].read())
        self._d.append(self._ports["in_science_dl"].read())

    def finalize(self):
        t0 = self._ports["in_start_datetime"].read()

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
        s_ax.legend(lines, labels, loc="upper left")

        if self._config["show"]:
            plt.show()

        if self._config["save"] and self._system.get_output_dir() is not None:
            plt.savefig(
                self._system.get_output_dir() + f"{self.name}.png".replace(" ", "_")
            )

        plt.close(plt.gcf())
