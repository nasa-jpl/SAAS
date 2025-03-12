from enum import Enum
from collections import deque
from typing import Dict, Union, NamedTuple

import numpy as np

from syssim import Node, NodeDifferential, InputPort, OutputPort

import matplotlib.pyplot as plt


class SimpleFlightState(str, Enum):
    SCIENCE = "SCIENCE"
    SUN_STUCK = "SUN_STUCK"
    EARTH_SUN_COM = "EARTH_SUN_COM"
    COOLING = "COOLING"
    SAFE = "SAFE"


class SafeModeTask(str, Enum):
    CHARGE = "charge"
    COOL = "cool"
    DL = "downlink"


class NodeSimpleFlightSoftwareInputs(NamedTuple):
    in_point_err: InputPort
    in_q_cmd_nadir: InputPort
    in_q_cmd_ss: InputPort
    in_q_cmd_com: InputPort
    in_q_cmd_cool: InputPort
    in_w_cmd_nadir: InputPort
    in_w_cmd_ss: InputPort
    in_w_cmd_com: InputPort
    in_w_cmd_cool: InputPort
    in_w_gimbal_nadir: InputPort
    in_w_gimbal_ss: InputPort
    in_w_gimbal_com: InputPort
    in_w_gimbal_cool: InputPort
    in_sc_temp: InputPort
    in_sc_soc: InputPort
    in_is_occluded_earth: InputPort
    in_is_occluded_sun: InputPort
    in_is_fault: InputPort

class NodeSimpleFlightSoftwareOutputs(NamedTuple):
    out_state: OutputPort
    out_science: OutputPort
    out_science_dl: OutputPort
    out_q_cmd: OutputPort
    out_w_cmd: OutputPort
    out_solar_gimbal_cmd: OutputPort

class NodeSimpleFlightSoftware(Node):
    def __init__(self, **kwargs):
        """
        Inputs
        - pointing error (from each pointer)
        - tau cmd (from each pointer)
        - s/c temp
        - s/c SoC
        - s/c body rate measure
        - s/c orientation measure
        - is occluded (for earth and sun) # TODO Make occlusion output for earth and sun separately

        Outputs
        - state
        - total science
        - total science downlinked
        """

        self._i = NodeSimpleFlightSoftwareInputs(
            InputPort("in_point_err", self),
            InputPort("in_q_cmd_nadir", self),
            InputPort("in_q_cmd_ss", self),
            InputPort("in_q_cmd_com", self),
            InputPort("in_q_cmd_cool", self),
            InputPort("in_w_cmd_nadir", self),
            InputPort("in_w_cmd_ss", self),
            InputPort("in_w_cmd_com", self),
            InputPort("in_w_cmd_cool", self),
            InputPort("in_gimbal_cmd_nadir", self),
            InputPort("in_gimbal_cmd_ss", self),
            InputPort("in_gimbal_cmd_com", self),
            InputPort("in_gimbal_cmd_cool", self),
            InputPort("in_sc_temp", self),
            InputPort("in_sc_soc", self),
            InputPort("in_is_occluded_earth", self),
            InputPort("in_is_occluded_sun", self),
            InputPort("in_is_fault", self)
        )
        self._o = NodeSimpleFlightSoftwareOutputs(
            OutputPort("out_fs_state", self),
            OutputPort("out_science", self),
            OutputPort("out_science_downlinked", self),
            OutputPort("out_q_cmd", self),
            OutputPort("out_w_cmd", self),
            OutputPort("out_solar_gimbal", self)
        )

        super().__init__(self._i, self._o, **kwargs)

    def initialize(self):
        self._state = SimpleFlightState.SCIENCE
        self._science = 0
        self._science_dl = 0
        self._max_science = self._config["max_science"]
        self._warn_soc = self._config["warn_soc"]
        self._warn_temp = self._config["warn_temp"]
        self._nominal_soc = self._config["nominal_soc"]
        self._nominal_temp = self._config["nominal_temp"]
        self._point_err_filt_len = self._config["point_err_filt_len"]
        self._point_err_sci_lock = self._config["point_err_sci_lock"]
        self._point_err_dl_lock = self._config["point_err_dl_lock"]
        self._err_filt_data = deque(maxlen=self._point_err_filt_len)
        self._sci_rate_partial_lock = self._config["sci_rate_partial_lock"]
        self._sci_rate_full_lock = self._config["sci_rate_full_lock"]
        self._dl_rate_partial_lock = self._config["dl_rate_partial_lock"]
        self._dl_rate_full_lock = self._config["dl_rate_full_lock"]
        self._t = 0
        self._prev_safe_dl_time = None
        self._prev_safe_dl_duration = None
        self._safe_mode_task = SafeModeTask.CHARGE
        self._dl_try_period = self._config["dl_try_period"]
        self._dl_try_duration = self._config["dl_try_duration"]
        self._last_safe_state_switch_time = None

        # Safe state transition PD
        # self._soc_p = 2.0
        # self._soc_d = 2.0
        # self._temp_p = 1.0
        # self._temp_d = 2.0
        # self._soc_last = None
        # self._temp_last = None

        # self._cmd_s = []
        # self._cmd_k = []
        # self._ts = []

    def update(self, sim_time: float):
        soc = self._i.in_sc_soc.read()
        temp = self._i.in_sc_temp.read()
        fault = self._i.in_is_fault.read()

        self._err_filt_data.append(np.linalg.norm(self._i.in_point_err.read()))

        if fault:
            self._state = SimpleFlightState.SAFE

        if self._state == SimpleFlightState.SCIENCE:
            self._state_science(sim_time, soc, temp)
        elif self._state == SimpleFlightState.SUN_STUCK:
            self._state_sun_stuck(sim_time, soc, temp)
        elif self._state == SimpleFlightState.EARTH_SUN_COM:
            self._state_es_com(sim_time, soc, temp)
        elif self._state == SimpleFlightState.COOLING:
            self._state_cooling(sim_time, soc, temp)
        elif self._state == SimpleFlightState.SAFE:
            self._state_safe(sim_time, soc, temp)
        else:
            self._state_safe(sim_time, soc, temp)

        if self._state == SimpleFlightState.SAFE:
            self._o.out_fs_state.shift_out(
                f"{self._state}->{self._safe_mode_task}"
            )
        else:
            self._o.out_state.shift_out(f"{self._state}")

        self._o.out_science.shift_out(self._science)
        self._o.out_science_dl.shift_out(self._science_dl)

        self._t = sim_time

        self._soc_last = soc
        self._temp_last = temp

    # def finalize(self):
    #     plt.plot(self._ts, self._cmd_k, label='temp')
    #     plt.plot(self._ts, self._cmd_s, label='soc')
    #     plt.legend()
    #     plt.show()

    def _state_science(self, ts: float, soc: float, temp: float):
        dt = ts - self._t

        # increment science depending on pointing
        self._science += self._science_model(dt)
        self._science = np.clip(self._science, 0, self._max_science)

        self._o.out_q_cmd.shift_out(self._i.in_q_cmd_nadir.read())
        self._o.out_w_cmd.shift_out(self._i.in_w_cmd_nadir.read())
        self._o.out_solar_gimbal_cmd.shift_out(np.array([np.radians(0.0)]))

        # process mode transitions
        # TO DOWNLINK
        if self._science == self._max_science:
            self._state = SimpleFlightState.EARTH_SUN_COM

        # TO SUN_STUCK
        if soc < self._warn_soc:
            self._state = SimpleFlightState.SUN_STUCK

        # TO COOLING
        if temp > self._warn_temp:
            self._state = SimpleFlightState.COOLING

        # TO SAFE
        if temp > self._warn_temp and soc < self._warn_soc:
            self._state = SimpleFlightState.SAFE

    def _state_sun_stuck(self, ts: float, soc: float, temp: float):
        self._o.out_q_cmd.shift_out(self._i.in_q_cmd_ss.read())
        self._o.out_w_cmd.shift_out(self._i.in_w_cmd_ss.read())
        self._o.out_solar_gimbal_cmd.shift_out(
            self._i.in_w_gimbal_ss.read()
        )

        # Mode transitions
        # TO SCIENCE
        if soc > self._nominal_soc:
            if self._science > 0.5 * self._max_science:
                self._state = SimpleFlightState.EARTH_SUN_COM
            else:
                self._state = SimpleFlightState.SCIENCE

        # TO COOLING
        if temp > self._warn_temp:
            self._state = SimpleFlightState.COOLING

        # TO SAFE
        if temp > self._warn_temp and soc < self._warn_soc:
            self._state = SimpleFlightState.SAFE

    def _state_es_com(self, ts: float, soc: float, temp: float):
        self._o.out_q_cmd.shift_out(self._i.in_q_cmd_com.read())
        self._o.out_w_cmd.shift_out(self._i.in_w_cmd_com.read())
        self._o.out_solar_gimbal_cmd.shift_out(
            self._i.in_w_gimbal_com.read()
        )

        dt = ts - self._t

        sci_dl_available = self._science_dl_model(dt)
        sci_dl_amt = np.clip(
            sci_dl_available, 0, np.min([sci_dl_available, self._science])
        )
        self._science -= sci_dl_amt
        self._science_dl += sci_dl_amt
        self._science = np.clip(self._science, 0, self._max_science)

        # process mode transitions
        # TO SCIENCE
        if self._science == 0:
            self._state = SimpleFlightState.SCIENCE

        # TO SUN_STUCK
        if soc < self._warn_soc:
            self._state = SimpleFlightState.SUN_STUCK

        # TO COOLING
        if temp > self._warn_temp:
            self._state = SimpleFlightState.COOLING

        # TO SAFE
        if temp > self._warn_temp and soc < self._warn_soc:
            self._state = SimpleFlightState.SAFE

    def _state_cooling(self, ts: float, soc: float, temp: float):
        self._o.out_q_cmd.shift_out(self._i.in_q_cmd_cool.read())
        self._o.out_w_cmd.shift_out(self._i.in_w_cmd_cool.read())
        self._o.out_solar_gimbal_cmd.shift_out(
            self._i.in_w_gimbal_cool.read()
        )

        # Mode transitions
        # TO SCIENCE
        if temp < self._nominal_temp:
            if self._science > 0.5 * self._max_science:
                self._state = SimpleFlightState.EARTH_SUN_COM
            else:
                self._state = SimpleFlightState.SCIENCE

        # TO SUN_STUCk
        if soc < self._warn_soc:
            self._state = SimpleFlightState.SUN_STUCK

        # TO SAFE
        if temp > self._warn_temp and soc < self._warn_soc:
            self._state = SimpleFlightState.SAFE

    def _state_safe(self, ts: float, soc: float, temp: float):
        # Maintain temp and state of charge
        # Periodically go to com if temp and soc okay
        # is_earth_occluded = self._i.in_is_occluded_earth.read()
        # is_sun_occluded = self._i.in_is_occluded_sun.read()

        # if self._prev_safe_dl_time == None:
        #     self._prev_safe_dl_time = ts

        # if self._last_safe_state_switch_time == None:
        #     self._last_safe_state_switch_time = ts

        # # TO SUN_STUCK
        # if soc < self._warn_soc and is_sun_occluded == False:
        #     if (
        #         ts - self._last_safe_state_switch_time
        #         > self._safe_state_commit_duration
        #     ):
        #         self._last_safe_state_switch_time = ts
        #         if self._safe_mode_task == SafeModeTask.DL:
        #             self._prev_safe_dl_time = ts
        #         self._safe_mode_task = SafeModeTask.CHARGE
        # elif temp > self._warn_temp:
        #     if (
        #         ts - self._last_safe_state_switch_time
        #         > self._safe_state_commit_duration
        #     ):
        #         self._last_safe_state_switch_time = ts
        #         if self._safe_mode_task == SafeModeTask.DL:
        #             self._prev_safe_dl_time = ts
        #         self._safe_mode_task = SafeModeTask.COOL
        # elif (
        #     ts - self._prev_safe_dl_time > self._dl_try_period
        #     and is_earth_occluded == False
        # ):
        #     self._safe_mode_task = SafeModeTask.DL
        #     self._prev_safe_dl_duration = ts
        # else:
        #     self._safe_mode_task = SafeModeTask.CHARGE

        self._safe_mode_task = self._next_safe_task(ts, soc, temp)

        if self._safe_mode_task == SafeModeTask.CHARGE:
            self._o.out_q_cmd.shift_out(self._i.in_q_cmd_ss.read())
            self._o.out_w_cmd.shift_out(self._i.in_w_cmd_ss.read())
            self._o.out_solar_gimbal_cmd.shift_out(
                self._i.in_w_gimbal_ss.read()
            )
        elif self._safe_mode_task == SafeModeTask.COOL:
            self._o.out_q_cmd.shift_out(self._i.in_q_cmd_cool.read())
            self._o.out_w_cmd.shift_out(self._i.in_w_cmd_cool.read())
            self._o.out_solar_gimbal_cmd.shift_out(
                self._i.in_w_gimbal_cool.read()
            )
        elif self._safe_mode_task == SafeModeTask.DL:
            self._o.out_q_cmd.shift_out(self._i.in_q_cmd_com.read())
            self._o.out_w_cmd.shift_out(self._i.in_w_cmd_com.read())
            self._o.out_solar_gimbal_cmd.shift_out(
                self._i.in_w_gimbal_com.read()
            )
            # if ts - self._prev_safe_dl_duration > self._dl_try_duration:
            #     self._prev_safe_dl_time = ts

    def _science_model(self, dt: float) -> float:
        err_stdv = np.std(self._err_filt_data)
        if err_stdv < 0.1 * self._point_err_sci_lock:
            return self._sci_rate_full_lock * dt
        elif err_stdv < self._point_err_sci_lock:
            return self._sci_rate_partial_lock * dt
        else:
            return 0

    def _science_dl_model(self, dt: float) -> float:
        is_earth_occluded = self._i.in_is_occluded_earth.read()
        err_stdv = np.std(self._err_filt_data)
        if is_earth_occluded:
            return 0
        elif err_stdv < 0.1 * self._point_err_dl_lock:
            return self._dl_rate_full_lock * dt
        elif err_stdv < self._point_err_dl_lock:
            return self._dl_rate_partial_lock * dt
        else:
            return 0

    def _next_safe_task(self, ts: float, soc: float, temp: float):
        k_percent = (self._warn_temp - temp) / (self._warn_temp - self._nominal_temp)

        is_earth_occluded = self._i.in_is_occluded_earth.read()
        is_sun_occluded = self._i.in_is_occluded_sun.read()

        # err_s = self._nominal_soc - soc
        # err_k = 1.0 - k_percent

        # if self._soc_last != None:
        #     derr_s = soc - self._soc_last
        # else:
        #     derr_s = 0

        # if self._temp_last != None:
        #     derr_k = k_percent - (self._warn_temp - self._temp_last) / (
        #         self._warn_temp - self._nominal_temp
        #     )
        # else:
        #     derr_k = 0

        # cmd_s = self._soc_p * err_s - self._soc_d * derr_s
        # cmd_k = self._temp_p * err_k - self._temp_d * derr_k

        # self._cmd_k.append(cmd_k)
        # self._cmd_s.append(cmd_s)
        # self._ts.append(ts)

        # if (
        #     k_percent > 0.5
        #     and soc > 0.5
        #     and self._is_safe_to_downlink(ts)
        #     and is_earth_occluded == False
        # ):
        #     self._prev_safe_dl_duration = ts
        #     return SafeModeTask.DL
        # elif cmd_s > cmd_k and is_sun_occluded:
        #     return SafeModeTask.CHARGE
        # elif cmd_k >= cmd_s:
        #     return SafeModeTask.COOL
        # else:
        #     return SafeModeTask.CHARGE
        
        if (
            k_percent > 0.5
            and soc > 0.5
            and self._is_safe_to_downlink(ts)
            and is_earth_occluded == False
        ):
            self._prev_safe_dl_duration = ts
            return SafeModeTask.DL
        elif soc < self._warn_soc and is_sun_occluded == False:
            return SafeModeTask.CHARGE
        elif temp > self._warn_temp:
            return SafeModeTask.COOL
        else:
            return self._safe_mode_task

    def _is_safe_to_downlink(self, t: float):
        period = self._dl_try_period + self._dl_try_duration
        t_mod = t % period
        if t_mod < self._dl_try_period:
            return False
        else:
            return True

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o


class NodeFaultProtectionMGSInputs(NamedTuple):
    in_gimbal_cmd: InputPort
    in_gimbal_measure: InputPort

class NodeFaultProtectionMGSOutputs(NamedTuple):
    out_gimbal_fault: OutputPort

class NodeFaultProtectionMGS(NodeDifferential):
    def __init__(self, **kwargs):
        self._i = NodeFaultProtectionMGSInputs(
            InputPort("in_gimbal_cmd", self),
            InputPort("in_gimbal_measure", self)
        )
        self._o = NodeFaultProtectionMGSOutputs(
            OutputPort("out_gimbal_fault", self)
        )

        super().__init__(None, self._i, self._o, **kwargs)

    def initialize(self):
        self._threshold = np.radians(self._config["monitor-threshold"])
        self._persist = self._config["monitor-persistance"]
        self._trig = False
        self._last_thresh = None
        self._last_dg = None

    def update(self, sim_time: float):
        g_cmd = self._i.in_gimbal_cmd.read()
        g_meas = self._i.in_gimbal_measure.read()
        if g_cmd == None:
            g_cmd = np.array([0])

        if g_meas == None:
            g_meas = np.array([0])
        dg = np.abs(g_cmd - g_meas)

        if self._trig == True:
            self._o.out_gimbal_fault.shift_out(self._trig)
        else:
            exceed = np.any(dg > self._threshold)
            if self._last_dg != None:
                last_exceed = np.any(self._last_dg > self._threshold)
            else:
                last_exceed = False

            if exceed and not last_exceed:
                self._last_thresh = sim_time
            elif last_exceed and not exceed:
                self._last_thresh = None

            if self._last_thresh != None:
                if (
                    (sim_time - self._last_thresh) > self._persist
                    and exceed
                    and self._trig == False
                ):
                    self._trig = True
                    self._system.detect_fault("gimbal fault", sim_time)

            self._last_dg = dg
            self._o.out_gimbal_fault.shift_out(self._trig)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o
