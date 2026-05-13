import numpy as np
from scipy.spatial.transform import Rotation
from typing import NamedTuple

from syssim import Node, InputPort, OutputPort

class NodeIMUSimpleInputs(NamedTuple):
    input_true_angular_rate: InputPort
    input_true_acceleration: InputPort
    input_q_sc2eci: InputPort

class NodeIMUSimpleOutputs(NamedTuple):
    output_measure_angular_rate: OutputPort
    output_measure_acceleration: OutputPort

class NodeIMUSimple(Node):
    # Measurement noise, constant bias
    def __init__(self, **kwargs):
        self._i = NodeIMUSimpleInputs(
            InputPort("true_w", self),
            InputPort("true_a", self),
            InputPort("q_sc2eci", self)
        )
        self._o = NodeIMUSimpleOutputs(
            OutputPort("measure_w", self),
            OutputPort("measure_a", self)
        )
        super().__init__(self._i, self._o, **kwargs)
    
    def initialize(self):
        self._w_sample = lambda w: np.random.normal(
            w, self._config["w_noise"]) + self._config["w_bias"]
        self._a_sample = lambda a: np.random.normal(
            a, self._config["a_noise"]) + self._config["a_bias"]



    def update(self, sim_time: float):
        q_sc2eci = self._i.input_q_sc2eci.read()
        if np.any(q_sc2eci) != None:
            r_sc2eci = Rotation.from_quat(
                [q_sc2eci[1], q_sc2eci[2], q_sc2eci[3], q_sc2eci[0]])
        else:
            r_sc2eci = Rotation.identity()

        w_t = self._i.input_true_angular_rate.read()
        if np.any(w_t) == None:
            w_t = np.zeros((3,))

        a_t = self._i.input_true_acceleration.read()
        if np.any(a_t) == None:
            a_t = np.zeros((3,))

        a_t = r_sc2eci.inv().apply(a_t)

        w_m = self._w_sample(w_t)
        a_m = self._a_sample(a_t)

        self._o.output_measure_angular_rate.shift_out(w_m)
        self._o.output_measure_acceleration.shift_out(a_m)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o
