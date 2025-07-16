import numpy as np
from typing import NamedTuple

from syssim import Node, InputPort, OutputPort

class NodeWheelEncoderInputs(NamedTuple):
    enc_in: InputPort

class NodeWheelEncoderOutputs(NamedTuple):
    enc_out: OutputPort

class NodeWheelEncoder(Node):
    # Measurement noise, constant bias
    def __init__(self, **kwargs):
        self._i = NodeWheelEncoderInputs(
            InputPort("enc_in", self),
        )
        self._o = NodeWheelEncoderOutputs(
            OutputPort("enc_out", self),
        )
        super().__init__(self._i, self._o, **kwargs)
    
    def initialize(self):
        self._w_sample = lambda w: np.random.normal(
            w, self._config["w_noise"]) + self._config["w_bias"]

    def update(self, sim_time: float):
        w_t = self._i.enc_in.read()
        if np.any(w_t) == None:
            w_t = 0
        else:
            w_t = np.linalg.norm(w_t)

        w_m = self._w_sample(w_t)

        self._o.enc_out.shift_out(w_m)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o
