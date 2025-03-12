from datetime import datetime
from typing import NamedTuple

import numpy as np

from syssim import Node, OutputPort

class NodeRandomTimeDateOutputs(NamedTuple):
    out_datetime: OutputPort

class NodeRandomTimeDate(Node):
    def __init__(self, **kwargs):
        self._o = NodeRandomTimeDateOutputs(
            OutputPort("out_datetime", self)
        )

        super().__init__((), self._o, **kwargs)

    def initialize(self):
        d0: datetime = self._config["date_begin"]
        d1: datetime = self._config["date_end"]

        new_stamp = np.random.uniform(d0.timestamp(), d1.timestamp())
        d_new = datetime.fromtimestamp(new_stamp)

        self._o.out_datetime.shift_out(d_new)

    @property
    def o(self):
        return self._o
    
    @property
    def i(self):
        return self._i


