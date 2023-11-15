from datetime import datetime

import numpy as np

from syssim import Node, OutputPort


class NodeRandomTimeDate(Node):
    def __init__(self, **kwargs):
        out_datetime = OutputPort("out_datetime", self)
        ports = {
            out_datetime.name: out_datetime,
        }
        super().__init__(ports, **kwargs)

    def initialize(self):
        d0: datetime = self._config["date_begin"]
        d1: datetime = self._config["date_end"]

        new_stamp = np.random.uniform(d0.timestamp(), d1.timestamp())
        d_new = datetime.fromtimestamp(new_stamp)

        self._ports["out_datetime"].shift_out(d_new)


