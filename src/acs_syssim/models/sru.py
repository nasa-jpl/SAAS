import numpy as np
from scipy.spatial.transform import Rotation
from typing import NamedTuple

from syssim.core import Node, InputPort, OutputPort

class NodeStellarReferenceUnitSimpleInputs(NamedTuple):
    input_q_sc2eci: InputPort

class NodeStellarReferenceUnitSimpleOutputs(NamedTuple):
    output_q_sc2eci_measure: OutputPort

class NodeStellarReferenceUnitSimple(Node):
    def __init__(self, **kwargs):
        self._i = NodeStellarReferenceUnitSimpleInputs(
            InputPort("input_q_sc2eci", self)
        )
        self._o = NodeStellarReferenceUnitSimpleOutputs(
            OutputPort("output_q_sc2eci_measure", self)
        )

        super().__init__(self._i, self._o, **kwargs)

    def update(self, sim_time: float):
        q_sc2eci = self._i.input_q_sc2eci.read()

        cross_noise = self._config["sru_cross_noise"]
        roll_noise = self._config["sru_roll_noise"]

        eul_sc2eci = Rotation.from_quat(
            [q_sc2eci[1], q_sc2eci[2], q_sc2eci[3], q_sc2eci[0]]
        ).as_euler("ZYX")

        eul_sc2eci_measure = [
            np.random.normal(eul_sc2eci[0], cross_noise),
            np.random.normal(eul_sc2eci[1], cross_noise),
            np.random.normal(eul_sc2eci[2], roll_noise),
        ]

        q_sc2eci_measure = Rotation.from_euler("ZYX", eul_sc2eci_measure).as_quat(canonical=True)

        # NOTE no idea why I had to negate the quaternion here. Should find out why...
        self._o.output_q_sc2eci_measure.shift_out(
            np.array(
                [
                    -q_sc2eci_measure[3],
                    -q_sc2eci_measure[0],
                    -q_sc2eci_measure[1],
                    -q_sc2eci_measure[2],
                ]
            )
        )

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o
