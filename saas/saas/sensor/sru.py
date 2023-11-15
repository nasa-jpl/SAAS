import numpy as np
from scipy.spatial.transform import Rotation

from syssim.core import Node, InputPort, OutputPort


class NodeStellarReferenceUnitSimple(Node):
    def __init__(self, **kwargs):
        input_q_sc2eci = InputPort("input_q_sc2eci", self)

        output_q_sc2eci_measure = OutputPort("output_q_sc2eci_measure", self)

        ports = {
            input_q_sc2eci.name: input_q_sc2eci,
            output_q_sc2eci_measure.name: output_q_sc2eci_measure,
        }

        super().__init__(ports, **kwargs)

    def update(self, sim_time: float):
        q_sc2eci = self._ports["input_q_sc2eci"].read()

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
        self._ports["output_q_sc2eci_measure"].shift_out(
            np.array(
                [
                    -q_sc2eci_measure[3],
                    -q_sc2eci_measure[0],
                    -q_sc2eci_measure[1],
                    -q_sc2eci_measure[2],
                ]
            )
        )
