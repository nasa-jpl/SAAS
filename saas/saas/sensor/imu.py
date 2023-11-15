import numpy as np
from scipy.spatial.transform import Rotation

from syssim import Node, InputPort, OutputPort


class NodeIMUSimple(Node):
    # Measurement noise, constant bias
    def __init__(self, **kwargs):
        input_true_angular_rate = InputPort("true_w", self)
        input_true_acceleration = InputPort("true_a", self)
        input_q_sc2eci = InputPort("q_sc2eci", self)

        output_measure_angular_rate = OutputPort("measure_w", self)
        output_measure_acceleration = OutputPort("measure_a", self)

        ports = {input_true_angular_rate.name: input_true_angular_rate,
                 input_true_acceleration.name: input_true_acceleration,
                 input_q_sc2eci.name: input_q_sc2eci,
                 output_measure_angular_rate.name: output_measure_angular_rate,
                 output_measure_acceleration.name: output_measure_acceleration, }

        self._w_sample = lambda w: np.random.normal(
            w, self._config["w_noise"]) + self._config["w_bias"]
        self._a_sample = lambda a: np.random.normal(
            a, self._config["a_noise"]) + self._config["a_bias"]

        super().__init__(ports, **kwargs)

    def update(self, sim_time: float):
        q_sc2eci = self._ports["q_sc2eci"].read()
        if np.any(q_sc2eci) != None:
            r_sc2eci = Rotation.from_quat(
                [q_sc2eci[1], q_sc2eci[2], q_sc2eci[3], q_sc2eci[0]])
        else:
            r_sc2eci = Rotation.identity()

        w_t = self._ports["true_w"].read()
        if np.any(w_t) == None:
            w_t = np.zeros((3,))

        a_t = self._ports["true_a"].read()
        if np.any(a_t) == None:
            a_t = np.zeros((3,))

        a_t = r_sc2eci.inv().apply(a_t)

        w_m = self._w_sample(w_t)
        a_m = self._a_sample(a_t)
        # print(f"a_m = {a_m}\nw_m={w_m}")

        self._ports["measure_w"].shift_out(w_m)
        self._ports["measure_a"].shift_out(a_m)
