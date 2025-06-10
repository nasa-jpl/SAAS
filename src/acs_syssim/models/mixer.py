from syssim.core.node import Node, NodeParameter
from syssim.core.port import InputPort, OutputPort
import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple
from scipy.optimize import linprog


class ReactionWheelMixerNode(Node):
    class Inputs(NamedTuple):
        commanded_torque: InputPort
        axis1_health: InputPort
        axis2_health: InputPort
        axis3_health: InputPort
        axis4_health: InputPort

    class Outputs(NamedTuple):
        wheel1_torque: OutputPort  # Scalar output for wheel 1
        wheel2_torque: OutputPort  # Scalar output for wheel 2
        wheel3_torque: OutputPort  # Scalar output for wheel 3
        wheel4_torque: OutputPort  # Scalar output for wheel 4

    class Parameters(NamedTuple):
        axis1: NodeParameter
        axis2: NodeParameter
        axis3: NodeParameter
        axis4: NodeParameter

    def __init__(
        self,
        axis1: NDArray = np.array([1, 0, 0]),
        axis2: NDArray = np.array([0, 1, 0]),
        axis3: NDArray = np.array([0, 0, 1]),
        axis4: NDArray = np.array([1, 1, 1]) / np.sqrt(3),
        **kwargs
    ):
        # Define ports
        self.input_ports = self.Inputs(
            InputPort("commanded_torque", self),
            InputPort("axis1_health", self),
            InputPort("axis2_health", self),
            InputPort("axis3_health", self),
            InputPort("axis4_health", self),
        )
        self.output_ports = self.Outputs(
            OutputPort("wheel1_torque", self),
            OutputPort("wheel2_torque", self),
            OutputPort("wheel3_torque", self),
            OutputPort("wheel4_torque", self),
        )
        # Define parameters (axes)
        self.parameters = self.Parameters(
            NodeParameter("axis1", axis1),
            NodeParameter("axis2", axis2),
            NodeParameter("axis3", axis3),
            NodeParameter("axis4", axis4),
        )
        super().__init__(self.input_ports, self.output_ports, self.parameters, **kwargs)

    @property
    def i(self):
        return self.input_ports

    @property
    def o(self):
        return self.output_ports

    @property
    def p(self):
        return self.parameters

    def update(self, sim_time: float):
        # Get commanded torque from input port
        commanded_torque: NDArray = self.i.commanded_torque.read()
        # Read wheel faults as int cast from boolean health status
        axis1_health: int = int(self.i.axis1_health.read())
        axis2_health: int = int(self.i.axis2_health.read())
        axis3_health: int = int(self.i.axis3_health.read())
        axis4_health: int = int(self.i.axis4_health.read())

        # Create problem bounds for milp
        bounds = []
        for z in [axis1_health, axis2_health, axis3_health, axis4_health]:
            if z == 0:
                bounds.append((0, 0))
            else:
                bounds.append((None, None))

        # Get axes from parameters
        axes = [
            self.p.axis1.value,
            self.p.axis2.value,
            self.p.axis3.value,
            self.p.axis4.value,
        ]

        axes = [np.array(a) / np.linalg.norm(a) for a in axes]
        A = np.stack(axes, axis=1)  # 3x4

        res = linprog(
            c=np.zeros(
                4
            ),  # Objective function is zero since we just want to satisfy the constraints
            A_eq=A,
            b_eq=commanded_torque,
            bounds=bounds,
            method="highs",
        )
        if res.success:
            self.o.wheel1_torque.shift_out(res.x[0])
            self.o.wheel2_torque.shift_out(res.x[1])
            self.o.wheel3_torque.shift_out(res.x[2])
            self.o.wheel4_torque.shift_out(res.x[3])
        else:
            self.o.wheel1_torque.shift_out(0.0)
            self.o.wheel2_torque.shift_out(0.0)
            self.o.wheel3_torque.shift_out(0.0)
            self.o.wheel4_torque.shift_out(0.0)
