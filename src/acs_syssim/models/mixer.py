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
            self.o.wheel1_torque.shift_out(res.x[0] * self.p.axis1.value)
            self.o.wheel2_torque.shift_out(res.x[1] * self.p.axis2.value)
            self.o.wheel3_torque.shift_out(res.x[2] * self.p.axis3.value)
            self.o.wheel4_torque.shift_out(res.x[3] * self.p.axis4.value)
        else:
            self.o.wheel1_torque.shift_out(0.0)
            self.o.wheel2_torque.shift_out(0.0)
            self.o.wheel3_torque.shift_out(0.0)
            self.o.wheel4_torque.shift_out(0.0)


class InternalAngularMomentumMuxerNode(Node):
    """Node that estimates the internal angular momentum of the system from reaction wheel speeds."""
    
    class Inputs(NamedTuple):
        wheel1_speed: InputPort
        wheel2_speed: InputPort
        wheel3_speed: InputPort
        wheel4_speed: InputPort

    class Outputs(NamedTuple):
        angular_momentum: OutputPort  # 3x1 vector output in body frame

    class Parameters(NamedTuple):
        inertia1: NodeParameter
        inertia2: NodeParameter
        inertia3: NodeParameter
        inertia4: NodeParameter

    def __init__(
        self,
        inertia1: float = 1.0,
        inertia2: float = 1.0,
        inertia3: float = 1.0,
        inertia4: float = 1.0,
        axis1_vector: NDArray = np.array([1, 0, 0]),
        axis2_vector: NDArray = np.array([0, 1, 0]),
        axis3_vector: NDArray = np.array([0, 0, 1]),
        axis4_vector: NDArray = np.array([1, 1, 1]),
        **kwargs
    ):
        """Initialize the angular momentum estimator node.
        
        Args:
            inertia1: Inertia moment for wheel 1 [kg*m^2]
            inertia2: Inertia moment for wheel 2 [kg*m^2]
            inertia3: Inertia moment for wheel 3 [kg*m^2]
            inertia4: Inertia moment for wheel 4 [kg*m^2]
            
        Ports:
            wheel1_speed: Speed of wheel 1 [rad/s]
            wheel2_speed: Speed of wheel 2 [rad/s]
            wheel3_speed: Speed of wheel 3 [rad/s]
            wheel4_speed: Speed of wheel 4 [rad/s]
            angular_momentum: Total internal angular momentum vector [Nms]
            
        Config:
            axis1_vector: Body frame unit vector for wheel 1 orientation [3x1]
            axis2_vector: Body frame unit vector for wheel 2 orientation [3x1]
            axis3_vector: Body frame unit vector for wheel 3 orientation [3x1]
            axis4_vector: Body frame unit vector for wheel 4 orientation [3x1]
        """
        # Define ports
        self.input_ports = self.Inputs(
            InputPort("wheel1_speed", self),
            InputPort("wheel2_speed", self),
            InputPort("wheel3_speed", self),
            InputPort("wheel4_speed", self),
        )
        self.output_ports = self.Outputs(
            OutputPort("angular_momentum", self),
        )
        
        # Define parameters (inertias)
        self.parameters = self.Parameters(
            NodeParameter("inertia1", inertia1),
            NodeParameter("inertia2", inertia2),
            NodeParameter("inertia3", inertia3),
            NodeParameter("inertia4", inertia4),
        )

        super().__init__(self.input_ports, self.output_ports, self.parameters, **kwargs)
        
        self._axis1 = np.array(self._config.get("axis1_vector", axis1_vector))
        self._axis2 = np.array(self._config.get("axis2_vector", axis2_vector))
        self._axis3 = np.array(self._config.get("axis3_vector", axis3_vector))
        self._axis4 = np.array(self._config.get("axis4_vector", axis4_vector))
        

    def initialize(self):
        """Initialize configuration parameters."""        # Normalize axis vectors
        self._axis1 = self._axis1 / np.linalg.norm(self._axis1)
        self._axis2 = self._axis2 / np.linalg.norm(self._axis2)
        self._axis3 = self._axis3 / np.linalg.norm(self._axis3)
        self._axis4 = self._axis4 / np.linalg.norm(self._axis4)

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
        """Update the angular momentum estimate based on wheel speeds."""
        # Read wheel speeds
        speed1 = self.i.wheel1_speed.read()
        speed2 = self.i.wheel2_speed.read()
        speed3 = self.i.wheel3_speed.read()
        speed4 = self.i.wheel4_speed.read()
        
        # Handle None inputs (default to zero)
        if speed1 is None:
            speed1 = 0.0
        if speed2 is None:
            speed2 = 0.0
        if speed3 is None:
            speed3 = 0.0
        if speed4 is None:
            speed4 = 0.0
        
        # Calculate individual wheel angular momentum vectors
        h1 = self.p.inertia1.value * speed1 * self._axis1
        h2 = self.p.inertia2.value * speed2 * self._axis2
        h3 = self.p.inertia3.value * speed3 * self._axis3
        h4 = self.p.inertia4.value * speed4 * self._axis4
        
        # Sum all wheel angular momentum contributions
        total_angular_momentum = h1 + h2 + h3 + h4
        
        # Output the total internal angular momentum
        self.o.angular_momentum.shift_out(total_angular_momentum)


