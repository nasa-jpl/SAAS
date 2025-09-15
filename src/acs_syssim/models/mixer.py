from syssim.core.node import Node, NodeParameter, NodeDifferential
from syssim.core.port import InputPort, OutputPort
import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple
from scipy.optimize import linprog
from tqdm import tqdm


class DelayNode(NodeDifferential):
    """A simple delay node that outputs the previous timestep's input."""
    
    class Inputs(NamedTuple):
        input: InputPort

    class Outputs(NamedTuple):
        output: OutputPort

    def __init__(self, initial_value=None, **kwargs):
        self.input_ports = self.Inputs(
            InputPort("input", self),
        )
        self.output_ports = self.Outputs(
            OutputPort("output", self),
        )
        self._initial_value = initial_value
        super().__init__(self._initial_value, self.input_ports, self.output_ports, **kwargs)

    @property
    def i(self):
        return self.input_ports

    @property
    def o(self):
        return self.output_ports

    def update(self, sim_time: float):
        """Read input and store as state for next timestep output."""
        current_input = self.i.input.read()
        
        # Output the previous timestep's value (stored in self._x)
        self.o.output.shift_out(self._x)
        
        # Store current input as state for next timestep
        self._x = current_input


class ReactionWheelMixerNode(Node):
    class Inputs(NamedTuple):
        commanded_torque: InputPort
        health: InputPort

    class Outputs(NamedTuple):
        wheel1_torque: OutputPort  # Scalar output for wheel 1
        wheel2_torque: OutputPort  # Scalar output for wheel 2
        wheel3_torque: OutputPort  # Scalar output for wheel 3
        wheel4_torque: OutputPort  # Scalar output for wheel 4
        wheel5_torque: OutputPort  # Scalar output for wheel 5
        wheel6_torque: OutputPort  # Scalar output for wheel 6
        wheel7_torque: OutputPort  # Scalar output for wheel 7
        wheel8_torque: OutputPort  # Scalar output for wheel 8

    class Parameters(NamedTuple):
        axis1: NodeParameter
        axis2: NodeParameter
        axis3: NodeParameter
        axis4: NodeParameter
        axis5: NodeParameter
        axis6: NodeParameter
        axis7: NodeParameter
        axis8: NodeParameter

    def __init__(
        self,
        axis1: NDArray = np.array([1, 0, 0]),
        axis2: NDArray = np.array([-1, 0, 0]),
        axis3: NDArray = np.array([0, 1, 0]),
        axis4: NDArray = np.array([0, -1, 0]),
        axis5: NDArray = np.array([0, 0, 1]),
        axis6: NDArray = np.array([0, 0, -1]),
        axis7: NDArray = np.array([1, 1, 0]),
        axis8: NDArray = np.array([-1, -1, 0]),
        **kwargs
    ):
        # Define ports
        self.input_ports = self.Inputs(
            InputPort("commanded_torque", self),
            InputPort("health", self),
        )
        self.output_ports = self.Outputs(
            OutputPort("wheel1_torque", self),
            OutputPort("wheel2_torque", self),
            OutputPort("wheel3_torque", self),
            OutputPort("wheel4_torque", self),
            OutputPort("wheel5_torque", self),
            OutputPort("wheel6_torque", self),
            OutputPort("wheel7_torque", self),
            OutputPort("wheel8_torque", self),
        )
        # Define parameters (axes)
        self.parameters = self.Parameters(
            NodeParameter("axis1", axis1),
            NodeParameter("axis2", axis2),
            NodeParameter("axis3", axis3),
            NodeParameter("axis4", axis4),
            NodeParameter("axis5", axis5),
            NodeParameter("axis6", axis6),
            NodeParameter("axis7", axis7),
            NodeParameter("axis8", axis8),
        )
        super().__init__(self.input_ports, self.output_ports, self.parameters, **kwargs)

        # Track wheels permanently disabled due to any RWA_X/ENC_X detection
        # store indices 0..7 for wheels 1..8
        self._disabled_wheels = set()

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
        
        # Read health dictionary
        health_dict = self.i.health.read()
        if health_dict is None:
            health_dict = {}

        # Update permanent disables: if MONSID/health ever reports RWA_X or ENC_X as not "Healthy",
        # mark that wheel permanently disabled for remainder of simulation.
        # (We expect health_dict values like "Healthy" / "Faulty".)
        for wheel_idx in range(1, 9):
            rwa_key = f"RWA_{wheel_idx}"
            enc_key = f"ENC_{wheel_idx}"
            status_rwa = health_dict.get(rwa_key, None)
            status_enc = health_dict.get(enc_key, None)
            if (status_rwa is not None and status_rwa != "Healthy") or (status_enc is not None and status_enc != "Healthy"):
                # store zero-based index
                self._disabled_wheels.add(wheel_idx - 1)

        # Check health status for each reaction wheel (RWA_X and ENC_X must both be Healthy)
        rwa_health = []
        for i in range(1, 9):  # RWA_1 through RWA_8
            idx0 = i - 1
            # If permanently disabled, mark unhealthy regardless of current health
            if idx0 in self._disabled_wheels:
                rwa_health.append(0)
                continue
            rwa_key = f"RWA_{i}"
            enc_key = f"ENC_{i}"
            is_rwa_healthy = health_dict.get(rwa_key, "Healthy") == "Healthy"
            is_enc_healthy = health_dict.get(enc_key, "Healthy") == "Healthy"
            is_healthy = is_rwa_healthy and is_enc_healthy
            rwa_health.append(int(is_healthy))

        # Create problem bounds for milp
        bounds = []
        for health_status in rwa_health:
            if health_status == 0:  # Faulty
                bounds.append((0, 0))
            else:  # Healthy
                bounds.append((None, None))
        bounds += [(0, None)] * 8  # t variables for L1 norm

        # Get axes from parameters
        axes = [
            self.p.axis1.value,
            self.p.axis2.value,
            self.p.axis3.value,
            self.p.axis4.value,
            self.p.axis5.value,
            self.p.axis6.value,
            self.p.axis7.value,
            self.p.axis8.value,
        ]

        axes = [np.array(a) / np.linalg.norm(a) for a in axes]
        A = np.stack(axes, axis=1)  # 3x8

        # Build augmented LP to minimize sum of absolute values of x (L1):
        # variables z = [x (8), t (8)]
        # minimize sum(t)
        # s.t. A x = commanded_torque
        #      x - t <= 0
        #      -x - t <= 0
        n = 8
        c = np.concatenate([np.zeros(n), np.ones(n) / n])    # minimize sum(t) / n

        # Equality: A * x + 0 * t = commanded_torque
        A_eq_aug = np.hstack([A, np.zeros((A.shape[0], n))])

        # Inequalities assembling
        I = np.eye(n)
        A_ub = np.vstack([
            np.hstack([I, -I]),   # x - t <= 0
            np.hstack([-I, -I]),  # -x - t <= 0  -> -x - t <= 0
        ])
        b_ub = np.zeros(2 * n)

        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq_aug,
            b_eq=commanded_torque,
            bounds=bounds,
            method="highs",
        )

        # res = linprog(
        #     c=np.zeros(8) / 8,
        #     A_eq=A,
        #     b_eq=commanded_torque,
        #     bounds=bounds,
        #     method="highs",
        # )
        if res.success:
            self.o.wheel1_torque.shift_out(res.x[0] * self.p.axis1.value)
            self.o.wheel2_torque.shift_out(res.x[1] * self.p.axis2.value)
            self.o.wheel3_torque.shift_out(res.x[2] * self.p.axis3.value)
            self.o.wheel4_torque.shift_out(res.x[3] * self.p.axis4.value)
            self.o.wheel5_torque.shift_out(res.x[4] * self.p.axis5.value)
            self.o.wheel6_torque.shift_out(res.x[5] * self.p.axis6.value)
            self.o.wheel7_torque.shift_out(res.x[6] * self.p.axis7.value)
            self.o.wheel8_torque.shift_out(res.x[7] * self.p.axis8.value)
        else:
            self.o.wheel1_torque.shift_out(np.zeros(3))
            self.o.wheel2_torque.shift_out(np.zeros(3))
            self.o.wheel3_torque.shift_out(np.zeros(3))
            self.o.wheel4_torque.shift_out(np.zeros(3))
            self.o.wheel5_torque.shift_out(np.zeros(3))
            self.o.wheel6_torque.shift_out(np.zeros(3))
            self.o.wheel7_torque.shift_out(np.zeros(3))
            self.o.wheel8_torque.shift_out(np.zeros(3))


class InternalAngularMomentumMuxerNode(Node):
    """Node that estimates the internal angular momentum of the system from reaction wheel speeds."""
    
    class Inputs(NamedTuple):
        wheel1_speed: InputPort
        wheel2_speed: InputPort
        wheel3_speed: InputPort
        wheel4_speed: InputPort
        wheel5_speed: InputPort
        wheel6_speed: InputPort
        wheel7_speed: InputPort
        wheel8_speed: InputPort

    class Outputs(NamedTuple):
        angular_momentum: OutputPort  # 3x1 vector output in body frame

    class Parameters(NamedTuple):
        inertia1: NodeParameter
        inertia2: NodeParameter
        inertia3: NodeParameter
        inertia4: NodeParameter
        inertia5: NodeParameter
        inertia6: NodeParameter
        inertia7: NodeParameter
        inertia8: NodeParameter

    def __init__(
        self,
        inertia1: float = 1.0,
        inertia2: float = 1.0,
        inertia3: float = 1.0,
        inertia4: float = 1.0,
        inertia5: float = 1.0,
        inertia6: float = 1.0,
        inertia7: float = 1.0,
        inertia8: float = 1.0,
        axis1_vector: NDArray = np.array([1, 0, 0]),
        axis2_vector: NDArray = np.array([-1, 0, 0]),
        axis3_vector: NDArray = np.array([0, 1, 0]),
        axis4_vector: NDArray = np.array([0, -1, 0]),
        axis5_vector: NDArray = np.array([0, 0, 1]),
        axis6_vector: NDArray = np.array([0, 0, -1]),
        axis7_vector: NDArray = np.array([1, 1, 0]),
        axis8_vector: NDArray = np.array([-1, -1, 0]),
        **kwargs
    ):
        """Initialize the angular momentum estimator node.
        
        Args:
            inertia1: Inertia moment for wheel 1 [kg*m^2]
            inertia2: Inertia moment for wheel 2 [kg*m^2]
            inertia3: Inertia moment for wheel 3 [kg*m^2]
            inertia4: Inertia moment for wheel 4 [kg*m^2]
            inertia5: Inertia moment for wheel 5 [kg*m^2]
            inertia6: Inertia moment for wheel 6 [kg*m^2]
            
        Ports:
            wheel1_speed: Speed of wheel 1 [rad/s]
            wheel2_speed: Speed of wheel 2 [rad/s]
            wheel3_speed: Speed of wheel 3 [rad/s]
            wheel4_speed: Speed of wheel 4 [rad/s]
            wheel5_speed: Speed of wheel 5 [rad/s]
            wheel6_speed: Speed of wheel 6 [rad/s]
            angular_momentum: Total internal angular momentum vector [Nms]
            
        Config:
            axis1_vector: Body frame unit vector for wheel 1 orientation [3x1]
            axis2_vector: Body frame unit vector for wheel 2 orientation [3x1]
            axis3_vector: Body frame unit vector for wheel 3 orientation [3x1]
            axis4_vector: Body frame unit vector for wheel 4 orientation [3x1]
            axis5_vector: Body frame unit vector for wheel 5 orientation [3x1]
            axis6_vector: Body frame unit vector for wheel 6 orientation [3x1]
        """
        # Define ports
        self.input_ports = self.Inputs(
            InputPort("wheel1_speed", self),
            InputPort("wheel2_speed", self),
            InputPort("wheel3_speed", self),
            InputPort("wheel4_speed", self),
            InputPort("wheel5_speed", self),
            InputPort("wheel6_speed", self),
            InputPort("wheel7_speed", self),
            InputPort("wheel8_speed", self),
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
            NodeParameter("inertia5", inertia5),
            NodeParameter("inertia6", inertia6),
            NodeParameter("inertia7", inertia7),
            NodeParameter("inertia8", inertia8),
        )

        super().__init__(self.input_ports, self.output_ports, self.parameters, **kwargs)
        
        self._axis1 = np.array(self._config.get("axis1_vector", axis1_vector))
        self._axis2 = np.array(self._config.get("axis2_vector", axis2_vector))
        self._axis3 = np.array(self._config.get("axis3_vector", axis3_vector))
        self._axis4 = np.array(self._config.get("axis4_vector", axis4_vector))
        self._axis5 = np.array(self._config.get("axis5_vector", axis5_vector))
        self._axis6 = np.array(self._config.get("axis6_vector", axis6_vector))
        self._axis7 = np.array(self._config.get("axis7_vector", axis7_vector))
        self._axis8 = np.array(self._config.get("axis8_vector", axis8_vector))
        

    def initialize(self):
        """Initialize configuration parameters."""        # Normalize axis vectors
        self._axis1 = self._axis1 / np.linalg.norm(self._axis1)
        self._axis2 = self._axis2 / np.linalg.norm(self._axis2)
        self._axis3 = self._axis3 / np.linalg.norm(self._axis3)
        self._axis4 = self._axis4 / np.linalg.norm(self._axis4)
        self._axis5 = self._axis5 / np.linalg.norm(self._axis5)
        self._axis6 = self._axis6 / np.linalg.norm(self._axis6)
        self._axis7 = self._axis7 / np.linalg.norm(self._axis7)
        self._axis8 = self._axis8 / np.linalg.norm(self._axis8)

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
        speed5 = self.i.wheel5_speed.read()
        speed6 = self.i.wheel6_speed.read()
        speed7 = self.i.wheel7_speed.read()
        speed8 = self.i.wheel8_speed.read()
        
        # Handle None inputs (default to zero)
        if speed1 is None:
            speed1 = 0.0
        if speed2 is None:
            speed2 = 0.0
        if speed3 is None:
            speed3 = 0.0
        if speed4 is None:
            speed4 = 0.0
        if speed5 is None:
            speed5 = 0.0
        if speed6 is None:
            speed6 = 0.0
        if speed7 is None:
            speed7 = 0.0
        if speed8 is None:
            speed8 = 0.0
        
        # Calculate individual wheel angular momentum vectors
        h1 = self.p.inertia1.value * speed1 * self._axis1
        h2 = self.p.inertia2.value * speed2 * self._axis2
        h3 = self.p.inertia3.value * speed3 * self._axis3
        h4 = self.p.inertia4.value * speed4 * self._axis4
        h5 = self.p.inertia5.value * speed5 * self._axis5
        h6 = self.p.inertia6.value * speed6 * self._axis6
        h7 = self.p.inertia7.value * speed7 * self._axis7
        h8 = self.p.inertia8.value * speed8 * self._axis8
        
        # Sum all wheel angular momentum contributions
        total_angular_momentum = h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8

        # Output the total internal angular momentum
        self.o.angular_momentum.shift_out(total_angular_momentum)


