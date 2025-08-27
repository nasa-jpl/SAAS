import numpy as np
from typing import NamedTuple
from syssim import Node, InputPort, OutputPort


class NodeAutonomyInputs(NamedTuple):
    health_report: InputPort


class NodeAutonomyOutputs(NamedTuple):
    target_orientation: OutputPort
    target_angular_rate: OutputPort
    is_safe_mode: OutputPort


class NodeAutonomy(Node):
    """
    Node responsible for deciding when the system should go to safe mode.
    
    In normal operation mode, outputs an orientation vector and angular rate
    that are slewing at a constant rate to mimic maintaining a nadir pointing
    vector during orbit.
    
    In safe mode, outputs a constant inertial vector with no angular rate.
    
    Safe mode is triggered when less than 3 reaction wheels or 0 star trackers
    are active.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the autonomy node.
        
        Ports:
            health_report (dict): Dictionary containing health status of components
                                 Keys: "RWA_1", "RWA_2", "RWA_3", "RWA_4", "SRU_1", "SRU_2"
                                 Values: Boolean health status (True=healthy, False=faulty)
            target_orientation (np.array): Target orientation quaternion [4x1, scalar first]
            target_angular_rate (np.array): Target angular rate [3x1, rad/s]
            is_safe_mode (bool): Flag indicating if system is in safe mode
            
        Configs:
            orbital_rate: Orbital angular rate in rad/s (default for LEO ~0.001027 rad/s)
            safe_mode_vector: Fixed inertial vector for safe mode [x, y, z]
                             Default is [0, 0, -1] (Earth-pointing)
            min_reaction_wheels: Minimum number of healthy reaction wheels (default: 3)
            min_star_trackers: Minimum number of healthy star trackers (default: 1)
        """
        self._i = NodeAutonomyInputs(
            InputPort("health_report", self),
        )
        
        self._o = NodeAutonomyOutputs(
            OutputPort("target_orientation", self),
            OutputPort("target_angular_rate", self),
            OutputPort("is_safe_mode", self),
        )
        
        super().__init__(self._i, self._o, **kwargs)
    
    def initialize(self):
        """Initialize the autonomy node configuration."""
        self.orbital_rate = self._config.get("orbital_rate", 0.001027)
        safe_mode_vec = self._config.get("safe_mode_vector", [0, 0, -1])
        self.safe_mode_vector = np.array(safe_mode_vec) / np.linalg.norm(safe_mode_vec)
        
        # Health monitoring thresholds
        self.min_reaction_wheels = self._config.get("min_reaction_wheels", 3)
        self.min_star_trackers = self._config.get("min_star_trackers", 1)
        
        # State variables
        self.is_safe_mode = False
        self.nadir_angle = 0.0
    
    def update(self, sim_time: float):
        """
        Main update function for the autonomy node.
        
        Args:
            sim_time: Current simulation time in seconds
        """
        # Read health report dictionary from input port
        health_report = self._i.health_report.read()
        
        # Handle None input (assume all healthy if no data)
        if health_report is None:
            health_report = {
                "RWA_1": True,
                "RWA_2": True,
                "RWA_3": True,
                "RWA_4": True,
                "SRU_1": True,
                "SRU_2": True
            }
        
        # Extract health status for each component (default to healthy if missing)
        rw1_health = health_report.get("RWA_1", True)
        rw2_health = health_report.get("RWA_2", True)
        rw3_health = health_report.get("RWA_3", True)
        rw4_health = health_report.get("RWA_4", True)
        st1_health = health_report.get("SRU_1", True)
        st2_health = health_report.get("SRU_2", True)
        
        # Count healthy components
        rw_active = sum([rw1_health, rw2_health, rw3_health, rw4_health])
        st_active = sum([st1_health, st2_health])
        
        # Determine if we should be in safe mode
        should_be_safe_mode = (rw_active < self.min_reaction_wheels or 
                              st_active < self.min_star_trackers)
        
        # Update safe mode status
        self.is_safe_mode = should_be_safe_mode
        
        # Generate appropriate trajectory based on mode
        if self.is_safe_mode:
            orientation_quat, angular_rate_vector = self._generate_safe_mode_trajectory()
        else:
            orientation_quat, angular_rate_vector = self._generate_nadir_pointing_trajectory(sim_time)
        
        # Output the trajectory commands
        self._o.target_orientation.shift_out(orientation_quat)
        self._o.target_angular_rate.shift_out(angular_rate_vector)
        self._o.is_safe_mode.shift_out(self.is_safe_mode)
    
    def _generate_nadir_pointing_trajectory(self, time: float):
        """
        Generate a nadir pointing trajectory that rotates with orbital motion.
        
        Args:
            time: Current simulation time in seconds
            
        Returns:
            Tuple of (orientation_quaternion, angular_rate_vector)
        """
        # Update nadir angle based on orbital motion
        self.nadir_angle = self.orbital_rate * time
        
        # Generate nadir pointing vector (simplified orbital mechanics)
        # Assuming circular orbit with nadir pointing in -Z direction
        # and rotation about the orbit normal (Y-axis)
        cos_angle = np.cos(self.nadir_angle)
        sin_angle = np.sin(self.nadir_angle)
        
        # Create rotation matrix for nadir pointing
        # Rotation about Y-axis to maintain nadir pointing
        rotation_matrix = np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ])
        
        # Convert to quaternion (scalar first format)
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rotation_matrix)
        quat_xyzw = r.as_quat()  # [x, y, z, w] format
        orientation_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
        
        # Angular rate vector (constant orbital rate about Y-axis)
        angular_rate_vector = np.array([0, self.orbital_rate, 0])
        
        return orientation_quat, angular_rate_vector
    
    def _generate_safe_mode_trajectory(self):
        """
        Generate safe mode orientation and angular rate.
        
        Returns:
            Tuple of (orientation_quaternion, angular_rate_vector)
        """
        # Fixed inertial quaternion pointing in safe mode direction
        # Convert safe mode vector to quaternion
        # Assume safe mode vector points in -Z direction initially
        target_vector = self.safe_mode_vector
        z_axis = np.array([0, 0, 1])
        
        # Calculate rotation quaternion to align z_axis with target_vector
        if np.allclose(target_vector, -z_axis):
            # 180 degree rotation about x-axis
            orientation_quat = np.array([0, 1, 0, 0])  # [w, x, y, z]
        elif np.allclose(target_vector, z_axis):
            # No rotation needed
            orientation_quat = np.array([1, 0, 0, 0])  # [w, x, y, z]
        else:
            # General case: find rotation quaternion
            cross_product = np.cross(z_axis, target_vector)
            dot_product = np.dot(z_axis, target_vector)
            
            if np.linalg.norm(cross_product) < 1e-6:
                # Vectors are parallel or anti-parallel
                orientation_quat = np.array([1, 0, 0, 0])  # [w, x, y, z]
            else:
                # Calculate quaternion from axis-angle
                angle = np.arccos(np.clip(dot_product, -1, 1))
                axis = cross_product / np.linalg.norm(cross_product)
                
                w = np.cos(angle / 2)
                xyz = axis * np.sin(angle / 2)
                orientation_quat = np.array([w, xyz[0], xyz[1], xyz[2]])  # [w, x, y, z]
        
        # Zero angular rate for safe mode
        angular_rate_vector = np.array([0.0, 0.0, 0.0])
        
        return orientation_quat, angular_rate_vector
    
    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o


# Example usage for testing
if __name__ == "__main__":
    # This would be used for standalone testing
    pass