"""Asteroid gravitational force calculation using pyshtools spherical harmonics."""

from typing import NamedTuple
import numpy as np
import pyshtools as pysh

from syssim.core import Node
from syssim.core import InputPort, OutputPort


class NodeAsteroidGravityInputs(NamedTuple):
    position: InputPort
    """Position vector [x, y, z] in asteroid body frame (meters)"""


class NodeAsteroidGravityOutputs(NamedTuple):
    gravity_accel: OutputPort
    """Gravitational acceleration vector [ax, ay, az] in asteroid body frame (m/s^2)"""


class NodeAsteroidGravity(Node):
    """Calculate gravitational force from an asteroid using spherical harmonics.
    
    This node computes gravitational acceleration and force at a given position
    relative to an asteroid center of mass using spherical harmonic coefficients
    from pyshtools built-in datasets.
    
    Supported asteroids: Ceres, Vesta, Eros
    """
    
    # Mapping of asteroid names to pyshtools dataset loaders
    ASTEROID_DATASETS = {
        'Ceres': 'CERES18D',  # JPL 18 degree gravity model
        'Vesta': 'VESTA20H',  # JPL 20 degree gravity model
        'Eros': 'JGE15A01',   # JPL 15 degree gravity model
    }
    
    def __init__(self, asteroid: str = 'Ceres', lmax: int = None, **kwargs):
        """Initialize asteroid gravity node.
        
        Parameters
        ----------
        asteroid : str
            Name of the asteroid ('Ceres', 'Vesta', or 'Eros')
        lmax : int, optional
            Maximum spherical harmonic degree to use. If None, uses the 
            maximum degree available in the dataset. Lower values may 
            improve performance at the cost of accuracy.
        """
        if asteroid not in self.ASTEROID_DATASETS:
            raise ValueError(f"Unknown asteroid '{asteroid}'. Choose from: {list(self.ASTEROID_DATASETS.keys())}")
        
        self._asteroid = asteroid
        self._lmax = lmax
        
        # Load gravity model from pyshtools datasets
        self._setup_gravity_model()
        
        self._i = NodeAsteroidGravityInputs(InputPort("position", self))
        self._o = NodeAsteroidGravityOutputs(
            OutputPort("gravity_accel", self)
        )
        
        super().__init__(self._i, self._o, **kwargs)
    
    def _setup_gravity_model(self):
        """Load spherical harmonic gravity model from pyshtools datasets."""
        # Load the appropriate dataset from pyshtools
        if self._lmax is not None:
            if self._asteroid == 'Ceres':
                self._gravity_model = pysh.datasets.Ceres.CERES18D(lmax=self._lmax)
            elif self._asteroid == 'Vesta':
                self._gravity_model = pysh.datasets.Vesta.VESTA20H(lmax=self._lmax)
            elif self._asteroid == 'Eros':
                self._gravity_model = pysh.datasets.Eros.JGE15A01(lmax=self._lmax)
        else:
            if self._asteroid == 'Ceres':
                self._gravity_model = pysh.datasets.Ceres.CERES18D()
            elif self._asteroid == 'Vesta':
                self._gravity_model = pysh.datasets.Vesta.VESTA20H()
            elif self._asteroid == 'Eros':
                self._gravity_model = pysh.datasets.Eros.JGE15A01()
        # Store the actual lmax being used
        self._max_degree = self._gravity_model.lmax
    
    def initialize(self):
        """Initialize the node before simulation."""
        pass
    
    def update(self, sim_time: float):
        """Compute gravitational force at the current position.
        
        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds
        """
        # Read position from input port
        position = self._i.position.read()
        
        if position is None or np.any(np.isnan(position)):
            # No valid position data
            self._o.gravity_accel.shift_out(np.array([0.0, 0.0, 0.0]), sim_time)
            return
        
        # Convert Cartesian position to spherical coordinates
        x, y, z = position
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Avoid singularity at origin
        if r < 1.0:  # Less than 1 meter from center
            self._o.gravity_accel.shift_out(np.array([0.0, 0.0, 0.0]), sim_time)
            return
        
        # Calculate spherical coordinates
        # lat: latitude (-90 to 90 degrees)
        # lon: longitude (0 to 360 degrees)
        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x))
        if lon < 0:
            lon += 360.0
        
        # Use pyshtools' built-in gravity method for point calculation
        # This calculates gravity at a single point (lat, lon, radius)
        # Returns vector components (r, theta, phi) in m/s^2
        omega = 0.0 if self._gravity_model.omega is None else self._gravity_model.omega
        a_r, a_theta, a_phi = pysh.gravmag.MakeGravGridPoint(
            self._gravity_model.coeffs,
            self._gravity_model.gm,
            self._gravity_model.r0,
            r,
            lat,
            lon,
            lmax=self._max_degree,
            # omega=omega
        )
        
        # Convert from spherical to Cartesian coordinates
        theta = np.radians(90.0 - lat)  # colatitude
        phi = np.radians(lon)  # longitude
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # Transformation matrix from spherical to Cartesian
        # Note: a_theta and a_phi are derivatives w.r.t. colatitude and longitude
        a_x = (a_r * sin_theta * cos_phi + 
               a_theta * cos_theta * cos_phi - 
               a_phi * sin_phi)
        a_y = (a_r * sin_theta * sin_phi + 
               a_theta * cos_theta * sin_phi + 
               a_phi * cos_phi)
        a_z = a_r * cos_theta - a_theta * sin_theta
        
        accel = np.array([-a_x, -a_y, -a_z])
        
        # Output the result
        self._o.gravity_accel.shift_out(accel, sim_time)
    
    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o


if __name__ == "__main__":
    """Test the asteroid gravity node with visualization using syssim framework."""
    import matplotlib.pyplot as plt
    import random
    from syssim.core import NodeSystem, Node
    from syssim.nodes.viz import NodeScope
    
    # Select a random asteroid
    asteroids = list(NodeAsteroidGravity.ASTEROID_DATASETS.keys())
    selected_asteroid = random.choice(asteroids)
    
    print(f"\n{'='*60}")
    print(f"Testing Asteroid Gravity Node: {selected_asteroid}")
    print(f"{'='*60}\n")
    
    # Create custom node that provides position along a circular orbit
    class NodeCircularOrbit(Node):
        class Outputs(NamedTuple):
            position: OutputPort
        
        def __init__(self, radius, angular_velocity=0.1, z_offset=0.0, **kwargs):
            """Generate position along a circular orbit.
            
            Parameters
            ----------
            radius : float
                Orbit radius in meters
            angular_velocity : float
                Angular velocity in rad/s (default: 0.1 rad/s)
            z_offset : float
                Fixed z-coordinate offset in meters
            """
            self._radius = radius
            self._omega = angular_velocity
            self._z_offset = z_offset
            self._o = self.Outputs(OutputPort("position", self))
            super().__init__((), self._o, **kwargs)
        
        def update(self, sim_time: float):
            theta = self._omega * sim_time
            x = self._radius * np.cos(theta)
            y = self._radius * np.sin(theta)
            z = self._z_offset
            position = np.array([x, y, z])
            self._o.position.shift_out(position)
        
        @property
        def i(self):
            return ()
        
        @property
        def o(self):
            return self._o
    
    # Create custom node to extract acceleration magnitude
    class NodeAccelMagnitude(Node):
        class Inputs(NamedTuple):
            accel: InputPort
        
        class Outputs(NamedTuple):
            magnitude: OutputPort
        
        def __init__(self, **kwargs):
            self._i = self.Inputs(InputPort("accel", self))
            self._o = self.Outputs(OutputPort("magnitude", self))
            super().__init__(self._i, self._o, **kwargs)
        
        def update(self, sim_time: float):
            accel = self._i.accel.read()
            if accel is not None:
                mag = np.linalg.norm(accel)
            else:
                mag = 0.0
            self._o.magnitude.shift_out(mag)
        
        @property
        def i(self):
            return self._i
        
        @property
        def o(self):
            return self._o
    
    # Create custom node to extract individual components
    class NodeVectorComponents(Node):
        class Inputs(NamedTuple):
            vector: InputPort
        
        class Outputs(NamedTuple):
            x: OutputPort
            y: OutputPort
            z: OutputPort
        
        def __init__(self, **kwargs):
            self._i = self.Inputs(InputPort("vector", self))
            self._o = self.Outputs(
                OutputPort("x", self),
                OutputPort("y", self),
                OutputPort("z", self)
            )
            super().__init__(self._i, self._o, **kwargs)
        
        def update(self, sim_time: float):
            vec = self._i.vector.read()
            if vec is not None and len(vec) >= 3:
                self._o.x.shift_out(vec[0])
                self._o.y.shift_out(vec[1])
                self._o.z.shift_out(vec[2])
            else:
                self._o.x.shift_out(0.0)
                self._o.y.shift_out(0.0)
                self._o.z.shift_out(0.0)
        
        @property
        def i(self):
            return self._i
        
        @property
        def o(self):
            return self._o

    # Create custom node to plot 3D orbit
    class NodeOrbit3DPlot(Node):
        class Inputs(NamedTuple):
            position: InputPort

        def __init__(self, **kwargs):
            self._i = self.Inputs(InputPort("position", self))
            self._positions = []
            super().__init__(self._i, (), **kwargs)

        def update(self, sim_time: float):
            pos = self._i.position.read()
            if pos is not None:
                self._positions.append(np.array(pos, dtype=float))

        def finalize(self, fault_history=None):
            if len(self._positions) == 0:
                return

            positions = np.array(self._positions)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Orbit trajectory")
            ax.scatter([0.0], [0.0], [0.0], c="red", s=50, label="Asteroid center")
            ax.set_title(f"{selected_asteroid}: 3D Orbit")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()

        @property
        def i(self):
            return self._i

        @property
        def o(self):
            return ()

    # Create custom node to plot acceleration components
    class NodeAccelComponentsPlot(Node):
        class Inputs(NamedTuple):
            accel: InputPort

        def __init__(self, **kwargs):
            self._i = self.Inputs(InputPort("accel", self))
            self._t = []
            self._ax = []
            self._ay = []
            self._az = []
            super().__init__(self._i, (), **kwargs)

        def update(self, sim_time: float):
            accel = self._i.accel.read()
            self._t.append(sim_time)
            if accel is not None and len(accel) >= 3:
                self._ax.append(float(accel[0]))
                self._ay.append(float(accel[1]))
                self._az.append(float(accel[2]))
            else:
                self._ax.append(0.0)
                self._ay.append(0.0)
                self._az.append(0.0)

        def finalize(self, fault_history=None):
            if len(self._t) == 0:
                return
            plt.figure()
            plt.plot(self._t, self._ax, label="a_x")
            plt.plot(self._t, self._ay, label="a_y")
            plt.plot(self._t, self._az, label="a_z")
            plt.xlabel("Time (s)")
            plt.ylabel("Acceleration (m/s²)")
            plt.title(f"{selected_asteroid}: Acceleration Components")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        @property
        def i(self):
            return self._i

        @property
        def o(self):
            return ()
    
    # Create the gravity node (this will load the dataset)
    print("Loading gravity model from pyshtools dataset...")
    grav_node = NodeAsteroidGravity(asteroid=selected_asteroid, sample_period=0.1, name="asteroid_gravity")
    
    # Get asteroid data from the loaded model
    r_ref = grav_node._gravity_model.r0
    gm = grav_node._gravity_model.gm
    mass = gm / 6.67430e-11  # Calculate mass from GM using gravitational constant
    
    print(f"Reference radius: {r_ref/1000:.1f} km")
    print(f"GM: {gm:.3e} m^3/s^2")
    print(f"Mass: {mass:.3e} kg")
    print(f"Max degree (lmax): {grav_node._max_degree}\n")
    
    # Create orbit at 2x the reference radius
    orbit_radius = 2.0 * r_ref
    orbit_period = 2 * np.pi * np.sqrt(orbit_radius**3 / gm)
    angular_velocity = 2 * np.pi / orbit_period
    
    print(f"Test orbit:")
    print(f"  Radius: {orbit_radius/1000:.1f} km ({orbit_radius/r_ref:.1f} asteroid radii)")
    print(f"  Period: {orbit_period:.1f} seconds ({orbit_period/60:.1f} minutes)")
    print(f"  Angular velocity: {angular_velocity:.6f} rad/s\n")
    
    # Create nodes
    orbit_node = NodeCircularOrbit(
        radius=orbit_radius,
        angular_velocity=angular_velocity,
        z_offset=0.0,
        name="circular_orbit"
    )
    
    accel_mag_node = NodeAccelMagnitude(name="accel_magnitude")
    position_components = NodeVectorComponents(name="position_components")
    accel_components = NodeVectorComponents(name="accel_components")
    orbit_plot_node = NodeOrbit3DPlot(name="orbit_3d_plot")
    accel_components_plot = NodeAccelComponentsPlot(name="accel_components_plot")
    
    # Create scope nodes for visualization
    scope_accel_mag = NodeScope(name="Acceleration Magnitude")
    scope_accel_mag._config = {
        "title": f"{selected_asteroid}: Gravitational Acceleration",
        "xlabel": "Time (s)",
        "ylabel": "Acceleration (m/s²)",
        "legend": ["|a|"],
        "show": True
    }
    
    # Create system and add nodes
    system = NodeSystem()
    system.add_node(orbit_node)
    system.add_node(grav_node)
    system.add_node(accel_mag_node)
    system.add_node(position_components)
    system.add_node(accel_components)
    system.add_node(orbit_plot_node)
    system.add_node(accel_components_plot)
    system.add_node(scope_accel_mag)
    
    # Connect nodes
    orbit_node.o.position >> grav_node.i.position
    grav_node.o.gravity_accel >> accel_mag_node.i.accel
    grav_node.o.gravity_accel >> accel_components.i.vector
    orbit_node.o.position >> position_components.i.vector
    orbit_node.o.position >> orbit_plot_node.i.position
    grav_node.o.gravity_accel >> accel_components_plot.i.accel
    
    # Connect to scopes
    accel_mag_node.o.magnitude >> scope_accel_mag.i.scope
    
    print("Starting simulation...")
    print(f"Simulating one complete orbit ({orbit_period:.1f} seconds)...\n")
    
    # Simulate for one complete orbit
    system.simulate(orbit_period, dt=0.1)
    
    print("\n=== Simulation Complete ===")
    print(f"Check the matplotlib windows for gravity field plots.")
    print(f"\nExpected surface gravity: {gm/r_ref**2:.3e} m/s²")
    print(f"Expected gravity at 2R: {gm/(4*r_ref**2):.3e} m/s² (1/4 of surface)")
    print(f"\n{'='*60}\n")
