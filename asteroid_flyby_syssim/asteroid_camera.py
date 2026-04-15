"""Asteroid camera rendering using Mitsuba 3 and pyshtools shape models."""

from typing import NamedTuple
import numpy as np
import pyshtools as pysh
import mitsuba as mi
import matplotlib.pyplot as plt
import drjit as dr
import platformdirs
import os
from pathlib import Path
from datetime import datetime, timezone

from syssim.core import Node, InputPort, OutputPort
from syssim.core.node import NodeParameter
from .starfield_hdri import generate_starfield_hdri

try:
    from skyfield.api import Loader, wgs84
    from skyfield.data import mpc
    from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
    from skyfield.iokit import download
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False


class NodeAsteroidCameraInputs(NamedTuple):
    camera_position: InputPort
    """Camera position [x, y, z] in asteroid body frame (meters)."""
    camera_target: InputPort
    """Optional camera target [x, y, z] in asteroid body frame (meters)."""


class NodeAsteroidCameraOutputs(NamedTuple):
    image: OutputPort
    """Rendered image as numpy array [height, width, channels]"""


class NodeAsteroidCamera(Node):
    """Render asteroid from a spacecraft-mounted camera using Mitsuba 3.

    This node renders a view of an asteroid from a camera position using
    Mitsuba 3 ray tracing and shape models from pyshtools.

    Supported asteroids: Ceres, Vesta, Eros
    """

    # Mapping of asteroid names to pyshtools shape datasets
    ASTEROID_SHAPE_DATASETS = {
        "Ceres": "DLR_SPG_shape",  # DLR stereo-photogrammetric shape
        "Vesta": "DLR_SPG_shape",  # DLR stereo-photogrammetric shape
        "Eros": "NLR_shape",  # Laser altimeter shape
    }

    class Parameters(NamedTuple):
        asteroid: NodeParameter
        resolution_width: NodeParameter
        resolution_height: NodeParameter
        fov: NodeParameter
        spp: NodeParameter  # samples per pixel
        look_at_origin: NodeParameter
        scale_factor: NodeParameter

    @staticmethod
    def _select_mitsuba_variant() -> str:
        """Pick the best available Mitsuba variant, preferring GPU backends."""
        # try:
        #     available = set(mi.variants())
        # except Exception:
        #     available = set()

        # # Prefer CUDA when present, then LLVM CPU JIT, then scalar fallback.
        # preferred = (
        #     "cuda_ad_rgb",
        #     "cuda_rgb",
        #     "llvm_ad_rgb",
        #     "llvm_rgb",
        #     "scalar_rgb",
        # )
        # for variant in preferred:
        #     if variant in available:
        #         return variant

        # Conservative fallback if variant enumeration fails.
        return "llvm_ad_rgb"

    def __init__(
        self,
        asteroid: str = "Ceres",
        resolution_width: int = 512,
        resolution_height: int = 512,
        fov: float = 45.0,
        spp: int = 32,
        look_at_origin: bool = True,
        scale_factor: float = 0.001,
        date: datetime = datetime.now(timezone.utc),
        **kwargs,
    ):
        """Initialize asteroid camera node.

        Parameters
        ----------
        asteroid : str
            Name of the asteroid ('Ceres', 'Vesta', or 'Eros')
        resolution_width : int
            Image width in pixels (default: 512)
        resolution_height : int
            Image height in pixels (default: 512)
        fov : float
            Field of view in degrees (default: 45.0)
        spp : int
            Samples per pixel for ray tracing (default: 32)
        look_at_origin : bool
            If True, camera always looks at asteroid center (default: True)
        scale_factor : float
            Scale factor for the entire scene (default: 1.0). Values < 1.0 scale down,
            values > 1.0 scale up. Affects asteroid size, camera positions, and clipping planes.
        """
        if asteroid not in self.ASTEROID_SHAPE_DATASETS:
            raise ValueError(
                f"Unknown asteroid '{asteroid}'. Choose from: {list(self.ASTEROID_SHAPE_DATASETS.keys())}"
            )

        self._i = NodeAsteroidCameraInputs(
            InputPort("camera_position", self),
            InputPort("camera_target", self),
        )
        self._o = NodeAsteroidCameraOutputs(OutputPort("image", self))

        self._p = self.Parameters(
            NodeParameter("asteroid", asteroid),
            NodeParameter("resolution_width", int(resolution_width)),
            NodeParameter("resolution_height", int(resolution_height)),
            NodeParameter("fov", float(fov)),
            NodeParameter("spp", int(spp)),
            NodeParameter("look_at_origin", bool(look_at_origin)),
            NodeParameter("scale_factor", float(scale_factor)),
        )

        # Auto-select rendering backend: CUDA GPU if available, CPU otherwise.
        selected_variant = self._select_mitsuba_variant()
        mi.set_variant(selected_variant)

        # Store date for sun position computation
        self._date = date

        self._cache_dir = platformdirs.user_cache_dir("syssim-smad", "saas")
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Load skyfield data
        self._load = Loader(self._cache_dir)
        self._ts = self._load.timescale()
        self._eph = self._load('de421.bsp')
        
        # Load MPC minor planet data
        self._minor_planets = self._load_mpc_data()
        
        # Compute sun direction in J2000 frame
        self._sun_direction = self._compute_sun_direction()

        # Generate/load starfield HDRI
        self._hdri_path = generate_starfield_hdri(output_dir=self._cache_dir, date=date)

        # Load shape model and create mesh
        self._load_shape_model()
        self._create_mitsuba_scene()

        super().__init__(self._i, self._o, self._p, **kwargs)

    def _load_shape_model(self):
        """Load asteroid shape model from pyshtools."""
        asteroid = self._p.asteroid.value

        # Load shape dataset with moderate resolution for rendering
        if asteroid == "Ceres":
            # Use moderate lmax for reasonable mesh size
            self._shape_model = pysh.datasets.Ceres.DLR_SPG_shape(lmax=180)
        elif asteroid == "Vesta":
            self._shape_model = pysh.datasets.Vesta.DLR_SPG_shape(lmax=180)
        elif asteroid == "Eros":
            self._shape_model = pysh.datasets.Eros.NLR_shape(lmax=180)

        # Expand to grid
        self._shape_grid = self._shape_model.expand(grid="DH2")

    def _create_mesh_from_shape(self):
        """Create a triangle mesh from the spherical-harmonic shape grid
        using Fibonacci sphere sampling for uniform point distribution.

        Returns
        -------
        positions : np.ndarray
            Vertex positions array of shape (N, 3).
        normals   : np.ndarray
            Per-vertex normals array of shape (N, 3).
        faces     : np.ndarray
            Triangle vertex indices array of shape (M, 3).
        """
        # ------------------------------------------------------------------
        # 1️⃣  Generate Fibonacci sphere points
        # ------------------------------------------------------------------
        # Number of points (tune this for resolution vs performance)
        # Using ~10k points gives good resolution without being too heavy
        n_points = 100000
        
        indices = np.arange(0, n_points, dtype=np.float64)  # Use float64 for pyshtools
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
        
        # Spherical coordinates on unit sphere
        theta = phi * indices  # Azimuthal angle
        z = 1 - (indices / float(n_points - 1)) * 2  # Height from -1 to 1
        radius_unit = np.sqrt(1 - z * z)  # Radius at height z
        
        x_unit = np.cos(theta) * radius_unit
        y_unit = np.sin(theta) * radius_unit
        z_unit = z
        
        # Convert to lat/lon for querying shape model
        lat = np.degrees(np.arcsin(z_unit))
        lon = np.degrees(np.arctan2(y_unit, x_unit)) % 360.0
        
        # ------------------------------------------------------------------
        # 2️⃣  Query shape model at each point
        # ------------------------------------------------------------------
        # Evaluate shape model at the Fibonacci points (convert to float for pyshtools)
        radii = np.zeros(n_points, dtype=np.float64)
        for i in range(n_points):
            radii[i] = self._shape_model.expand(lat=float(lat[i]), lon=float(lon[i]))
        
        # ------------------------------------------------------------------
        # 3️⃣  Convert to Cartesian coordinates with actual radii
        # ------------------------------------------------------------------
        positions = np.stack([
            radii * x_unit,
            radii * y_unit,
            radii * z_unit
        ], axis=1).astype(np.float32)  # Convert to float32 for rendering
        
        # ------------------------------------------------------------------
        # 4️⃣  Build Delaunay triangulation on the sphere
        # ------------------------------------------------------------------
        from scipy.spatial import ConvexHull
        
        # Project points back to unit sphere for triangulation
        points_unit = np.stack([x_unit, y_unit, z_unit], axis=1)
        
        # Use ConvexHull to triangulate (works well for sphere)
        hull = ConvexHull(points_unit)
        faces = hull.simplices.astype(np.uint32)
        
        # Ensure correct winding order (counter-clockwise)
        # Check if face normals point outward
        for i, face in enumerate(faces):
            v0, v1, v2 = positions[face]
            face_center = (v0 + v1 + v2) / 3.0
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # If normal points inward (dot product with center < 0), flip winding
            if np.dot(face_normal, face_center) < 0:
                faces[i] = face[[0, 2, 1]]  # Swap to reverse winding
        
        # ------------------------------------------------------------------
        # 5️⃣  Compute per-vertex normals (smooth shading)
        # ------------------------------------------------------------------
        nverts = positions.shape[0]
        normals = np.zeros((nverts, 3), dtype=np.float32)
        
        # Accumulate face normals onto vertices
        for face in faces:
            i0, i1, i2 = face
            p0, p1, p2 = positions[i0], positions[i1], positions[i2]
            
            # Compute face normal
            edge1 = p1 - p0
            edge2 = p2 - p0
            face_normal = np.cross(edge1, edge2)
            
            # Accumulate
            normals[i0] += face_normal
            normals[i1] += face_normal
            normals[i2] += face_normal
        
        # Normalize all vertex normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)  # avoid division by zero

        # ------------------------------------------------------------------
        # 6️⃣  Return NumPy arrays
        # ------------------------------------------------------------------
        return positions, normals, faces



    def _save_mesh_as_ply(self):
        """Save asteroid mesh as Stanford PLY file in system cache.
        
        Returns
        -------
        str
            Path to the saved PLY file
        """
        # Get cache directory
        cache_dir = platformdirs.user_cache_dir("syssim-smad", "saas")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create filename based on asteroid and shape model resolution
        asteroid = self._p.asteroid.value
        lmax = self._shape_model.lmax
        ply_filename = f"asteroid_{asteroid.lower()}_lmax{lmax}.ply"
        ply_path = os.path.join(cache_dir, ply_filename)
        
        # Check if already cached - if so, skip mesh creation
        if os.path.exists(ply_path):
            self._ply_path = ply_path
            return ply_path
        
        # Create mesh from shape model (only if not cached)
        vertex_positions, vertex_normals, faces = self._create_mesh_from_shape()
        
        # Already numpy arrays from the refactored method
        positions_np = vertex_positions
        normals_np = vertex_normals
        faces_np = faces
        
        # Write PLY file
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Asteroid: {asteroid}\n")
            f.write(f"comment Shape model lmax: {lmax}\n")
            f.write(f"element vertex {len(positions_np)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write(f"element face {len(faces_np)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices with normals
            for pos, norm in zip(positions_np, normals_np):
                f.write(f"{pos[0]} {pos[1]} {pos[2]} {norm[0]} {norm[1]} {norm[2]}\n")
            
            # Write faces (PLY uses 0-based indexing, prepend count)
            for face in faces_np:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        self._ply_path = ply_path
        return ply_path
    
    def _load_mpc_data(self):
        """Load MPC minor planet orbital data and return indexed dataframe.
        
        Returns
        -------
        dict or None
            Dictionary with asteroid names as keys and orbital data rows as values,
            or None if loading fails.
        """
        if not HAS_SKYFIELD:
            return None
        
        try:
            print("  Loading MPC minor planet data...")
            # Check if MPCORB.DAT is already in the cache
            mpc_file = os.path.join(self._cache_dir, "MPCORB.DAT")
            if not os.path.exists(mpc_file):
                print(f"  MPC data not cached. Downloading to {self._cache_dir}")
                # Unzip the .gz file that was downloaded
                gz_path = mpc_file + ".gz"
                if not os.path.exists(gz_path):
                    download(mpc.MPCORB_URL, path=gz_path)  # Ensure data is downloaded to cache
                if os.path.exists(gz_path):
                    import gzip
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(mpc_file, 'wb') as f_out:
                            f_out.write(f_in.read())
                    os.remove(gz_path)  # Remove the .gz file after extraction
                # remove the first 43 lines which are not data
                with open(mpc_file, 'r') as f:
                    lines = f.readlines()
                with open(mpc_file, 'w') as f:
                    f.writelines(lines[43:])

            with self._load.open(mpc_file) as f:
                minor_planets = mpc.load_mpcorb_dataframe(f)
            
            # Filter out orbits with missing data
            bad_orbits = minor_planets.semimajor_axis_au.isnull()
            minor_planets = minor_planets[~bad_orbits]
            
            # Index by designation for fast lookup
            minor_planets = minor_planets.set_index('designation', drop=False)
            
            return minor_planets
        except Exception as e:
            print(f"Warning: Failed to load MPC data: {e}")
            return None
    
    def _compute_sun_direction(self):
        """Compute sun direction in J2000 frame at the given datetime.
        
        Calculates the direction from the asteroid to the sun by:
        1. Loading the asteroid's orbit from MPC MPCORB data
        2. Computing the asteroid's heliocentric position using Kepler orbits
        3. Getting the sun's heliocentric position from the ephemeris
        4. Computing the vector from asteroid to sun
        
        Returns
        -------
        np.ndarray
            Normalized sun direction vector [x, y, z] in J2000 frame.
            Points toward the sun (light comes from this direction).
        """
        if not HAS_SKYFIELD or self._minor_planets is None:
            # Fallback to default direction
            print("Warning: MPC data unavailable. Using default sun direction.")
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        try:
            # Convert datetime to skyfield time object
            t = self._ts.from_datetime(self._date)
            
            # Get asteroid name
            asteroid_name = self._p.asteroid.value
            
            # Construct the MPC designation for the asteroid
            mpc_designations = {
                "Ceres": "(1) Ceres",
                "Vesta": "(4) Vesta",
                "Eros": "(433) Eros",
            }
            
            mpc_designation = mpc_designations.get(asteroid_name)
            if mpc_designation is None:
                print(f"Warning: No MPC designation for asteroid {asteroid_name}")
                return np.array([0.0, 0.0, -1.0], dtype=np.float32)
            
            print(f"  Computing position for {asteroid_name} ({mpc_designation})...")
            
            # Look up the asteroid's orbital data
            try:
                asteroid_row = self._minor_planets.loc[mpc_designation]
            except KeyError:
                print(f"Warning: {mpc_designation} not found in MPC database")
                return np.array([0.0, 0.0, -1.0], dtype=np.float32)
            
            # Get positions
            sun = self._eph['sun']
            
            # Compute asteroid position using Kepler orbits (heliocentric)
            asteroid = sun + mpc.mpcorb_orbit(asteroid_row, self._ts, GM_SUN)
            
            # Get sun position relative to solar system barycenter
            sun_astrometric = sun.at(t)
            sun_pos = sun_astrometric.position.au
            
            # Get asteroid position
            asteroid_astrometric = asteroid.at(t)
            asteroid_pos = asteroid_astrometric.position.au
            
            # Calculate vector from asteroid to sun
            sun_direction_vec = sun_pos - asteroid_pos
            
            # Normalize to unit vector
            magnitude = np.linalg.norm(sun_direction_vec)
            if magnitude < 1e-6:
                # Fallback if something goes wrong
                print("Warning: Invalid sun direction calculation, using default.")
                return np.array([0.0, 0.0, -1.0], dtype=np.float32)
            
            sun_direction = sun_direction_vec / magnitude
            
            return sun_direction.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to compute sun direction: {e}")
            import traceback
            traceback.print_exc()
            print("Using default sun direction.")
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

    
    def _create_mitsuba_scene(self):
        """Create base Mitsuba scene by saving mesh as PLY file.""" 
        # Save mesh to PLY file in cache (checks for existing file first)
        self._save_mesh_as_ply()

    def _build_scene(self, camera_pos, camera_target):
        """Build complete Mitsuba scene with camera at given position."""
        # Camera transform
        camera_pos = np.array(camera_pos, dtype=np.float32)
        camera_target = np.array(camera_target, dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Compute camera basis
        forward = camera_target - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Create transformation matrix
        transform = mi.ScalarTransform4f.look_at(
            origin=camera_pos, target=camera_target, up=up
        )
        
        # Scale clipping planes according to scene scale
        scale = self._p.scale_factor.value
        near_clip = 100.0 * scale      # 100 meters scaled
        far_clip = 1e7 * scale         # 10,000 km scaled
        
        # Build scene dictionary
        scene_dict = {
            "type": "scene",
            # Integrator for rendering
            "integrator": {
                "type": "path",
                "max_depth": 8,  # Maximum path depth for ray tracing
            },
            # Camera
            "camera": {
                "type": "perspective",
                "fov": self._p.fov.value,
                "to_world": transform,
                "near_clip": near_clip,
                "far_clip": far_clip,
                "film": {
                    "type": "hdrfilm",
                    "width": self._p.resolution_width.value,
                    "height": self._p.resolution_height.value,
                    "rfilter": {"type": "gaussian"},
                },
                "sampler": {
                    "type": "independent",
                    "sample_count": self._p.spp.value,
                },
            },
            # Asteroid mesh loaded from PLY file
            "asteroid": {
                "type": "ply",
                "filename": self._ply_path,
                "face_normals": False,  # Use vertex normals (smooth shading)
                "to_world": mi.ScalarTransform4f.scale(scale),  # Apply scene scaling
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": 1.0},
                },
            },
            # Sun-like directional light
            "sun": {
                "type": "directional",
                "direction": self._sun_direction.tolist(),
                "irradiance": {"type": "rgb", "value": 1.0},
            },
            # Environment map with starfield HDRI
            "envmap": {
                "type": "envmap",
                "filename": self._hdri_path,
            },
        }

        return mi.load_dict(scene_dict)

    def initialize(self):
        """Initialize the node before simulation."""
        pass

    def update(self, sim_time: float):
        """Render asteroid from current camera position.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds
        """
        # Read camera position from input port
        camera_pos = self._i.camera_position.read()

        if camera_pos is None or np.any(np.isnan(camera_pos)):
            # No valid camera position, output None
            self._o.image.shift_out(None, sim_time)
            return

        # Apply scene scaling to camera position
        camera_pos_scaled = camera_pos * self._p.scale_factor.value

        # Determine camera target from input if present, otherwise fallback.
        camera_target_in = self._i.camera_target.read()
        if camera_target_in is not None and not np.any(np.isnan(camera_target_in)):
            camera_target = np.array(camera_target_in, dtype=float) * self._p.scale_factor.value
        elif self._p.look_at_origin.value:
            camera_target = np.array([0.0, 0.0, 0.0])
        else:
            # Backward compatible fallback.
            camera_target = np.array([0.0, 0.0, 0.0])

        # Build scene with camera
        scene = self._build_scene(camera_pos_scaled, camera_target)

        # Render
        image = mi.render(scene)

        # Convert to numpy array
        image_np = np.array(image)

        # Clip and convert to 8-bit
        image_np = np.clip(image_np, 0, 1)
        image_8bit = (image_np * 255).astype(np.uint8)

        # Output the rendered image
        self._o.image.shift_out(image_8bit, sim_time)

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return self._o

    @property
    def p(self):
        return self._p


if __name__ == "__main__":
    """Test the asteroid camera node by rendering a single frame."""
    import random
    from syssim.core import NodeSystem
    from syssim.nodes.source import NodeConstant

    # Custom node to display rendered images
    class NodeImageDisplay(Node):
        class Inputs(NamedTuple):
            image: InputPort

        def __init__(self, title="Rendered Image", **kwargs):
            self._i = self.Inputs(InputPort("image", self))
            self._title = title
            self._image = None
            super().__init__(self._i, (), **kwargs)

        def update(self, sim_time: float):
            image = self._i.image.read()
            if image is not None:
                self._image = image

        def finalize(self, fault_history=None):
            if self._image is not None:
                print(f"Displaying image...")
                print(f"  Shape: {self._image.shape}")
                print(f"  Dtype: {self._image.dtype}")
                print(f"  Value range: [{self._image.min()}, {self._image.max()}]\n")

                plt.figure(figsize=(12, 9))
                plt.imshow(self._image)
                plt.title(self._title, fontsize=16, fontweight="bold")
                plt.axis("off")
                plt.tight_layout()
                plt.show()
            else:
                print("No image received for display!")

        @property
        def i(self):
            return self._i

        @property
        def o(self):
            return ()

    print("\n" + "=" * 60)
    print("Testing Asteroid Camera Node")
    print("=" * 60 + "\n")

    # Select a random asteroid
    asteroids = list(NodeAsteroidCamera.ASTEROID_SHAPE_DATASETS.keys())
    selected_asteroid = random.choice(asteroids)

    print(f"Selected asteroid: {selected_asteroid}")
    print("Loading shape model and setting up renderer...")

    # Create camera node
    camera_node = NodeAsteroidCamera(
        asteroid=selected_asteroid,
        resolution_width=800,
        resolution_height=600,
        fov=60.0,
        spp=64,
        look_at_origin=True,
        name="asteroid_camera",
    )

    # Get shape information
    r_mean = np.mean(camera_node._shape_grid.data)
    
    # Note: With Fibonacci sampling, we use a fixed number of points
    n_vertices = 10000  # As defined in _create_mesh_from_shape
    n_faces_approx = 2 * n_vertices - 4  # Approximate for closed convex hull
    
    print(f"Mean radius: {r_mean/1000:.1f} km")
    print(f"Shape model lmax: {camera_node._shape_model.lmax}")
    print(f"Mesh vertices: ~{n_vertices}")
    print(f"Mesh faces: ~{n_faces_approx}")
    print(f"PLY file: {camera_node._ply_path}\n")

    # Set camera position at 3x mean radius, viewing from angle
    camera_distance = 3.0 * r_mean
    camera_position = np.array(
        [
            camera_distance * np.cos(np.radians(30)),
            camera_distance * np.sin(np.radians(30)),
            camera_distance * 0.3,
        ]
    )

    print(
        f"Camera position: [{camera_position[0]/1000:.1f}, {camera_position[1]/1000:.1f}, {camera_position[2]/1000:.1f}] km"
    )
    print(f"Distance from center: {np.linalg.norm(camera_position)/1000:.1f} km")
    print(f"Field of view: {camera_node.p.fov.value}°")
    print(
        f"Resolution: {camera_node.p.resolution_width.value}x{camera_node.p.resolution_height.value}"
    )
    print(f"Samples per pixel: {camera_node.p.spp.value}\n")

    # Create position source node
    position_node = NodeConstant(value=camera_position, name="camera_position")

    # Create image display node
    display_node = NodeImageDisplay(
        title=f"Rendered View of {selected_asteroid}", name="image_display"
    )

    # Create system and add nodes
    system = NodeSystem()
    system.add_node(position_node)
    system.add_node(camera_node)
    system.add_node(display_node)

    # Connect nodes
    position_node.o.constant_out >> camera_node.i.camera_position
    camera_node.o.image >> display_node.i.image

    print("Rendering...")
    # Run simulation for single step
    system.simulate(t_f=0.1, dt=0.1)

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60 + "\n")
