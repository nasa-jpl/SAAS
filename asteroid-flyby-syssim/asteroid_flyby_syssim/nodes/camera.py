"""Asteroid camera rendering using Mitsuba 3 and pyshtools shape models."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pyshtools as pysh
import mitsuba as mi
import drjit as dr
import struct
import platformdirs
import os
import pickle
from pathlib import Path
from datetime import datetime, timezone

from syssim.core import EmptySpec, InputPort, Node, NodeParameter, OutputPort, input_port, output_port, parameter
from ..starfield_hdri import generate_starfield_hdri


_ASTEROID_VISIBILITY_INTEGRATOR_REGISTERED = False


def _ensure_asteroid_visibility_integrator_registered():
    """Register the Mitsuba integrator used for asteroid visibility masks."""
    global _ASTEROID_VISIBILITY_INTEGRATOR_REGISTERED
    if _ASTEROID_VISIBILITY_INTEGRATOR_REGISTERED:
        return

    class AsteroidVisibilityIntegrator(mi.SamplingIntegrator):
        """Mitsuba integrator that returns one for asteroid hits and zero otherwise."""

        def __init__(self, props=mi.Properties()):
            """Initialize the visibility integrator.

            Parameters
            ----------
            props : mitsuba.Properties, optional
                Mitsuba property set forwarded by the renderer.
            """
            super().__init__(props)

        def sample(
            self,
            scene: mi.Scene,
            sampler: mi.Sampler,
            ray: mi.RayDifferential3f,
            medium: mi.Medium = None,
            active: bool = True,
        ) -> tuple[mi.Color3f, bool, list[float]]:
            """Sample a ray and return a binary asteroid-hit color.

            Parameters
            ----------
            scene : mitsuba.Scene
                Scene to intersect.
            sampler : mitsuba.Sampler
                Mitsuba sampler supplied by the renderer.
            ray : mitsuba.RayDifferential3f
                Camera ray to test against scene geometry.
            medium : mitsuba.Medium, optional
                Participating medium, unused by this integrator.
            active : bool, optional
                Mitsuba active mask for vectorized rendering.

            Returns
            -------
            tuple[mitsuba.Color3f, bool, list[float]]
                Binary hit color, validity flag, and empty auxiliary output list.
            """
            del sampler, medium

            ray = mi.Ray3f(ray)
            active = mi.Bool(active)

            pi: mi.PreliminaryIntersection3f = scene.ray_intersect_preliminary(ray)
            hit = active & pi.is_valid()

            mask_value = dr.select(hit, 1.0, 0.0)
            return mi.Color3f(mask_value), mi.Bool(True), []

    mi.register_integrator(
        "asteroid_visibility",
        lambda props: AsteroidVisibilityIntegrator(props),
    )
    _ASTEROID_VISIBILITY_INTEGRATOR_REGISTERED = True

try:
    from skyfield.api import Loader, wgs84
    from skyfield.data import mpc
    from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
    from skyfield.iokit import download
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False


# Global in-memory cache for asteroid shape models to avoid redundant loading
_ASTEROID_SHAPE_CACHE = {}

# Process-level cache for Skyfield resources keyed by cache directory.
_SKYFIELD_RESOURCE_CACHE: dict[str, dict[str, Any]] = {}
_SUN_DIRECTION_CACHE: dict[str, np.ndarray] = {}


def _get_shape_cache_dir() -> Path:
    """Get the system cache directory for asteroid shape models."""
    cache_dir = Path(platformdirs.user_cache_dir("asteroid-flyby-syssim", "saas"))
    shape_cache_dir = cache_dir / "asteroid_shape_models"
    shape_cache_dir.mkdir(parents=True, exist_ok=True)
    return shape_cache_dir


def _get_cached_shape_model(asteroid: str, lmax: int = 180):
    """Get or load asteroid shape model from cache (disk + memory).
    
    Caching strategy:
    1. Check in-memory cache first
    2. Check disk cache next
    3. Load from pyshtools and save to disk if not cached
    
    Parameters
    ----------
    asteroid : str
        Asteroid name ('Ceres', 'Vesta', or 'Eros').
    lmax : int
        Maximum degree of spherical harmonics expansion (default: 180).
    
    Returns
    -------
    shape_grid
        Expanded shape grid from pyshtools.
    mean_radius_m : float
        Mean radius of asteroid in meters.
    """
    cache_key = (asteroid, lmax)
    
    # Check in-memory cache first
    if cache_key in _ASTEROID_SHAPE_CACHE:
        return _ASTEROID_SHAPE_CACHE[cache_key]
    
    # Check disk cache
    cache_dir = _get_shape_cache_dir()
    cache_file = cache_dir / f"{asteroid}_lmax{lmax}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                shape_grid = cached_data["shape_grid"]
                mean_radius_m = cached_data["mean_radius_m"]
                _ASTEROID_SHAPE_CACHE[cache_key] = (shape_grid, mean_radius_m)
                return shape_grid, mean_radius_m
        except Exception as e:
            import logging
            logging.warning(f"Failed to load cached shape model from {cache_file}: {e}. Reloading from pyshtools.")
    
    # Load from pyshtools and save to cache
    if asteroid == "Ceres":
        shape_model = pysh.datasets.Ceres.DLR_SPG_shape(lmax=lmax)
    elif asteroid == "Vesta":
        shape_model = pysh.datasets.Vesta.DLR_SPG_shape(lmax=lmax)
    elif asteroid == "Eros":
        shape_model = pysh.datasets.Eros.NLR_shape(lmax=lmax)
    else:
        raise ValueError(f"Unknown asteroid: {asteroid}")
    
    shape_grid = shape_model.expand(grid="DH2")
    mean_radius_m = float(np.mean(shape_grid.data))
    
    # Save to disk cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump({
                "shape_grid": shape_grid,
                "mean_radius_m": mean_radius_m
            }, f)
    except Exception as e:
        import logging
        logging.warning(f"Failed to save shape model cache to {cache_file}: {e}")
    
    # Store in memory cache
    _ASTEROID_SHAPE_CACHE[cache_key] = (shape_grid, mean_radius_m)
    
    return shape_grid, mean_radius_m


def _load_mpc_dataframe(cache_dir: str, load: Any):
    """Load MPC minor-planet dataframe from cache dir once per process."""
    if not HAS_SKYFIELD:
        return None

    mpc_file = os.path.join(cache_dir, "MPCORB.DAT")
    if not os.path.exists(mpc_file):
        gz_path = mpc_file + ".gz"
        if not os.path.exists(gz_path):
            download(mpc.MPCORB_URL, path=gz_path)
        if os.path.exists(gz_path):
            import gzip
            with gzip.open(gz_path, "rb") as f_in:
                with open(mpc_file, "wb") as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_path)

        # Skip MPC header section; table parsing expects data rows.
        with open(mpc_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(mpc_file, "w", encoding="utf-8") as f:
            f.writelines(lines[43:])

    with load.open(mpc_file) as f:
        minor_planets = mpc.load_mpcorb_dataframe(f)

    bad_orbits = minor_planets.semimajor_axis_au.isnull()
    minor_planets = minor_planets[~bad_orbits]
    return minor_planets.set_index("designation", drop=False)


def _get_skyfield_resources(cache_dir: str) -> dict[str, Any]:
    """Get shared Skyfield resources for the given cache directory."""
    if not HAS_SKYFIELD:
        return {
            "load": None,
            "timescale": None,
            "ephemeris": None,
            "minor_planets": None,
        }

    cached = _SKYFIELD_RESOURCE_CACHE.get(cache_dir)
    if cached is not None:
        return cached

    load = Loader(cache_dir, verbose=False)
    resources = {
        "load": load,
        "timescale": load.timescale(),
        "ephemeris": load("de421.bsp"),
        "minor_planets": _load_mpc_dataframe(cache_dir=cache_dir, load=load),
    }
    _SKYFIELD_RESOURCE_CACHE[cache_dir] = resources
    return resources


@dataclass
class NodeAsteroidCameraInputs:
    """Input ports for asteroid camera rendering."""

    camera_position: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Camera position [x, y, z] in asteroid body frame (meters)."""
    camera_target: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Optional camera target [x, y, z] in asteroid body frame (meters)."""


@dataclass
class NodeAsteroidCameraOutputs:
    """Output ports for rendered asteroid products."""

    image: OutputPort[np.ndarray | None] = output_port(np.ndarray | None)
    """Rendered image as numpy array [height, width, channels]"""
    asteroid_mask: OutputPort[np.ndarray] = output_port(np.ndarray)
    """Binary semantic mask image identifying asteroid pixels."""
    asteroid_visible: OutputPort[bool] = output_port(bool)
    """Boolean indicating whether the asteroid intersects the frame."""


@dataclass
class NodeAsteroidCameraParameters:
    """Faultable camera and renderer parameters."""

    asteroid: NodeParameter[str] = parameter("Ceres", value_type=str)
    """Asteroid shape model name: Ceres, Vesta, or Eros."""
    resolution_width: NodeParameter[int] = parameter(512, value_type=int)
    """Rendered image width in pixels."""
    resolution_height: NodeParameter[int] = parameter(512, value_type=int)
    """Rendered image height in pixels."""
    fov: NodeParameter[float] = parameter(45.0, value_type=float)
    """Perspective camera field of view in degrees."""
    spp: NodeParameter[int] = parameter(32, value_type=int)
    """Samples per pixel for Mitsuba rendering."""
    look_at_origin: NodeParameter[bool] = parameter(True, value_type=bool)
    """Whether to use asteroid center as the fallback camera target."""
    scale_factor: NodeParameter[float] = parameter(0.001, value_type=float)
    """Scene scale factor applied to geometry and camera positions."""
    use_integrator_mask: NodeParameter[bool] = parameter(False, value_type=bool)
    """Whether to render masks with the Mitsuba visibility integrator."""


class NodeAsteroidCamera(
    Node[NodeAsteroidCameraInputs, NodeAsteroidCameraOutputs, NodeAsteroidCameraParameters, EmptySpec]
):
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
    FIBONACCI_SPHERE_POINTS = 15000

    Inputs = NodeAsteroidCameraInputs
    Outputs = NodeAsteroidCameraOutputs
    Parameters = NodeAsteroidCameraParameters

    @staticmethod
    def _select_mitsuba_variant() -> str:
        """Pick the best available Mitsuba variant, preferring GPU backends."""
        try:
            available = set(mi.variants())
        except Exception:
            available = set()

        preferred = (
            "cuda_ad_rgb",
            "cuda_rgb",
            "llvm_ad_rgb",
            "llvm_rgb",
            "scalar_rgb",
        )
        for variant in preferred:
            if variant in available:
                return variant

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
        use_integrator_mask: bool = False,
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
        use_integrator_mask : bool
            If True, compute mask/visibility with the old Mitsuba visibility integrator
            (slower, second render pass). If False, use geometric estimate (faster).
        """
        if asteroid not in self.ASTEROID_SHAPE_DATASETS:
            raise ValueError(
                f"Unknown asteroid '{asteroid}'. Choose from: {list(self.ASTEROID_SHAPE_DATASETS.keys())}"
            )

        super().__init__(**kwargs)
        self._i = self.i
        self._o = self.o
        self._p = self.p
        self.p.asteroid.set_nominal(asteroid)
        self.p.resolution_width.set_nominal(int(resolution_width))
        self.p.resolution_height.set_nominal(int(resolution_height))
        self.p.fov.set_nominal(float(fov))
        self.p.spp.set_nominal(int(spp))
        self.p.look_at_origin.set_nominal(bool(look_at_origin))
        self.p.scale_factor.set_nominal(float(scale_factor))
        self.p.use_integrator_mask.set_nominal(bool(use_integrator_mask))

        # Auto-select rendering backend: CUDA GPU if available, CPU otherwise.
        selected_variant = self._select_mitsuba_variant()
        mi.set_variant(selected_variant)
        _ensure_asteroid_visibility_integrator_registered()

        # Store date for sun position computation
        self._date = date

        self._cache_dir = platformdirs.user_cache_dir("syssim-smad", "saas")
        os.makedirs(self._cache_dir, exist_ok=True)

        # Load shared Skyfield resources once per process/cache directory.
        skyfield_resources = _get_skyfield_resources(self._cache_dir)
        self._load = skyfield_resources["load"]
        self._ts = skyfield_resources["timescale"]
        self._eph = skyfield_resources["ephemeris"]
        self._minor_planets = skyfield_resources["minor_planets"]
        
        # Compute sun direction in J2000 frame
        self._sun_direction = self._compute_sun_direction()

        # In inertial frame usage, reuse a shared cached starfield asset.
        self._hdri_path = generate_starfield_hdri(output_dir=self._cache_dir)

        # Load shape model and create mesh
        self._load_shape_model()
        self._create_mitsuba_scene()

    def _load_shape_model(self):
        """Load asteroid shape model from pyshtools (cached)."""
        asteroid = self._p.asteroid.value
        self._shape_lmax = 180
        self._shape_grid, self._mean_radius_m = _get_cached_shape_model(asteroid, lmax=self._shape_lmax)

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
        # Number of points tuned for RL throughput: enough fidelity for training,
        # substantially less startup mesh generation cost than the previous default.
        n_points = self.FIBONACCI_SPHERE_POINTS
        
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
        # Sample the cached DH2 grid directly (nearest-neighbor lookup) to avoid
        # repeated spherical-harmonic expansions at startup.
        grid = np.asarray(self._shape_grid.data, dtype=np.float64)
        n_lat, n_lon = grid.shape
        lat_idx = np.rint((90.0 - lat) / 180.0 * (n_lat - 1)).astype(np.int64)
        lon_idx = np.rint((lon % 360.0) / 360.0 * (n_lon - 1)).astype(np.int64)
        lat_idx = np.clip(lat_idx, 0, n_lat - 1)
        lon_idx = np.mod(lon_idx, n_lon)
        radii = grid[lat_idx, lon_idx]
        
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
        lmax = int(self._shape_lmax)
        ply_filename = f"asteroid_{asteroid.lower()}_lmax{lmax}_binary_v2.ply"
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
        with open(ply_path, 'wb') as f:
            # Write PLY header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"comment Asteroid: {asteroid}\n".encode("ascii"))
            f.write(f"comment Shape model lmax: {lmax}\n".encode("ascii"))
            f.write(f"element vertex {len(positions_np)}\n".encode("ascii"))
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")
            f.write(f"element face {len(faces_np)}\n".encode("ascii"))
            f.write(b"property list uchar int vertex_indices\n")
            f.write(b"end_header\n")
            
            # Write vertices with normals
            vertex_data = np.column_stack((positions_np, normals_np)).astype(np.float32, copy=False)
            f.write(vertex_data.tobytes(order="C"))
            
            # Write faces (PLY uses 0-based indexing, prepend count)
            for face in faces_np:
                f.write(struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2])))
        
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
            return _load_mpc_dataframe(cache_dir=self._cache_dir, load=self._load)
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
        asteroid_name = self._p.asteroid.value
        cached_sun_direction = _SUN_DIRECTION_CACHE.get(asteroid_name)
        if cached_sun_direction is not None:
            return cached_sun_direction.copy()

        if not HAS_SKYFIELD or self._minor_planets is None or self._ts is None or self._eph is None:
            # Fallback to default direction
            print("Warning: MPC data unavailable. Using default sun direction.")
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        try:
            # Convert datetime to skyfield time object
            t = self._ts.from_datetime(self._date)
            
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
            sun_direction = sun_direction.astype(np.float32)
            _SUN_DIRECTION_CACHE[asteroid_name] = sun_direction.copy()
            return sun_direction
            
        except Exception as e:
            print(f"Warning: Failed to compute sun direction: {e}")
            import traceback
            traceback.print_exc()
            print("Using default sun direction.")
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

    
    def _create_mitsuba_scene(self):
        """Create and cache a base Mitsuba scene."""
        # Save mesh to PLY file in cache (checks for existing file first)
        self._save_mesh_as_ply()
        default_camera_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        default_camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._scene = self._build_scene(default_camera_pos, default_camera_target, integrator_type="path")
        self._scene_params = mi.traverse(self._scene)

    def _build_scene(self, camera_pos, camera_target, integrator_type: str = "path"):
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
        if integrator_type == "path":
            integrator = {
                "type": "path",
                "max_depth": 8,  # Maximum path depth for ray tracing
            }
        else:
            integrator = {"type": integrator_type}

        scene_dict = {
            "type": "scene",
            # Integrator for rendering
            "integrator": integrator,
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

    def _update_scene_camera(self, camera_pos: np.ndarray, camera_target: np.ndarray):
        """Update camera transform of the cached scene instead of rebuilding it."""
        transform = mi.ScalarTransform4f.look_at(
            origin=np.array(camera_pos, dtype=np.float32),
            target=np.array(camera_target, dtype=np.float32),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self._scene_params["camera.to_world"] = transform
        self._scene_params.update()

    @staticmethod
    def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Return a unit vector with zero fallback.

        Parameters
        ----------
        v : np.ndarray
            Vector to normalize.
        eps : float, optional
            Minimum norm treated as nonzero.

        Returns
        -------
        np.ndarray
            Normalized vector, or zeros with the input shape when the norm is small.
        """
        n = np.linalg.norm(v)
        if n < eps:
            return np.zeros_like(v)
        return v / n

    def _estimate_visibility_and_mask(
        self,
        camera_pos_scaled: np.ndarray,
        camera_target_scaled: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Estimate asteroid visibility and binary mask from camera geometry.

        This avoids a second full render pass by using projected angular size.
        """
        width = int(self._p.resolution_width.value)
        height = int(self._p.resolution_height.value)
        mask = np.zeros((height, width), dtype=np.uint8)

        forward = self._safe_normalize(camera_target_scaled - camera_pos_scaled)
        to_center = -camera_pos_scaled
        dist = float(np.linalg.norm(to_center))
        if dist <= 1e-9:
            return mask, False

        center_dir = to_center / dist
        scale = float(self._p.scale_factor.value)
        radius_scaled = max(float(self._mean_radius_m) * scale, 1e-9)
        if dist <= radius_scaled:
            return np.full((height, width), 255, dtype=np.uint8), True

        angular_radius = float(np.arcsin(np.clip(radius_scaled / dist, 0.0, 1.0)))
        half_fov = np.deg2rad(float(self._p.fov.value) * 0.5)
        center_angle = float(np.arccos(np.clip(np.dot(forward, center_dir), -1.0, 1.0)))
        visible = center_angle <= (half_fov + angular_radius)
        if not visible:
            return mask, False

        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up_hint)
        if np.linalg.norm(right) < 1e-9:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        right = self._safe_normalize(right)
        up = self._safe_normalize(np.cross(right, forward))

        z = float(np.dot(center_dir, forward))
        if z <= 1e-9:
            return mask, False
        x = float(np.dot(center_dir, right) / z)
        y = float(np.dot(center_dir, up) / z)

        fx = 0.5 * width / np.tan(half_fov)
        fy = fx
        cx = 0.5 * width
        cy = 0.5 * height

        u = cx + fx * x
        v = cy - fy * y
        r_px = max(1.0, fx * np.tan(angular_radius) / z)

        yy, xx = np.ogrid[:height, :width]
        disk = (xx - u) ** 2 + (yy - v) ** 2 <= r_px ** 2
        mask[disk] = 255
        return mask, bool(disk.any())

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
        camera_pos = self.i.camera_position.read().value

        if camera_pos is None or np.any(np.isnan(camera_pos)):
            # No valid camera position, output None
            self.o.image.write(None, sim_time)
            return

        # Apply scene scaling to camera position
        camera_pos_scaled = camera_pos * self._p.scale_factor.value

        # Determine camera target from input if present, otherwise fallback.
        camera_target_in = self.i.camera_target.read().value
        if camera_target_in is not None and not np.any(np.isnan(camera_target_in)):
            camera_target = np.array(camera_target_in, dtype=float) * self._p.scale_factor.value
        elif self._p.look_at_origin.value:
            camera_target = np.array([0.0, 0.0, 0.0])
        else:
            # Backward compatible fallback.
            camera_target = np.array([0.0, 0.0, 0.0])

        # Reuse the cached scene by updating only camera transform.
        self._update_scene_camera(camera_pos_scaled, camera_target)

        # Render RGB once.
        image = mi.render(self._scene)

        # Convert to numpy array
        image_np = np.array(image)

        # Clip and convert to 8-bit
        image_np = np.clip(image_np, 0, 1)
        image_8bit = (image_np * 255).astype(np.uint8)

        if self._p.use_integrator_mask.value:
            # Optional compatibility path: render semantic mask with the old integrator.
            mask_scene = self._build_scene(
                camera_pos_scaled,
                camera_target,
                integrator_type="asteroid_visibility",
            )
            mask = mi.render(mask_scene)
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            mask_np = np.clip(mask_np, 0, 1)
            mask_8bit = (mask_np * 255).astype(np.uint8)
            asteroid_visible = bool(np.any(mask_np > 0.0))
        else:
            # Fast path: estimate visibility/mask from geometry.
            mask_8bit, asteroid_visible = self._estimate_visibility_and_mask(
                camera_pos_scaled=np.asarray(camera_pos_scaled, dtype=np.float64),
                camera_target_scaled=np.asarray(camera_target, dtype=np.float64),
            )

        # Output the rendered image
        self.o.image.write(image_8bit, sim_time)
        self.o.asteroid_mask.write(mask_8bit, sim_time)
        self.o.asteroid_visible.write(asteroid_visible, sim_time)
