"""Generate starfield HDRI from Hipparcos star catalog."""

import os
import numpy as np
from datetime import datetime
from pathlib import Path
import platformdirs
from rich.progress import track

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from skyfield.api import load
    from skyfield.iokit import Loader
    from skyfield.data import hipparcos
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False


# Configuration
STARFIELD_RESOLUTION = (4096, 2048)  # Equirectangular (longitude x latitude)
MIN_MAGNITUDE = 8.0
STAR_SIZE_FACTOR = 1.5
BRIGHTNESS_GAMMA = 2.2


def generate_starfield_hdri(
    resolution=STARFIELD_RESOLUTION,
    min_magnitude=MIN_MAGNITUDE,
    output_dir=None,
    date=None,
):
    """Generate starfield HDRI from Hipparcos catalog.
    
    Parameters
    ----------
    resolution : tuple
        Output resolution (width, height) in pixels
    min_magnitude : float
        Maximum star magnitude to include (brighter stars only)
    output_dir : str, optional
        Directory to save HDRI. If None, uses system cache.
    date : datetime, optional
        Date/time for star positions. If None, uses current date.
    
    Returns
    -------
    str
        Path to generated HDRI file (.exr or .png fallback)
    """
    if not HAS_SKYFIELD:
        raise ImportError(
            "Skyfield required for starfield generation. "
            "Install with: pip install skyfield"
        )
    
    if output_dir is None:
        output_dir = platformdirs.user_cache_dir("syssim-smad", "saas")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if date is None:
        date = datetime.now()
    
    # Generate filename based on parameters
    hdri_filename = (
        f"starfield_hdri_{resolution[0]}x{resolution[1]}_"
        f"mag{min_magnitude:.1f}.exr"
    )
    hdri_path = os.path.join(output_dir, hdri_filename)
    
    # Return cached version if it exists
    if os.path.exists(hdri_path):
        return hdri_path
    
    print("Generating starfield HDRI...")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Magnitude limit: {min_magnitude}")
    print(f"  Date: {date}")
    
    # Load ephemeris and Hipparcos catalog
    print("  Loading star catalog...")
    # ts = load.timescale()
    
    cache_dir = platformdirs.user_cache_dir("syssim-smad", "saas")
    os.makedirs(cache_dir, exist_ok=True)
    
    eph = Loader(cache_dir)('de421.bsp')
    
    # Download Hipparcos data to cache
    hipparcos_path = os.path.join(cache_dir, 'hipparcos.dat')
    with load.open(hipparcos.URL, filename=hipparcos_path) as f:
        hip_df = hipparcos.load_dataframe(f)
    
    # Filter by magnitude
    hip_df = hip_df[hip_df['magnitude'] <= min_magnitude]
    print(f"  Loaded {len(hip_df)} stars")
    
    # Create HDRI array
    hdri = np.zeros((resolution[1], resolution[0], 3), dtype=np.float32)
    
    # Render stars
    print("  Rendering stars...")
    for idx, star_data in track(
        hip_df.iterrows(),
        total=len(hip_df),
        description="  Rendering stars",
    ):
        try:
            # Equirectangular coordinates
            ra_deg = star_data['ra_degrees']
            dec_deg = star_data['dec_degrees']
            
            # Map to pixel coordinates
            x = int((ra_deg / 360.0) * resolution[0]) % resolution[0]
            y = int(((90 - dec_deg) / 180.0) * resolution[1])
            
            if 0 <= y < resolution[1]:
                # Calculate brightness from magnitude
                magnitude = star_data['magnitude']
                brightness = 10 ** ((-magnitude + 5) / 2.5)
                brightness = np.clip(brightness, 0, 100)
                brightness = brightness ** (1.0 / BRIGHTNESS_GAMMA)
                
                # Star color (simplified)
                if magnitude < 2.0:
                    color = np.array([0.9, 0.9, 1.0])  # Blue-white
                else:
                    color = np.array([1.0, 1.0, 1.0])  # White
                
                # Render star with gaussian falloff
                star_radius = int(max(1, STAR_SIZE_FACTOR * (1 + (5 - magnitude) * 0.2)))
                
                for dy in range(-star_radius, star_radius + 1):
                    for dx in range(-star_radius, star_radius + 1):
                        px = (x + dx) % resolution[0]
                        py = y + dy
                        
                        if 0 <= py < resolution[1]:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= star_radius:
                                falloff = np.exp(-dist**2 / (2 * (star_radius/2)**2))
                                hdri[py, px] += brightness * falloff * color
        except Exception as e:
            continue
    
    # Save as EXR if possible, otherwise PNG
    if HAS_OPENEXR:
        print(f"  Saving to {hdri_path}...")
        header = OpenEXR.Header(resolution[0], resolution[1])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        
        exr = OpenEXR.OutputFile(hdri_path, header)
        exr.writePixels({
            'R': hdri[:, :, 0].astype(np.float32).tobytes(),
            'G': hdri[:, :, 1].astype(np.float32).tobytes(),
            'B': hdri[:, :, 2].astype(np.float32).tobytes()
        })
        exr.close()
        return hdri_path
    elif HAS_PIL:
        # Fallback to PNG
        tone_mapped = np.clip(hdri ** (1/2.2), 0, 1)
        tone_mapped = (tone_mapped * 255).astype(np.uint8)
        img = Image.fromarray(tone_mapped)
        
        png_path = hdri_path.replace('.exr', '.png')
        img.save(png_path)
        print(f"  Saved PNG fallback to {png_path}")
        return png_path
    else:
        raise ImportError("Pillow or OpenEXR required for HDRI output")
