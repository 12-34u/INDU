import numpy as np

def generate_hillshade(dem: np.ndarray, gsd_m: float, azimuth_deg: float = 315, elevation_deg: float = 45) -> np.ndarray:
    """
    Generates a synthetic hillshade from the DEM.
    Returns a normalized [0, 1] float array.
    """
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    
    dy, dx = np.gradient(dem, gsd_m, gsd_m)
    
    slope = np.pi/2. - np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    
    shaded = np.sin(elevation_rad) * np.sin(slope) + \
             np.cos(elevation_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
             
    shaded = np.clip(shaded, 0, 1)
    return shaded
