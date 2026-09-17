import numpy as np
from scipy import ndimage

def calculate_slope(dem: np.ndarray, gsd_m: float) -> np.ndarray:
    """
    Calculates the slope map in degrees from a DEM.
    Uses a simple Sobel filter approximation for gradients.
    """
    # Gradients in x and y
    dy, dx = np.gradient(dem, gsd_m, gsd_m)
    
    # Calculate slope in radians
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    
    # Convert to degrees
    slope_deg = np.degrees(slope_rad)
    return slope_deg
