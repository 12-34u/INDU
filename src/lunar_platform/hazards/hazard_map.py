import numpy as np
from scipy import ndimage

def calculate_crater_hazard(craters: list[dict], width: int, height: int, gsd_m: float) -> np.ndarray:
    """
    Calculates a crater hazard map [0, 1] based on distance to nearest crater.
    craters is a list of dicts with 'x', 'y', 'diameter_m'.
    x and y are in meters (local coordinates).
    """
    hazard_map = np.zeros((height, width), dtype=np.float32)
    
    if not craters:
        return hazard_map
        
    # We will compute a distance transform. First, create a binary mask of crater interiors.
    mask = np.ones((height, width), dtype=bool)
    
    y_grid, x_grid = np.ogrid[:height, :width]
    
    for c in craters:
        cx_px = c['x'] / gsd_m
        cy_px = c['y'] / gsd_m
        radius_px = (c['diameter_m'] / 2.0) / gsd_m
        
        # Calculate squared distance from crater center
        dist_sq = (x_grid - cx_px)**2 + (y_grid - cy_px)**2
        
        # Inside crater -> hazard = 1.0 (mask = 0)
        mask[dist_sq <= radius_px**2] = 0
        
    # Distance transform from the crater edges
    # ndimage.distance_transform_edt computes distance to nearest zero
    distance_px = ndimage.distance_transform_edt(mask)
    distance_m = distance_px * gsd_m
    
    # Map distance to hazard.
    # Inside crater = 1.0. 
    # Close to crater = high hazard, falling off to 0 linearly up to e.g. 50 meters
    max_hazard_dist_m = 50.0
    
    # hazard = max(0, 1 - distance/max_dist)
    hazard_map = 1.0 - (distance_m / max_hazard_dist_m)
    hazard_map = np.clip(hazard_map, 0.0, 1.0)
    
    # Inside craters distance is 0, so hazard is 1.0
    return hazard_map
