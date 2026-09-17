import numpy as np
from scipy import ndimage

def calculate_roughness(dem: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Calculates terrain roughness as the standard deviation of elevation 
    within a local window.
    """
    if window_size % 2 == 0:
        window_size += 1
        
    # Calculate local mean
    local_mean = ndimage.uniform_filter(dem, size=window_size)
    
    # Calculate local variance: E[X^2] - (E[X])^2
    local_sq_mean = ndimage.uniform_filter(dem**2, size=window_size)
    variance = local_sq_mean - local_mean**2
    
    # Ensure no negative variances due to float precision
    variance = np.maximum(variance, 0)
    
    # Roughness is standard deviation
    roughness = np.sqrt(variance)
    return roughness
