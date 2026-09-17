import pytest
import numpy as np
from lunar_platform.terrain.slope import calculate_slope
from lunar_platform.terrain.roughness import calculate_roughness

def test_calculate_slope():
    # Create a flat DEM
    dem = np.zeros((10, 10))
    slope = calculate_slope(dem, 1.0)
    assert np.all(slope == 0.0)
    
    # Create an angled DEM (45 degrees)
    # If GSD is 1.0, an increase of 1m per pixel gives 45 deg slope.
    # Note: gradient calculates central differences, so edge effects exist.
    x = np.arange(10)
    dem_45 = np.tile(x, (10, 1))
    slope_45 = calculate_slope(dem_45, 1.0)
    
    # Check interior pixels (away from edges)
    assert np.allclose(slope_45[1:-1, 1:-1], 45.0)

def test_calculate_roughness():
    # Flat DEM
    dem = np.zeros((10, 10))
    roughness = calculate_roughness(dem, window_size=3)
    assert np.all(roughness == 0.0)
    
    # Single spike
    dem_spike = np.zeros((10, 10))
    dem_spike[5, 5] = 10.0
    roughness_spike = calculate_roughness(dem_spike, window_size=3)
    
    # Roughness around the spike should be non-zero
    assert roughness_spike[5, 5] > 0
    assert roughness_spike[0, 0] == 0
