import pytest
from lunar_platform.scaling.gsd import calculate_scale_factor, compute_new_dimensions

def test_calculate_scale_factor():
    assert calculate_scale_factor(0.25, 1.0) == 0.25
    assert calculate_scale_factor(1.0, 0.25) == 4.0
    assert calculate_scale_factor(0.5, 0.5) == 1.0

def test_compute_new_dimensions():
    assert compute_new_dimensions(100, 200, 0.5) == (50, 100)
    assert compute_new_dimensions(100, 200, 2.0) == (200, 400)
    assert compute_new_dimensions(10, 10, 0.01) == (1, 1) # Minimum size 1x1
