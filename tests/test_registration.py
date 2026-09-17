import pytest
import numpy as np
import cv2
from lunar_platform.registration.sift import SiftMatcher
from lunar_platform.registration.ransac import estimate_transform
from lunar_platform.registration.geometry import compute_reprojection_rmse, calculate_spatial_coverage

def test_sift_matcher():
    # Create a noisy synthetic image
    np.random.seed(42)
    img_a = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Create a translated version
    img_b = np.zeros((100, 100), dtype=np.uint8)
    img_b[10:100, 10:100] = img_a[0:90, 0:90]
    
    matcher = SiftMatcher()
    pts_a, pts_b, scores, name = matcher.match(img_a, img_b)
    
    assert name == "sift"
    assert len(pts_a) > 0
    assert len(pts_b) > 0
    assert len(pts_a) == len(pts_b)

def test_estimate_transform():
    # Create synthetic points
    pts_a = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    # Translate by (5, 5)
    pts_b = pts_a + np.array([5, 5])
    
    transform, inliers = estimate_transform(pts_a, pts_b, model_type="similarity")
    
    assert transform.shape == (3, 3)
    assert inliers.all()
    # Translation vector should be close to [5, 5]
    assert np.allclose(transform[0, 2], 5.0)
    assert np.allclose(transform[1, 2], 5.0)

def test_compute_rmse():
    pts_a = np.array([[0, 0], [10, 0]], dtype=np.float32)
    pts_b = np.array([[5, 0], [15, 0]], dtype=np.float32)
    transform = np.array([
        [1, 0, 5],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    rmse = compute_reprojection_rmse(pts_a, pts_b, transform)
    assert np.isclose(rmse, 0.0)
