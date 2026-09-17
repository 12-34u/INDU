import cv2
import numpy as np
from typing import Tuple

def estimate_transform(pts_a: np.ndarray, pts_b: np.ndarray, model_type: str = "similarity", threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the geometric transformation from A to B.
    
    Args:
        pts_a: (N, 2) array of points in image A
        pts_b: (N, 2) array of corresponding points in image B
        model_type: 'similarity', 'affine', or 'homography'
        threshold: RANSAC threshold
        
    Returns:
        transform_matrix: 3x3 homogeneous transformation matrix
        inliers_mask: (N,) boolean array indicating inliers
    """
    if len(pts_a) < 3:
        return np.eye(3), np.zeros(len(pts_a), dtype=bool)
        
    if model_type == "similarity":
        # OpenCV's estimateAffinePartial2D estimates a similarity transform (rotation + scale + translation)
        # It requires at least 2 points, but 3+ is better for robust RANSAC
        M, inliers = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=threshold)
        if M is None:
            return np.eye(3), np.zeros(len(pts_a), dtype=bool)
        # Convert 2x3 to 3x3
        transform = np.eye(3)
        transform[:2, :] = M
        
    elif model_type == "affine":
        if len(pts_a) < 3:
             return np.eye(3), np.zeros(len(pts_a), dtype=bool)
        M, inliers = cv2.estimateAffine2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=threshold)
        if M is None:
            return np.eye(3), np.zeros(len(pts_a), dtype=bool)
        transform = np.eye(3)
        transform[:2, :] = M
        
    elif model_type == "homography":
        if len(pts_a) < 4:
            return np.eye(3), np.zeros(len(pts_a), dtype=bool)
        # MAGSAC++ is very robust for homography estimation
        transform, inliers = cv2.findHomography(pts_a, pts_b, method=cv2.USAC_MAGSAC, ransacReprojThreshold=threshold)
        if transform is None:
            return np.eye(3), np.zeros(len(pts_a), dtype=bool)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    inliers_mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(pts_a), dtype=bool)
    return transform, inliers_mask
