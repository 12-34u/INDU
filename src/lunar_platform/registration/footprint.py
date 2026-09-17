import numpy as np
from ..core.coordinates import LocalPoint, ImageFootprint

def estimate_footprint(image_width: int, image_height: int, transform: np.ndarray, working_gsd_m: float) -> ImageFootprint:
    """
    Estimates the footprint of the image in the local sector coordinate system.
    Note: This assumes the reference image origin (0,0) corresponds to local coordinate (0,0).
    If the reference image has an offset, it must be added.
    """
    # Corners in pixel coordinates of the input image
    corners = np.array([
        [0, 0],
        [image_width - 1, 0],
        [image_width - 1, image_height - 1],
        [0, image_height - 1]
    ], dtype=np.float32)
    
    # Homogeneous coordinates
    corners_hom = np.hstack((corners, np.ones((4, 1))))
    
    # Transform to reference image pixel coordinates
    transformed_corners = (transform @ corners_hom.T).T
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]
    
    # Convert reference pixel coordinates to local meters
    # Assuming (0,0) in ref image = (0,0) in local coords
    local_corners = transformed_corners * working_gsd_m
    
    return ImageFootprint(
        top_left=LocalPoint(x=local_corners[0][0], y=local_corners[0][1]),
        top_right=LocalPoint(x=local_corners[1][0], y=local_corners[1][1]),
        bottom_right=LocalPoint(x=local_corners[2][0], y=local_corners[2][1]),
        bottom_left=LocalPoint(x=local_corners[3][0], y=local_corners[3][1]),
    )
