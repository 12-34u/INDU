import numpy as np

def compute_reprojection_rmse(pts_a: np.ndarray, pts_b: np.ndarray, transform: np.ndarray) -> float:
    """Computes the Root Mean Square Error (RMSE) of the reprojection."""
    if len(pts_a) == 0:
        return float('inf')
        
    # Convert pts_a to homogeneous coordinates
    pts_a_hom = np.hstack((pts_a, np.ones((len(pts_a), 1))))
    
    # Apply transform
    pts_a_transformed = (transform @ pts_a_hom.T).T
    
    # Normalize by z-coordinate
    pts_a_transformed = pts_a_transformed[:, :2] / pts_a_transformed[:, 2:]
    
    # Calculate Euclidean distances
    distances = np.linalg.norm(pts_a_transformed - pts_b, axis=1)
    
    # RMSE
    return float(np.sqrt(np.mean(distances**2)))

def calculate_spatial_coverage(points: np.ndarray, image_width: int, image_height: int, grid_size: int = 10) -> float:
    """
    Calculates the spatial coverage of points by dividing the image into a grid 
    and counting the fraction of occupied cells.
    """
    if len(points) == 0:
        return 0.0
        
    cell_w = image_width / grid_size
    cell_h = image_height / grid_size
    
    occupied = set()
    for pt in points:
        x, y = pt
        if 0 <= x < image_width and 0 <= y < image_height:
            cell_x = int(x / cell_w)
            cell_y = int(y / cell_h)
            occupied.add((cell_x, cell_y))
            
    return len(occupied) / (grid_size * grid_size)
