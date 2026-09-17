from ..core.metadata import RegistrationQuality
from ..utils.config import get_config

def evaluate_quality(quality: RegistrationQuality) -> bool:
    """Evaluates if the registration meets the configured quality thresholds."""
    config = get_config("registration")
    
    if quality.num_raw_matches < config.get("min_matches", 30):
        return False
    if quality.inlier_ratio < config.get("min_inlier_ratio", 0.35):
        return False
    if quality.reprojection_rmse_px > config.get("max_rmse_px", 5.0):
        return False
    if quality.spatial_coverage < config.get("min_spatial_coverage", 0.30):
        return False
        
    return True
