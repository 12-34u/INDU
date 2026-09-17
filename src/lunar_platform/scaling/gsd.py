import math

def calculate_scale_factor(source_gsd: float, target_gsd: float) -> float:
    """
    Calculates the scaling factor to go from source GSD to target GSD.
    e.g., source 0.25m/px -> target 1.0m/px => scale_factor = 0.25 (need to shrink by 4x)
    """
    if target_gsd <= 0 or source_gsd <= 0:
        raise ValueError("GSD must be strictly positive.")
    return source_gsd / target_gsd

def compute_new_dimensions(width: int, height: int, scale_factor: float) -> tuple[int, int]:
    """Computes new image dimensions based on a scale factor."""
    new_width = max(1, int(round(width * scale_factor)))
    new_height = max(1, int(round(height * scale_factor)))
    return new_width, new_height
