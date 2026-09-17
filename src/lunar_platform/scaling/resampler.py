import cv2
import numpy as np
from .gsd import calculate_scale_factor, compute_new_dimensions
from ..core.metadata import ImageMetadata

def resample_image(image: np.ndarray, source_gsd: float, target_gsd: float) -> tuple[np.ndarray, float]:
    """
    Resamples an image to the target GSD.
    Returns the resampled image and the scale_factor used.
    """
    if np.isclose(source_gsd, target_gsd, rtol=1e-3):
        return image.copy(), 1.0
        
    scale_factor = calculate_scale_factor(source_gsd, target_gsd)
    h, w = image.shape[:2]
    new_w, new_h = compute_new_dimensions(w, h, scale_factor)
    
    # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
    
    resampled = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resampled, scale_factor

def harmonize_metadata(metadata: ImageMetadata, target_gsd: float) -> ImageMetadata:
    """Updates metadata to reflect the new working GSD after resampling."""
    scale_factor = calculate_scale_factor(metadata.native_gsd_m, target_gsd)
    new_w, new_h = compute_new_dimensions(metadata.width, metadata.height, scale_factor)
    
    new_metadata = metadata.model_copy(deep=True)
    new_metadata.width = new_w
    new_metadata.height = new_h
    new_metadata.gsd_m_per_px = target_gsd
    new_metadata.scale_factor = scale_factor
    return new_metadata
