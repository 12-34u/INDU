from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class ImageMetadata(BaseModel):
    """Metadata for a raw or processed lunar image."""
    image_id: str
    sensor: str
    instrument: str
    acquisition_time: Optional[datetime] = None
    width: int
    height: int
    gsd_m_per_px: float
    sun_azimuth: Optional[float] = None
    sun_elevation: Optional[float] = None
    camera_position: Optional[Dict[str, float]] = None
    camera_orientation: Optional[Dict[str, float]] = None
    coordinate_system: Optional[str] = None
    source_path: str
    
    # Track the native GSD and working GSD during processing
    native_gsd_m: float
    scale_factor: float = 1.0

class RegistrationQuality(BaseModel):
    """Quality metrics for a registration attempt."""
    num_raw_matches: int
    num_inliers: int
    inlier_ratio: float
    reprojection_rmse_px: float
    reprojection_rmse_m: float
    spatial_coverage: float
    processing_time_ms: float
    confidence: float

class RegistrationResult(BaseModel):
    """Result of registering an input image against a reference map/image."""
    image_id: str
    reference_id: str
    transform_matrix: list[list[float]]  # 3x3 homogeneous matrix
    model_type: str  # 'similarity', 'affine', 'homography'
    quality: RegistrationQuality
    estimated_x: float
    estimated_y: float
    estimated_rotation_deg: float
    estimated_scale: float
    matcher: str
