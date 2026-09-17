from pydantic import BaseModel
from typing import Optional, Tuple

class LocalPoint(BaseModel):
    """A point in the local Cartesian sector coordinate system (meters)."""
    x: float
    y: float
    z: Optional[float] = 0.0

class GeographicPoint(BaseModel):
    """A point in the global lunar coordinate system (degrees)."""
    latitude: float
    longitude: float
    elevation: Optional[float] = 0.0

class ImageFootprint(BaseModel):
    """The estimated footprint of a registered image in local coordinates."""
    top_left: LocalPoint
    top_right: LocalPoint
    bottom_right: LocalPoint
    bottom_left: LocalPoint
    
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y)"""
        xs = [self.top_left.x, self.top_right.x, self.bottom_right.x, self.bottom_left.x]
        ys = [self.top_left.y, self.top_right.y, self.bottom_right.y, self.bottom_left.y]
        return min(xs), min(ys), max(xs), max(ys)
