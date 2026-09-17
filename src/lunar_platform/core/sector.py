from pydantic import BaseModel
from typing import Optional, List
from .coordinates import GeographicPoint, ImageFootprint

class CoordinateSystem(BaseModel):
    type: str = "local_lunar_xy"

class LunarSector(BaseModel):
    """Configuration and state for a single working lunar sector."""
    id: str
    center: GeographicPoint
    width_m: float
    height_m: float
    working_gsd_m: float
    coordinate_system: CoordinateSystem = CoordinateSystem()
    
    @property
    def pixels_width(self) -> int:
        return int(self.width_m / self.working_gsd_m)
        
    @property
    def pixels_height(self) -> int:
        return int(self.height_m / self.working_gsd_m)
