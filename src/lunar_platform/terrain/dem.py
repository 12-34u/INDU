import numpy as np
import rasterio
from typing import Protocol, Optional
from pathlib import Path

class DEMProvider(Protocol):
    def get_dem(self) -> np.ndarray:
        """Returns the DEM as a 2D float array in meters."""
        ...
        
    def get_gsd_m(self) -> float:
        """Returns the Ground Sample Distance (resolution) in meters."""
        ...

class LocalDEMProvider(DEMProvider):
    """Loads a real DEM using rasterio."""
    def __init__(self, dem_path: str | Path):
        self.dem_path = Path(dem_path)
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM not found at {self.dem_path}")
            
    def get_dem(self) -> np.ndarray:
        with rasterio.open(self.dem_path) as src:
            dem = src.read(1)
        return dem.astype(np.float32)
        
    def get_gsd_m(self) -> float:
        # Simplification: Assume square pixels and metric CRS for the MVP.
        with rasterio.open(self.dem_path) as src:
            return float(src.transform[0])

class SyntheticDEMProvider(DEMProvider):
    """Generates procedural terrain using numpy when no real DEM is available."""
    def __init__(self, width: int = 1000, height: int = 1000, gsd_m: float = 1.0):
        self.width = width
        self.height = height
        self.gsd_m = gsd_m
        
    def get_dem(self) -> np.ndarray:
        # Create a simple sloped terrain with some sine wave noise
        y_grid, x_grid = np.mgrid[0:self.height, 0:self.width]
        
        # Base slope
        dem = y_grid * 0.05
        
        # Add some macro structures
        dem += np.sin(x_grid / 50.0) * 2.0
        dem += np.cos(y_grid / 30.0) * 1.5
        
        return dem.astype(np.float32)
        
    def get_gsd_m(self) -> float:
        return self.gsd_m
