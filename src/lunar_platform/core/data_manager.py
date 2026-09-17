import os
import json
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

class DataMode(Enum):
    DEMO = "DEMO"
    REAL_LOCAL = "REAL_LOCAL"

class DataManager:
    """
    Manages data path resolution and mode selection (DEMO vs REAL_LOCAL).
    Never touches data/raw at runtime.
    """
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.demo_dir = self.base_dir / "demo"
        self.sector_dir = self.base_dir / "sector"
        
        # Default active mode
        self.active_mode = DataMode.DEMO
        self.active_sector = "sector_001"
        
        # Determine initial fallback if requested sector data doesn't exist
        if not self._check_real_local_available(self.active_sector):
            self.active_mode = DataMode.DEMO
            
    def _check_real_local_available(self, sector_id: str) -> bool:
        """Checks if the required small sector files exist."""
        sector_path = self.sector_dir / sector_id
        if not sector_path.exists():
            return False
            
        # Basic check for at least the imagery folder and manifest
        if not (sector_path / "manifest.yaml").exists():
            return False
        if not (sector_path / "imagery").exists():
            return False
            
        return True

    def set_mode(self, mode: DataMode, sector_id: str = "sector_001") -> None:
        """Sets the active mode. Does NOT silently fallback if files are missing."""
        if mode == DataMode.REAL_LOCAL:
            if not self._check_real_local_available(sector_id):
                raise FileNotFoundError(
                    f"Real local sector data for {sector_id} is missing. "
                    f"Expected data at {self.sector_dir / sector_id}"
                )
                
        self.active_mode = mode
        self.active_sector = sector_id

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.active_mode.value,
            "sector": self.active_sector,
            "real_local_available": self._check_real_local_available(self.active_sector)
        }

    def get_input_image_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "input" / "demo_input.png"
        return self.sector_dir / self.active_sector / "imagery" / "ohrc_crop.tif"
        
    def get_input_metadata_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "input" / "demo_input_metadata.json"
        # In REAL_LOCAL, metadata might be alongside or inside manifest
        return self.sector_dir / self.active_sector / "imagery" / "ohrc_crop_metadata.json"

    def get_reference_image_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "reference" / "demo_ref.tif"
        return self.sector_dir / self.active_sector / "imagery" / "lro_nac_reference_crop.tif"

    def get_dem_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "dem" / "demo_dem.tif"
        return self.sector_dir / self.active_sector / "dem" / "dem.tif"

    def get_craters_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "craters" / "demo_craters.json"
        return self.sector_dir / self.active_sector / "craters" / "craters.json"
