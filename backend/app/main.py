from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
import json
import cv2
import numpy as np
from pathlib import Path

from lunar_platform.core.data_manager import DataManager, DataMode
from lunar_platform.navigation.astar import plan_path_astar
from lunar_platform.terrain.dem import LocalDEMProvider, SyntheticDEMProvider
from lunar_platform.terrain.slope import calculate_slope
from lunar_platform.terrain.roughness import calculate_roughness
from lunar_platform.hazards.hazard_map import calculate_crater_hazard
from lunar_platform.navigation.cost_map import generate_cost_map

app = FastAPI(title="Lunar Localization & Traversability API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_manager = DataManager()

class PathRequest(BaseModel):
    start: Tuple[int, int]
    goal: Tuple[int, int]

class ModeRequest(BaseModel):
    mode: str

@app.get("/")
def read_root():
    return {"message": "Lunar Localization Platform API is running"}

@app.get("/data/status")
def get_data_status():
    return data_manager.get_status()

@app.post("/data/mode")
def set_data_mode(req: ModeRequest):
    try:
        mode = DataMode(req.mode.upper())
        data_manager.set_mode(mode)
        return {"status": "success", "current_mode": mode.value}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be DEMO or REAL_LOCAL.")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

def _get_terrain_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Helper to fetch and calculate full-resolution terrain data."""
    dem_path = data_manager.get_dem_path()
    if dem_path.exists():
        provider = LocalDEMProvider(dem_path)
    else:
        provider = SyntheticDEMProvider(width=1000, height=1000, gsd_m=1.0)
        
    dem = provider.get_dem()
    gsd_m = provider.get_gsd_m()
    
    slope = calculate_slope(dem, gsd_m)
    roughness = calculate_roughness(dem)
    
    craters_path = data_manager.get_craters_path()
    craters = []
    if craters_path.exists():
        with open(craters_path, "r") as f:
            craters = json.load(f).get("craters", [])
            
    height, width = dem.shape
    hazard = calculate_crater_hazard(craters, width, height, gsd_m)
    
    return dem, slope, roughness, hazard, gsd_m

@app.get("/terrain")
def get_terrain(max_grid_size: int = 100):
    cache_dir = Path("data/cache/terrain")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    mode_str = data_manager.active_mode.value.lower()
    sector_str = data_manager.active_sector
    cache_file = cache_dir / f"{sector_str}_{mode_str}_{max_grid_size}.json"
    
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
            
    try:
        dem, slope, roughness, hazard, gsd_m = _get_terrain_data()
        
        orig_h, orig_w = dem.shape
        
        # Calculate new dimensions preserving aspect ratio
        if orig_w > orig_h:
            new_w = max_grid_size
            new_h = int(orig_h * (max_grid_size / orig_w))
        else:
            new_h = max_grid_size
            new_w = int(orig_w * (max_grid_size / orig_h))
            
        # Ensure at least 1x1
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # Downsample using bilinear interpolation
        dem_small = cv2.resize(dem, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        slope_small = cv2.resize(slope, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rough_small = cv2.resize(roughness, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        hazard_small = cv2.resize(hazard, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # We need physical dimensions to send to frontend so it can reconstruct the correct bounding box
        width_m = orig_w * gsd_m
        height_m = orig_h * gsd_m
        
        result = {
            "sector_id": sector_str,
            "width_m": float(width_m),
            "height_m": float(height_m),
            "grid": {
                "width": new_w,
                "height": new_h
            },
            # Rounding to 3 decimals to save space in JSON
            "elevation": np.round(dem_small.flatten(), 3).tolist(),
            "slope": np.round(slope_small.flatten(), 3).tolist(),
            "roughness": np.round(rough_small.flatten(), 3).tolist(),
            "hazards": np.round(hazard_small.flatten(), 3).tolist(),
            "min_elevation": float(np.min(dem_small)),
            "max_elevation": float(np.max(dem_small)),
            "min_slope": float(np.min(slope_small)),
            "max_slope": float(np.max(slope_small))
        }
        
        # Craters (raw coords so frontend can draw them exactly)
        craters_path = data_manager.get_craters_path()
        if craters_path.exists():
            with open(craters_path, "r") as f:
                result["craters"] = json.load(f).get("craters", [])
        else:
            result["craters"] = []
            
        # Save to cache
        with open(cache_file, "w") as f:
            json.dump(result, f)
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan_path")
def plan_path(request: PathRequest):
    try:
        dem, slope, roughness, hazard, _ = _get_terrain_data()
        
        config = {"max_slope_deg": 20.0}
        cost_map = generate_cost_map(slope, roughness, hazard, config)
        
        # Run A* on the full resolution grid (1000x1000)
        path = plan_path_astar(cost_map, request.start, request.goal)
        
        if not path:
            raise HTTPException(status_code=404, detail="No path found")
            
        return {"path": path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
