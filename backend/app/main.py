from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import base64
import json
import cv2
import numpy as np
from pathlib import Path

from lunar_platform.core.data_manager import DataManager, DataMode
from lunar_platform.crater_detection.rim_points import RimPointDetector
from lunar_platform.crater_detection.shadow_pairs import ShadowPairDetector
from lunar_platform.localization.estimator import run_pipeline, run_real_pipeline
from lunar_platform.localization.image_match import BasemapLocalizer
from lunar_platform.map.reference_map import (
    build_reference_map,
    build_reference_map_from_basemap,
    to_hazard_dicts,
)
from lunar_platform.observation.basemap import build_sector_basemap_from_source
from lunar_platform.observation.real_view import native_camera
from lunar_platform.navigation.astar import plan_path_astar
from lunar_platform.navigation.route import route_metrics
from lunar_platform.registration.pipeline import load_image, run_registration
from lunar_platform.registration.products import write_registration_products
from lunar_platform.simulation.camera import NadirCamera, RoverPose
from lunar_platform.terrain.dem import LocalDEMProvider, SyntheticDEMProvider
from lunar_platform.terrain.slope import calculate_slope
from lunar_platform.terrain.roughness import calculate_roughness
from lunar_platform.terrain.traversability import compute_traversability
from lunar_platform.hazards.hazard_map import calculate_crater_hazard
from lunar_platform.navigation.cost_map import generate_cost_map
from lunar_platform.utils.config import get_config

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

class RegistrationRequest(BaseModel):
    """'source_reference' or 'self_check'."""
    pair: str = "source_reference"
    max_dimension_px: int = 1600
    illumination_representation: Optional[str] = None

class PerceptionRequest(BaseModel):
    """Rover pose to observe from, in sector metres (y down)."""
    x_m: float
    y_m: float
    heading_deg: float = 0.0
    # Left unset, the camera is resolved per mode: the demo keeps its 512 px
    # at 0.5 m/px, while real imagery is sampled at the browse product's own
    # resolution rather than magnified past what it carries.
    image_size_px: Optional[int] = None
    observation_gsd_m: Optional[float] = None
    # None follows the active data mode. True forces the real-imagery path and
    # fails loudly if data/raw cannot supply it, rather than quietly rendering
    # a synthetic frame in its place.
    use_real_data: Optional[bool] = None

# Recomputing the DEM and its derived layers costs seconds; every endpoint here
# needs the same arrays, so they are memoised per data mode and sector.
_terrain_cache: Dict[str, Any] = {}

def _terrain_cache_key() -> str:
    return f"{data_manager.active_mode.value}:{data_manager.active_sector}"

def _invalidate_terrain_cache() -> None:
    _terrain_cache.clear()
    _simulation_cache.clear()

_detector = RimPointDetector()

# Real-imagery simulation. The basemap, its detected landmarks and the
# localiser's basemap features are all expensive to build and identical
# between observations, so they are built once per sector and reused.
_simulation_cache: Dict[str, Any] = {}
_real_detector = ShadowPairDetector()
_real_localizer: Optional[BasemapLocalizer] = None


def _simulation_config() -> dict:
    return get_config("simulation") or {}


def _simulation_basemap():
    """
    The sector basemap cut from real OHRC imagery under data/raw.

    Raises 409 rather than falling back to synthetic terrain: a caller that
    asked for real data and silently received a rendering has no way to tell.
    """
    key = f"basemap:{data_manager.active_sector}"
    if key in _simulation_cache:
        return _simulation_cache[key]

    source = data_manager.get_simulation_source()
    if source is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No Chandrayaan-2 OHRC bundle with a readable browse product was found "
                f"under {data_manager.raw_dir}. The simulation cannot run on real data."
            ),
        )

    centre = data_manager.get_sector_centre_lonlat()
    config = _simulation_config()
    try:
        basemap = build_sector_basemap_from_source(
            source,
            centre_longitude=centre[0] if centre else None,
            centre_latitude=centre[1] if centre else None,
            size_m=float(config.get("sector_size_m", 2400.0)),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not build sector basemap: {exc}")

    _simulation_cache[key] = basemap
    return basemap


def _real_reference_craters(basemap):
    """Landmarks detected in the real basemap, built once per sector."""
    key = f"refs:{data_manager.active_sector}:{basemap.width_px}"
    if key not in _simulation_cache:
        reference_config = _simulation_config().get("reference_map", {})
        _simulation_cache[key] = build_reference_map_from_basemap(
            basemap,
            ShadowPairDetector(sun_direction_deg=basemap.sun_direction_deg),
            min_diameter_m=float(reference_config.get("min_diameter_m", 20.0)),
            max_craters=int(reference_config.get("max_craters", 400)),
        )
    return _simulation_cache[key]


def _basemap_localizer() -> BasemapLocalizer:
    global _real_localizer
    if _real_localizer is None:
        config = _simulation_config().get("localization", {})
        _real_localizer = BasemapLocalizer(
            n_features=int(config.get("n_features", 4000)),
            min_inliers=int(config.get("min_inliers", 8)),
            max_heading_residual_deg=float(config.get("max_heading_residual_deg", 5.0)),
            max_scale_residual=float(config.get("max_scale_residual", 0.10)),
        )
    return _real_localizer


def _real_dem_provider(basemap):
    """
    Elevation for the real sector: the LOLA DEM when present, generated if not.

    A failure to read the DEM falls back to generated terrain rather than
    taking the sector down, but it says so in the returned description instead
    of passing the substitute off as measured.
    """
    label = data_manager.get_raw_dem_label()
    if label is not None:
        try:
            from lunar_platform.terrain.lola_dem import LolaPolarDEM, SectorDEMProvider

            elevation = LolaPolarDEM(label).sample_sector(basemap)
            return SectorDEMProvider(elevation), elevation
        except Exception as exc:
            _simulation_cache["elevation_error"] = str(exc)

    return (
        SyntheticDEMProvider(
            width=basemap.width_px, height=basemap.height_px, gsd_m=basemap.gsd_m
        ),
        None,
    )


def _nac_camera():
    """
    The SPICE camera model for the reference NAC image, built once.

    Loading kernels is global state in SPICE and reading them is not free, so
    the model is cached for the process rather than rebuilt per request.
    """
    if "nac_camera" in _simulation_cache:
        return _simulation_cache["nac_camera"]

    from lunar_platform.planetary.spice_adapter import (
        NacCameraModel,
        SpiceUnavailable,
        parse_nac_label,
    )

    kernels = data_manager.get_spice_kernels()
    if kernels is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No SPICE kernels found under {data_manager.get_spice_root()}. "
                "The NAC image carries no geographic metadata, so it cannot be placed "
                "on the Moon without them."
            ),
        )
    if not kernels.is_complete:
        raise HTTPException(
            status_code=409,
            detail=(
                "The SPICE kernel set is incomplete; missing: "
                + ", ".join(kernels.missing())
            ),
        )

    reference = data_manager.get_nac_reference()
    if reference is None:
        raise HTTPException(
            status_code=409, detail="No LROC NAC product with a label was found under data/raw."
        )

    try:
        observation = parse_nac_label(
            reference["label_path"].read_text("utf-8", "replace"), reference["product_id"]
        )
        camera = NacCameraModel(observation, kernels)
    except SpiceUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not build the NAC camera model: {exc}")

    _simulation_cache["nac_camera"] = camera
    return camera


def _use_real_simulation(requested: Optional[bool]) -> bool:
    """
    Whether this request runs on real imagery.

    Unset follows the active data mode, so switching the mode switches the
    simulation and nothing changes under a caller that did not ask for it.
    """
    if requested is None:
        return data_manager.active_mode == DataMode.REAL_RAW
    return bool(requested)


def _resolve_camera(request: PerceptionRequest, basemap=None):
    """Camera for this request, defaulting per mode."""
    observation_config = _simulation_config().get("observation", {})

    if basemap is not None:
        size = request.image_size_px or int(observation_config.get("size_px", 256))
        configured_gsd = observation_config.get("gsd_m", "native")
        if request.observation_gsd_m is not None:
            return NadirCamera(size, size, float(request.observation_gsd_m))
        if configured_gsd == "native" or configured_gsd is None:
            return native_camera(basemap, size)
        return NadirCamera(size, size, float(configured_gsd))

    return NadirCamera(
        width_px=request.image_size_px or 512,
        height_px=request.image_size_px or 512,
        gsd_m=request.observation_gsd_m if request.observation_gsd_m is not None else 0.5,
    )

def _navigation_config() -> dict:
    return get_config("navigation") or {}

def _sector_craters() -> List[dict]:
    craters_path = data_manager.get_craters_path()
    if not craters_path.exists():
        return []
    with open(craters_path, "r") as f:
        return json.load(f).get("craters", [])

def _reference_craters(width_m: float, height_m: float):
    key = f"ref:{_terrain_cache_key()}:{width_m}x{height_m}"
    if key not in _terrain_cache:
        _terrain_cache[key] = build_reference_map(width_m, height_m, _sector_craters())
    return _terrain_cache[key]

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
        _invalidate_terrain_cache()
        return {"status": "success", "current_mode": mode.value}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be DEMO or REAL_LOCAL.")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

def _get_terrain_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Helper to fetch and calculate full-resolution terrain data."""
    cache_key = _terrain_cache_key()
    if cache_key in _terrain_cache:
        return _terrain_cache[cache_key]

    dem_path = data_manager.get_dem_path()
    if data_manager.active_mode == DataMode.REAL_RAW and not dem_path.exists():
        # Elevation comes from the LOLA DEM under data/raw when one is there,
        # and is generated only when it is not. Either way it is built on the
        # basemap's exact extent and posting: the rover pose, the route and the
        # observation all live in one sector frame.
        basemap = _simulation_basemap()
        provider, elevation = _real_dem_provider(basemap)
        dem = provider.get_dem()
        gsd_m = provider.get_gsd_m()
        _simulation_cache["elevation"] = (
            elevation.describe() if elevation is not None else {"is_synthetic": True}
        )
        # Hazards, unlike elevation, are real here: they come from the craters
        # detected in the imagery rather than from the demo crater file.
        hazard = calculate_crater_hazard(
            to_hazard_dicts(_real_reference_craters(basemap)),
            basemap.width_px,
            basemap.height_px,
            gsd_m,
        )
        result = (
            dem,
            calculate_slope(dem, gsd_m),
            calculate_roughness(dem),
            hazard,
            gsd_m,
        )
        _terrain_cache[cache_key] = result
        return result

    if dem_path.exists():
        provider = LocalDEMProvider(dem_path)
    elif data_manager.active_mode == DataMode.REAL_LOCAL:
        # Substituting synthetic terrain here would be indistinguishable from
        # real elevation in the response.
        raise HTTPException(
            status_code=404,
            detail=(
                f"REAL_LOCAL DEM is missing at {dem_path}. "
                f"Prepare it with: python scripts/build_terrain.py "
                f"--sector {data_manager.active_sector} --dtm <DTM product>"
            ),
        )
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

    _terrain_cache[cache_key] = (dem, slope, roughness, hazard, gsd_m)
    return dem, slope, roughness, hazard, gsd_m

@app.get("/terrain")
def get_terrain(max_grid_size: int = 100):
    cache_dir = Path("data/cache/terrain")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    mode_str = data_manager.active_mode.value.lower()
    sector_str = data_manager.active_sector
    # Bumped when the response schema changes OR when what feeds it changes, so
    # old cache files are ignored. v4 added detected craters and real elevation:
    # a v3 file for the same sector and mode holds generated terrain.
    cache_file = cache_dir / f"{sector_str}_{mode_str}_{max_grid_size}_v4.json"
    
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

        # Same array the planner searches, so the shaded surface and the route agree.
        traversability, _, cost_stats = compute_traversability(
            slope, roughness, hazard, _navigation_config()
        )
        trav_small = cv2.resize(traversability, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
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
            # Full-resolution DEM geometry. The planner searches this grid, not
            # the downsampled display grid above, so the client needs both to
            # convert between sector metres and planner cells.
            "gsd_m": float(gsd_m),
            "dem_width": int(orig_w),
            "dem_height": int(orig_h),
            # Rounding to 3 decimals to save space in JSON
            "elevation": np.round(dem_small.flatten(), 3).tolist(),
            "slope": np.round(slope_small.flatten(), 3).tolist(),
            "roughness": np.round(rough_small.flatten(), 3).tolist(),
            "hazards": np.round(hazard_small.flatten(), 3).tolist(),
            "traversability": np.round(trav_small.flatten(), 3).tolist(),
            "min_elevation": float(np.min(dem_small)),
            "max_elevation": float(np.max(dem_small)),
            "min_slope": float(np.min(slope_small)),
            "max_slope": float(np.max(slope_small)),
            "cost_stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cost_stats.items()}
        }
        
        # Craters (raw coords so frontend can draw them exactly). In real mode
        # these are the craters detected in the imagery, which is the same set
        # the hazard layer above was built from.
        if data_manager.active_mode == DataMode.REAL_RAW:
            result["craters"] = to_hazard_dicts(
                _real_reference_craters(_simulation_basemap())
            )
            result["craters_are_detected"] = True
        else:
            craters_path = data_manager.get_craters_path()
            if craters_path.exists():
                with open(craters_path, "r") as f:
                    result["craters"] = json.load(f).get("craters", [])
            else:
                result["craters"] = []
            result["craters_are_detected"] = False
            
        # Save to cache
        with open(cache_file, "w") as f:
            json.dump(result, f)
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan_path")
def plan_path(request: PathRequest):
    try:
        dem, slope, roughness, hazard, gsd_m = _get_terrain_data()

        config = _navigation_config()
        traversability, cost_map, cost_stats = compute_traversability(
            slope, roughness, hazard, config
        )

        # Run A* on the full resolution grid (1000x1000)
        path = plan_path_astar(cost_map, request.start, request.goal, config)

        if not path:
            raise HTTPException(status_code=404, detail="No path found")

        metrics = route_metrics(
            path, slope, cost_map, gsd_m, cost_stats.get("high_cost_threshold")
        )
        return {"path": path, "metrics": metrics}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/capabilities")
def get_data_capabilities():
    """
    What the system can currently do.

    DEM and SPICE are reported as flags and never gate registration; their
    absence disables geolocation features only.
    """
    capabilities = data_manager.get_capabilities()
    for key in ("source_product", "reference_product"):
        product = capabilities.get(key)
        if product:
            # Member lists are long and only the notable ones matter to the UI.
            product["members"] = [
                m for m in product.get("members", [])
                if m["kind"] in ("image", "browse", "geometry", "nav")
            ]
    return capabilities


@app.post("/registration/run")
def run_registration_endpoint(request: RegistrationRequest):
    """
    Run the registration workflow and write its products.

    'source_reference' registers the discovered source against the discovered
    reference. 'self_check' registers a real source tile against a known
    transform of itself, which is the only accuracy figure here that can be
    verified without camera geometry.
    """
    capabilities = data_manager.get_capabilities()
    if not capabilities["registration"]:
        raise HTTPException(
            status_code=409,
            detail="Registration blocked: " + "; ".join(capabilities["messages"]),
        )

    config = get_config("registration") or {}
    config["max_dimension_px"] = request.max_dimension_px
    if request.illumination_representation:
        config["illumination_representation"] = request.illumination_representation

    try:
        if request.pair == "self_check":
            outcome, extra = _run_self_check(capabilities, config)
        else:
            outcome = run_registration(
                capabilities["source_ref"],
                capabilities["reference_ref"],
                "ch2_ohrc_vs_lronac",
                config,
            )
            extra = {}

        written = write_registration_products(
            outcome,
            Path("outputs/registration"),
            capabilities["source_ref"],
            capabilities["reference_ref"],
        )
        manifest = written["manifest"]
        manifest.update(extra)
        manifest["n_matches"] = len(outcome.matches)
        manifest["n_inliers"] = sum(1 for m in outcome.matches if m["inlier"])
        manifest["matches_preview"] = outcome.matches[:400]
        return manifest
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_self_check(capabilities: dict, config: dict):
    """Controlled validation against a known transform on real source imagery."""
    from lunar_platform.preprocessing.normalization import to_uint8
    from lunar_platform.registration.validation import (
        apply_illumination_change,
        apply_known_transform,
        transform_error,
    )

    base, _ = load_image(capabilities["source_ref"], config["max_dimension_px"])
    base = to_uint8(base)
    warped, truth = apply_known_transform(base)
    warped = apply_illumination_change(warped)

    work_dir = data_manager.get_working_registration_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = work_dir / "selfcheck_fixed.png"
    moving_path = work_dir / "selfcheck_moving.png"
    cv2.imwrite(str(fixed_path), base)
    cv2.imwrite(str(moving_path), warped)

    outcome = run_registration(str(moving_path), str(fixed_path), "ohrc_self_check", config)

    extra = {"known_transform_applied": {
        "rotation_deg": 7.0, "scale": 0.82, "translation_px": [14.0, -9.0], "gamma": 1.7
    }}
    if outcome.transform is not None:
        # The pipeline solves moving -> fixed; the truth was applied fixed -> moving.
        extra["known_transform_error"] = transform_error(
            outcome.transform, np.linalg.inv(truth), base.shape[1], base.shape[0]
        )
    return outcome, extra


@app.get("/registration/result/{scene_id}")
def get_registration_result(scene_id: str):
    metrics_path = Path("outputs/registration") / scene_id / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"No registration result for {scene_id}")
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/registration/image/{scene_id}/{name}")
def get_registration_image(scene_id: str, name: str):
    """Serve a display-resolution derivative, never the full science raster."""
    if not name.startswith("display_") or not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only display derivatives are served.")

    # Resolve and confine to the scene directory so a crafted name cannot escape it.
    root = (Path("outputs/registration") / scene_id).resolve()
    path = (root / name).resolve()
    if not str(path).startswith(str(root)) or not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found for {scene_id}")
    return FileResponse(path, media_type="image/png")


@app.get("/reference_map")
def get_reference_map(real: Optional[bool] = None):
    """
    Landmark map the simulation places the rover against.

    On real imagery every landmark is a crater detected in the Chandrayaan-2
    browse product. In the demo the catalog craters come from the sector's own
    crater file and the rest are generated; that map is synthetic and says so.
    """
    if _use_real_simulation(real):
        basemap = _simulation_basemap()
        craters = _real_reference_craters(basemap)
        return {
            "sector_id": data_manager.active_sector,
            "width_m": basemap.width_m,
            "height_m": basemap.height_m,
            "is_synthetic": False,
            "note": (
                "Craters detected in real Chandrayaan-2 OHRC imagery by shadow/highlight "
                "pairing. These are detections, not a published crater catalog: they have "
                "not been scored against one, and their diameters come from shadow "
                "geometry rather than a fitted rim."
            ),
            "basemap": basemap.describe(),
            "craters": [c.model_dump() for c in craters],
        }

    dem, _, _, _, gsd_m = _get_terrain_data()
    height, width = dem.shape
    width_m, height_m = width * gsd_m, height * gsd_m
    craters = _reference_craters(width_m, height_m)

    return {
        "sector_id": data_manager.active_sector,
        "width_m": width_m,
        "height_m": height_m,
        "is_synthetic": True,
        "note": "Synthetic landmark map. Catalog craters are real to this project's demo data; the rest are generated.",
        "craters": [c.model_dump() for c in craters],
    }


@app.get("/simulation/basemap")
def get_simulation_basemap():
    """What the simulation is standing on: the real basemap's provenance."""
    basemap = _simulation_basemap()
    described = basemap.describe()
    described["sector_id"] = data_manager.active_sector
    described["n_reference_craters"] = len(_real_reference_craters(basemap))
    return described


@app.get("/simulation/basemap.png")
def get_simulation_basemap_image(max_size: int = 1024):
    """
    The basemap raster itself, as a display PNG.

    A display derivative only - the science raster it came from is never
    served, matching how registration products are handled.
    """
    basemap = _simulation_basemap()
    image = basemap.image
    if max(image.shape[:2]) > max_size:
        scale = max_size / float(max(image.shape[:2]))
        image = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the basemap")
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.get("/reference/nac")
def get_nac_geometry(coverage_grid: int = 9):
    """
    Where the LROC NAC reference image actually sits on the Moon.

    The NAC label carries spacecraft clock counts and no geographic metadata,
    so this is computed from the mission's SPICE kernels rather than read. It
    also answers, rather than assumes, whether the NAC image overlaps the OHRC
    sector: both products are narrow ribbons, and overlapping bounding boxes
    would not settle it.
    """
    from lunar_platform.planetary.spice_adapter import sector_coverage

    camera = _nac_camera()
    kernels = data_manager.get_spice_kernels()

    result = {
        "kernels": kernels.describe() if kernels else None,
        "footprint": camera.footprint(),
        "cross_track_gsd_m": round(camera.ground_sampling_m(), 3),
        "optics": {
            "focal_length_mm": camera.focal_length_mm,
            "pixel_pitch_mm": camera.pixel_pitch_mm,
            "boresight_sample": camera.boresight_sample,
            "distortion_k": camera.distortion_k,
        },
    }

    try:
        basemap = _simulation_basemap()
        result["sector_coverage"] = sector_coverage(camera, basemap, grid=coverage_grid)
    except HTTPException:
        result["sector_coverage"] = None

    return result


@app.post("/perception")
def run_perception(request: PerceptionRequest):
    """
    Run observation -> quadrants -> points -> craters -> match -> position.

    In DEMO mode the observation is rendered from the requested pose, so that
    pose is ground truth and the returned position error is a real measurement
    of this pipeline against it.
    """
    try:
        use_real = _use_real_simulation(request.use_real_data)

        if use_real:
            basemap = _simulation_basemap()
            width_m, height_m = basemap.width_m, basemap.height_m
            camera = _resolve_camera(request, basemap)
            craters = _real_reference_craters(basemap)
        else:
            dem, _, _, _, gsd_m = _get_terrain_data()
            height, width = dem.shape
            width_m, height_m = width * gsd_m, height * gsd_m
            camera = _resolve_camera(request)
            craters = _reference_craters(width_m, height_m)

        # Keep the whole frame on real imagery: a view hanging off the edge is
        # half empty, and the empty half is not terrain.
        margin = max(camera.footprint_m) / 2.0 if use_real else 0.0
        low_x = min(margin, width_m / 2.0)
        low_y = min(margin, height_m / 2.0)
        pose = RoverPose(
            x_m=float(np.clip(request.x_m, low_x, max(low_x, width_m - low_x))),
            y_m=float(np.clip(request.y_m, low_y, max(low_y, height_m - low_y))),
            heading_deg=request.heading_deg,
        )

        if use_real:
            detector = ShadowPairDetector(sun_direction_deg=basemap.sun_direction_deg)
            result = run_real_pipeline(
                pose, camera, basemap, craters, detector, _basemap_localizer()
            )
        else:
            is_synthetic = data_manager.active_mode == DataMode.DEMO
            result = run_pipeline(pose, camera, craters, _detector, is_synthetic=is_synthetic)

        ok, encoded = cv2.imencode(".png", result.image)
        if not ok:
            raise RuntimeError("Failed to encode observation image")

        return {
            "pose": {"x_m": pose.x_m, "y_m": pose.y_m, "heading_deg": pose.heading_deg},
            "camera": {
                "width_px": camera.width_px,
                "height_px": camera.height_px,
                "gsd_m": camera.gsd_m,
                "footprint_m": list(camera.footprint_m),
                "max_range_m": camera.max_range_m,
            },
            "image_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "quadrants": [q.bounds() for q in result.quadrants],
            "points": [p.model_dump() for p in result.points],
            "candidates": [c.model_dump() for c in result.candidates],
            "localization": result.localization.model_dump(),
            "visible_truth": result.visible_truth,
            "stages": [s.model_dump() for s in result.stages],
            "detector": {
                "name": result.detector_name,
                "is_synthetic": result.detector_is_synthetic,
            },
            # Real-imagery fields. Absent in the demo, which has no basemap and
            # no geodetic frame to report a position in.
            "sector": {"width_m": width_m, "height_m": height_m},
            "is_real_imagery": use_real,
            "basemap": result.basemap,
            "correspondences": result.correspondences,
            "geodetic": result.geodetic,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
