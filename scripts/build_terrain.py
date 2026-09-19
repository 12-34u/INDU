"""
Build the sector DEM, and optionally a crater catalog, for REAL_LOCAL mode.

Elevation cannot be derived from a single monoscopic image. Supply an existing
digital terrain model: LOLA/SLDEM2015, an LRO NAC stereo DTM, or an Ames Stereo
Pipeline output. This script reprojects that product onto the sector's local
metric grid; it does not synthesise elevation.

The output grid is a lunar orthographic projection centred on the sector, so
one pixel is one metre and LocalDEMProvider can read the GSD off the transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform as warp_transform

from lunar_platform.io.raster_loader import open_source_raster, resolve_raster_uri

LUNAR_RADIUS_M = 1737400
LUNAR_GEOGRAPHIC_CRS = f"+proj=longlat +R={LUNAR_RADIUS_M} +no_defs"


def local_metric_crs(center_lat: float, center_lon: float) -> str:
    return (
        f"+proj=ortho +lat_0={center_lat} +lon_0={center_lon} "
        f"+R={LUNAR_RADIUS_M} +units=m +no_defs"
    )


def load_manifest(data_dir: Path, sector_id: str) -> dict:
    manifest_path = data_dir / "sector" / sector_id / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sector manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def sector_geometry(manifest: dict, working_gsd_m: float) -> dict:
    center = manifest["center"]
    size = manifest.get("size", {})
    width_m = float(size.get("width_m", 1000))
    height_m = float(size.get("height_m", 1000))

    return {
        "center_lat": float(center["latitude"]),
        "center_lon": float(center["longitude"]),
        "width_m": width_m,
        "height_m": height_m,
        "gsd_m": working_gsd_m,
        "width_px": int(round(width_m / working_gsd_m)),
        "height_px": int(round(height_m / working_gsd_m)),
    }


def build_dem(dtm_path: Path, out_path: Path, geom: dict, force: bool) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists. Re-run with --force to overwrite.")

    dst_crs = local_metric_crs(geom["center_lat"], geom["center_lon"])
    # The sector is centred on the projection origin, so it spans +/- half its size.
    dst_transform = from_origin(
        -geom["width_m"] / 2.0, geom["height_m"] / 2.0, geom["gsd_m"], geom["gsd_m"]
    )
    destination = np.full((geom["height_px"], geom["width_px"]), np.nan, dtype=np.float32)

    print(f"[dem] reading {resolve_raster_uri(dtm_path)}")
    with open_source_raster(dtm_path) as src:
        print(f"[dem] source: {src.width} x {src.height}, crs={src.crs}, dtype={src.dtypes[0]}")
        if src.crs is None:
            raise ValueError(
                f"{dtm_path.name} has no CRS, so it cannot be reprojected onto the sector grid. "
                "Supply a map-projected DTM."
            )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(destination)
    if not valid.any():
        raise ValueError(
            "The reprojected DEM is entirely empty. The DTM footprint probably does not "
            "cover the sector at "
            f"lat={geom['center_lat']}, lon={geom['center_lon']}."
        )

    coverage = float(valid.mean())
    if coverage < 1.0:
        print(f"[dem] WARNING: only {coverage:.1%} of the sector is covered; filling gaps with the mean")
        destination[~valid] = float(destination[valid].mean())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": geom["width_px"],
        "height": geom["height_px"],
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(destination, 1)

    print(
        f"[dem] wrote {out_path} ({geom['width_px']} x {geom['height_px']} px @ {geom['gsd_m']} m/px, "
        f"elevation {float(destination.min()):.1f} to {float(destination.max()):.1f} m)"
    )


def build_craters(craters_csv: Path, out_path: Path, geom: dict, force: bool) -> None:
    """
    Convert a crater catalog to the local sector frame.

    Accepts either lon/lat columns or x/y columns already in local metres.
    Output x/y are metres from the sector's top-left corner, with y increasing
    downward, which is what calculate_crater_hazard expects.
    """
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists. Re-run with --force to overwrite.")

    with open(craters_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{craters_csv} contains no rows.")

    fields = {name.lower() for name in rows[0]}
    dst_crs = local_metric_crs(geom["center_lat"], geom["center_lon"])
    craters = []

    if {"lon", "lat"} <= fields:
        lons = [float(r["lon"]) for r in rows]
        lats = [float(r["lat"]) for r in rows]
        xs, ys = warp_transform(LUNAR_GEOGRAPHIC_CRS, dst_crs, lons, lats)
    elif {"x", "y"} <= fields:
        xs = [float(r["x"]) - geom["width_m"] / 2.0 for r in rows]
        ys = [geom["height_m"] / 2.0 - float(r["y"]) for r in rows]
    else:
        raise ValueError(
            f"{craters_csv} must have lon/lat columns or x/y columns, plus diameter_m. "
            f"Found: {sorted(fields)}"
        )

    for i, (row, x, y) in enumerate(zip(rows, xs, ys)):
        local_x = x + geom["width_m"] / 2.0
        local_y = geom["height_m"] / 2.0 - y
        if not (0 <= local_x <= geom["width_m"] and 0 <= local_y <= geom["height_m"]):
            continue
        craters.append(
            {
                "id": row.get("id") or f"crater_{i}",
                "x": round(local_x, 2),
                "y": round(local_y, 2),
                "diameter_m": float(row["diameter_m"]),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"craters": craters}, f, indent=2)

    print(f"[craters] wrote {out_path} ({len(craters)} of {len(rows)} inside the sector)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sector", default="sector_001", help="Sector ID (default: sector_001)")
    parser.add_argument("--dtm", type=Path, required=True, help="Source DTM (LOLA/SLDEM, NAC DTM, ASP output)")
    parser.add_argument("--craters", type=Path, help="Optional crater catalog CSV (lon,lat,diameter_m)")
    parser.add_argument("--gsd", type=float, help="Working GSD in m/px (default: from configs/sector.yaml)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing products")
    args = parser.parse_args()

    working_gsd_m = args.gsd
    if working_gsd_m is None:
        config_path = Path("configs/sector.yaml")
        if config_path.exists():
            with open(config_path) as f:
                working_gsd_m = float(yaml.safe_load(f)["sector"].get("working_gsd_m", 1.0))
        else:
            working_gsd_m = 1.0

    manifest = load_manifest(args.data_dir, args.sector)
    geom = sector_geometry(manifest, working_gsd_m)
    sector_dir = args.data_dir / "sector" / args.sector

    print(f"--- Terrain preparation: {args.sector} ---")
    print(
        f"grid: {geom['width_px']} x {geom['height_px']} px @ {geom['gsd_m']} m/px "
        f"centred on lat={geom['center_lat']}, lon={geom['center_lon']}"
    )

    try:
        build_dem(args.dtm, sector_dir / "dem" / "dem.tif", geom, args.force)
        if args.craters:
            build_craters(args.craters, sector_dir / "craters" / "craters.json", geom, args.force)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    craters_path = sector_dir / "craters" / "craters.json"
    if not craters_path.exists():
        craters_path.parent.mkdir(parents=True, exist_ok=True)
        with open(craters_path, "w") as f:
            json.dump({"craters": []}, f, indent=2)
        print(f"[craters] wrote empty {craters_path} (no catalog supplied)")

    imagery_dir = sector_dir / "imagery"
    print("\nTerrain prepared.")
    if not imagery_dir.exists():
        print(f"Still missing: {imagery_dir}")
        print("REAL_LOCAL also needs imagery. Run:")
        print(f"  python scripts/build_sector.py --sector {args.sector} --ohrc <product> --nac <product>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
