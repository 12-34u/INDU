#!/usr/bin/env python3
"""
Cut a small, self-contained sector bundle out of the data/raw archive.

The archive is 6.2 GB and cannot be deployed. Almost none of it is read at
runtime: what the application actually serves is a crop of imagery, a crop of
elevation, and a crater list. This writes exactly those, at a size that fits in
the repository and therefore in the container image.

Nothing under data/raw is modified. The products land in
data/sector/<sector_id>/, which is what DATA_MODE=REAL_LOCAL reads.

    python scripts/build_sector_bundle.py
    python scripts/build_sector_bundle.py --size-m 2400 --imagery-gsd 0.5

Source imagery is the FULL-RESOLUTION OHRC product, not the browse preview.
The browse is a 10x decimation; downsampling the real image to 0.5 m/px gives
genuine half-metre detail, where upsampling the preview would invent it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from lunar_platform.core.data_manager import DataManager
from lunar_platform.crater_detection.shadow_pairs import ShadowPairDetector
from lunar_platform.io.ohrc_browse import load_geometry_grid
from lunar_platform.observation.basemap import build_sector_basemap_from_source
from lunar_platform.registration.pipeline import load_image
from lunar_platform.preprocessing.normalization import to_uint8

# Geometry of the full-resolution OHRC product, from its PDS4 label.
OHRC_FULL_LINES = 93693
OHRC_FULL_SAMPLES = 12000


def full_resolution_path(manager: DataManager) -> Path | None:
    """
    The extracted full-resolution OHRC raster, if it is on disk.

    GDAL cannot open it - it is a flat binary whose geometry lives in a
    sidecar PDS4 label rather than a header - so it is read with a memmap and
    the dimensions the label states.
    """
    for product in manager.get_raw_inventory().get("source_products", []):
        path = Path(product.path)
        if path.suffix.lower() == ".img" and "_d_img_" in path.name:
            return path
    return None


def cut_ohrc(manager, grid, centre_lonlat, size_m, gsd_m, notes):
    """Sector crop of OHRC, from full resolution where available."""
    raster = full_resolution_path(manager)
    if raster is None:
        notes.append(
            "Full-resolution OHRC not on disk; falling back to the browse preview, "
            "which is a 10x decimation."
        )
        return None, None

    native_gsd = 0.2853  # measured from the NAV grid, not the label's nominal 0.24
    pixel, scan = grid.nearest_image_point(*centre_lonlat)
    half = int(size_m / native_gsd / 2)

    row0 = max(0, int(scan) - half)
    col0 = max(0, int(pixel) - half)
    row1 = min(OHRC_FULL_LINES, row0 + 2 * half)
    col1 = min(OHRC_FULL_SAMPLES, col0 + 2 * half)

    memory = np.memmap(
        raster, dtype=np.uint8, mode="r", shape=(OHRC_FULL_LINES, OHRC_FULL_SAMPLES)
    )
    tile = np.array(memory[row0:row1, col0:col1])

    out = max(8, int(size_m / gsd_m))
    resized = cv2.resize(tile, (out, out), interpolation=cv2.INTER_AREA)
    notes.append(
        f"OHRC cut from the full-resolution product at {native_gsd:.4f} m/px "
        f"({tile.shape[1]}x{tile.shape[0]} px) and resampled to {gsd_m} m/px."
    )
    return resized, {"row0": row0, "col0": col0, "native_gsd_m": native_gsd}


def cut_nac(manager, camera, centre_lonlat, size_m, notes):
    """Sector crop of the NAC reference, placed by its SPICE camera model."""
    reference = manager.get_capabilities().get("reference_ref")
    if not reference or camera is None:
        notes.append("No NAC reference or no SPICE; reference crop omitted.")
        return None

    pixel = camera.image_point(*centre_lonlat)
    if pixel is None:
        notes.append("Sector centre falls outside the NAC frame; reference crop omitted.")
        return None

    line, sample = pixel
    gsd = float(camera.ground_sampling_m())
    half = int(size_m / gsd / 2)
    data, _ = load_image(
        reference,
        window=(int(sample) - half, int(line) - half, 2 * half, 2 * half),
        out_shape=(2 * half, 2 * half),
    )
    notes.append(f"NAC cut at its own {gsd:.3f} m/px sampling.")
    return to_uint8(data)


def write_geotiff(path: Path, array: np.ndarray, gsd_m: float, dtype=None):
    """A plain metric GeoTIFF: the app reads pixel size from the transform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array if dtype is None else array.astype(dtype)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=data.shape[0], width=data.shape[1], count=1,
        dtype=data.dtype,
        transform=from_origin(0, 0, gsd_m, gsd_m),
        compress="lzw",
    ) as dst:
        dst.write(data, 1)
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sector", default=None, help="Sector id (default: the active one)")
    parser.add_argument("--size-m", type=float, default=2400.0)
    parser.add_argument("--imagery-gsd", type=float, default=0.5, help="Metres per pixel for the OHRC crop")
    parser.add_argument("--dem-gsd", type=float, default=5.0, help="Metres per pixel for the elevation grid")
    args = parser.parse_args()

    manager = DataManager()
    sector = args.sector or manager.active_sector
    out_root = manager.sector_dir / sector
    notes: list[str] = []

    source = manager.get_simulation_source()
    if source is None:
        raise SystemExit("No Chandrayaan-2 product under data/raw; nothing to cut.")

    centre = manager.get_sector_centre_lonlat()
    if centre is None:
        raise SystemExit(f"{sector} has no centre coordinate in its manifest.")

    grid = load_geometry_grid(
        Path(source.get("archive") or source.get("geometry_path")),
        source.get("geometry_member"),
    )
    basemap = build_sector_basemap_from_source(source, centre[0], centre[1], args.size_m)

    written: dict[str, int] = {}

    # --- imagery -----------------------------------------------------------
    ohrc, provenance = cut_ohrc(manager, grid, centre, args.size_m, args.imagery_gsd, notes)
    if ohrc is not None:
        written["imagery/ohrc_crop.tif"] = write_geotiff(
            out_root / "imagery" / "ohrc_crop.tif", ohrc, args.imagery_gsd
        )

    camera = None
    kernels = manager.get_spice_kernels()
    reference = manager.get_nac_reference()
    if kernels and kernels.is_complete and reference:
        from lunar_platform.planetary.spice_adapter import NacCameraModel, parse_nac_label

        camera = NacCameraModel(
            parse_nac_label(reference["label_path"].read_text("utf-8", "replace"),
                            reference["product_id"]),
            kernels,
        )
    nac = cut_nac(manager, camera, centre, args.size_m, notes)
    if nac is not None:
        written["imagery/lro_nac_reference_crop.tif"] = write_geotiff(
            out_root / "imagery" / "lro_nac_reference_crop.tif",
            nac,
            float(camera.ground_sampling_m()),
        )

    # --- elevation ---------------------------------------------------------
    label = manager.get_raw_dem_label()
    if label is not None:
        from lunar_platform.terrain.lola_dem import LolaPolarDEM

        cells = max(8, int(args.size_m / args.dem_gsd))
        elevation = LolaPolarDEM(label).sample_sector(basemap, output_px=cells)
        written["dem/dem.tif"] = write_geotiff(
            out_root / "dem" / "dem.tif", elevation.elevation_m, args.dem_gsd, dtype=np.float32
        )
        notes.append(
            f"Elevation sampled from {elevation.product} (posted at "
            f"{elevation.source_gsd_m:.0f} m) onto a {cells}x{cells} grid at "
            f"{args.dem_gsd} m/px."
        )

    # --- craters -----------------------------------------------------------
    from lunar_platform.map.reference_map import build_reference_map_from_basemap, to_hazard_dicts

    craters = build_reference_map_from_basemap(
        basemap, ShadowPairDetector(sun_direction_deg=basemap.sun_direction_deg)
    )
    craters_path = out_root / "craters" / "craters.json"
    craters_path.parent.mkdir(parents=True, exist_ok=True)
    craters_path.write_text(json.dumps({"craters": to_hazard_dicts(craters)}, indent=1))
    written["craters/craters.json"] = craters_path.stat().st_size
    notes.append(f"{len(craters)} craters detected in the sector imagery.")

    # --- provenance --------------------------------------------------------
    metadata = {
        "sector_id": sector,
        "built_from": "data/raw (not shipped)",
        "centre": {"longitude": centre[0], "latitude": centre[1]},
        "size_m": args.size_m,
        "imagery_gsd_m": args.imagery_gsd,
        "dem_gsd_m": args.dem_gsd,
        "basemap": basemap.describe(),
        "ohrc_source": provenance,
        "notes": notes,
    }
    meta_path = out_root / "imagery" / "ohrc_crop_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=1, default=str))
    written["imagery/ohrc_crop_metadata.json"] = meta_path.stat().st_size

    print(f"--- sector bundle: {out_root} ---")
    total = 0
    for name, size in written.items():
        total += size
        print(f"  {name:<42} {size / 1e6:8.2f} MB")
    print(f"  {'TOTAL':<42} {total / 1e6:8.2f} MB")
    print()
    for note in notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
