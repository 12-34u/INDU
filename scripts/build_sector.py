"""
Prepare small sector imagery products from large source archives.

Source products are read in place and never modified or extracted; output is
written only under data/sector/<sector_id>/imagery/.

A geographic crop requires a map-projected source. Raw OHRC and LRO NAC EDRs
are in camera space and carry no CRS, so they must be projected first with
ISIS3 (spiceinit followed by cam2map), or cropped explicitly via --pixel-window.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import rasterio
import yaml
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from lunar_platform.io.raster_loader import open_source_raster, resolve_raster_uri

# IAU mean lunar radius. Using an Earth datum here would shift the crop by km.
LUNAR_GEOGRAPHIC_CRS = "+proj=longlat +R=1737400 +no_defs"

SOURCES = {
    "ohrc": {
        "catalog": "ohrc",
        "image": "ohrc_crop.tif",
        "metadata": "ohrc_crop_metadata.json",
    },
    "nac": {
        "catalog": "lro_nac",
        "image": "lro_nac_reference_crop.tif",
        "metadata": "lro_nac_reference_crop_metadata.json",
    },
}


def load_manifest(data_dir: Path, sector_id: str) -> dict:
    manifest_path = data_dir / "sector" / sector_id / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sector manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def load_catalog(data_dir: Path, name: str) -> dict:
    catalog_path = data_dir / "reference_catalog" / f"{name}.yaml"
    if not catalog_path.exists():
        return {}
    with open(catalog_path) as f:
        return yaml.safe_load(f) or {}


def sector_bounds_lonlat(manifest: dict) -> tuple[float, float, float, float]:
    bounds = manifest.get("bounds")
    if not bounds:
        raise ValueError("Sector manifest has no 'bounds' section.")
    return (
        float(bounds["min_longitude"]),
        float(bounds["min_latitude"]),
        float(bounds["max_longitude"]),
        float(bounds["max_latitude"]),
    )


def clamp_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    col_end = min(width, int(window.col_off + window.width))
    row_end = min(height, int(window.row_off + window.height))

    if col_end <= col_off or row_end <= row_off:
        raise ValueError(
            "The sector bounds do not overlap this product. "
            "Check that the manifest bounds and the product footprint refer to "
            "the same region."
        )
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def prepare_source(
    kind: str,
    source_path: Path,
    sector_dir: Path,
    data_dir: Path,
    bounds: tuple[float, float, float, float],
    pixel_window: list[int] | None,
    force: bool,
) -> dict:
    spec = SOURCES[kind]
    imagery_dir = sector_dir / "imagery"
    out_image = imagery_dir / spec["image"]

    if out_image.exists() and not force:
        raise FileExistsError(f"{out_image} already exists. Re-run with --force to overwrite.")

    catalog = load_catalog(data_dir, spec["catalog"])
    uri = resolve_raster_uri(source_path)
    print(f"[{kind}] reading {uri}")

    with open_source_raster(source_path) as src:
        print(f"[{kind}] source: {src.width} x {src.height}, crs={src.crs}, dtype={src.dtypes[0]}")

        if pixel_window is not None:
            col, row, win_w, win_h = pixel_window
            window = clamp_window(Window(col, row, win_w, win_h), src.width, src.height)
            georeferenced = False
        elif src.crs is not None:
            projected = transform_bounds(LUNAR_GEOGRAPHIC_CRS, src.crs, *bounds, densify_pts=21)
            window = clamp_window(
                from_bounds(*projected, transform=src.transform), src.width, src.height
            )
            georeferenced = True
        else:
            raise ValueError(
                f"{source_path.name} has no CRS, so it cannot be cropped by lat/lon.\n"
                "  Either project it first with ISIS3:\n"
                "    spiceinit from=product.cub\n"
                "    cam2map from=product.cub to=product_projected.cub\n"
                "  or crop by pixel with --pixel-window COL ROW WIDTH HEIGHT."
            )

        data = src.read(window=window)
        profile = src.profile.copy()
        # Block sizes inherited from a striped source are not valid for a tiled
        # output, which requires multiples of 16.
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        tiled = int(window.width) >= 256 and int(window.height) >= 256
        profile.update(
            driver="GTiff",
            width=int(window.width),
            height=int(window.height),
            compress="deflate",
            tiled=tiled,
        )
        if tiled:
            profile.update(blockxsize=256, blockysize=256)

        if georeferenced:
            profile.update(transform=src.window_transform(window), crs=src.crs)
            gsd_m = abs(float(src.transform[0]))
        else:
            profile.pop("transform", None)
            profile.pop("crs", None)
            gsd_m = float(catalog.get("gsd_m", 0.0))

        imagery_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_image, "w", **profile) as dst:
            dst.write(data)

    metadata = {
        "image_id": f"{catalog.get('product_id', source_path.stem)}_{sector_dir.name}",
        "sensor": catalog.get("mission", "unknown"),
        "instrument": catalog.get("instrument", "unknown"),
        "width": int(window.width),
        "height": int(window.height),
        "gsd_m_per_px": gsd_m,
        "native_gsd_m": float(catalog.get("gsd_m", gsd_m)),
        "source_path": str(source_path),
        "source_product_id": catalog.get("product_id"),
        "georeferenced": georeferenced,
        "crop_window": {
            "col_off": int(window.col_off),
            "row_off": int(window.row_off),
            "width": int(window.width),
            "height": int(window.height),
        },
        "sector_bounds_lonlat": list(bounds) if georeferenced else None,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }

    out_metadata = imagery_dir / spec["metadata"]
    with open(out_metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{kind}] wrote {out_image} ({window.width} x {window.height} px)")
    if not georeferenced:
        print(f"[{kind}] WARNING: crop is not georeferenced (pixel window on a product with no CRS)")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sector", default="sector_001", help="Sector ID (default: sector_001)")
    parser.add_argument("--ohrc", type=Path, help="Chandrayaan-2 OHRC product (.zip/.img/.xml/.tif)")
    parser.add_argument("--nac", type=Path, help="LRO NAC product (.IMG/.cub/.tif)")
    parser.add_argument(
        "--pixel-window",
        nargs=4,
        type=int,
        metavar=("COL", "ROW", "WIDTH", "HEIGHT"),
        help="Crop by pixel window instead of sector bounds (for products with no CRS)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing sector products")
    args = parser.parse_args()

    if not args.ohrc and not args.nac:
        parser.error("Provide at least one of --ohrc or --nac")

    manifest = load_manifest(args.data_dir, args.sector)
    bounds = sector_bounds_lonlat(manifest)
    sector_dir = args.data_dir / "sector" / args.sector

    print(f"--- Sector preparation: {args.sector} ---")
    print(f"bounds (lon/lat): {bounds}")

    requested = [(k, p) for k, p in (("ohrc", args.ohrc), ("nac", args.nac)) if p]
    for kind, path in requested:
        try:
            prepare_source(
                kind, path, sector_dir, args.data_dir, bounds, args.pixel_window, args.force
            )
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            print(f"[{kind}] FAILED: {exc}", file=sys.stderr)
            return 1

    dem_path = sector_dir / "dem" / "dem.tif"
    print("\nImagery prepared.")
    if not dem_path.exists():
        print(f"Still missing: {dem_path}")
        print("REAL_LOCAL also needs a DEM. Run:")
        print(f"  python scripts/build_terrain.py --sector {args.sector} --dtm <DTM product>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
