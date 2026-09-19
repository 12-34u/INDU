"""
Inspect data/raw and prepare working products for the registration workflow.

Nothing under data/raw is ever modified, and the 1.1 GB OHRC image member is
never extracted. Reading the archive's central directory and its small labels
is cheap; the full image is catalogued and left alone.

Default behaviour is report-only. Anything that writes has to be asked for:

    python scripts/prepare_real_data.py                  # inventory only
    python scripts/prepare_real_data.py --prepare-source # write working crop
    python scripts/prepare_real_data.py --register       # run registration
    python scripts/prepare_real_data.py --self-check     # controlled validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lunar_platform.core.data_manager import DataManager
from lunar_platform.io.raw_inventory import (
    geometry_grid_summary,
    read_geometry_grid,
)
from lunar_platform.registration.pipeline import load_image, run_registration
from lunar_platform.registration.products import write_registration_products
from lunar_platform.registration.validation import (
    apply_illumination_change,
    apply_known_transform,
    transform_error,
)
from lunar_platform.utils.config import get_config


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def report_inventory(dm: DataManager) -> dict:
    inventory = dm.get_raw_inventory(refresh=True)
    print(f"--- data/raw inventory ({inventory['raw_dir']}) ---")
    if not inventory["exists"]:
        print("  data/raw does not exist.")
        return inventory

    for role in ("source_products", "reference_products"):
        print(f"\n{role.replace('_', ' ').upper()}")
        if not inventory[role]:
            print("  none discovered")
        for product in inventory[role]:
            print(f"  {product.product_id}")
            print(f"    path       : {product.path}")
            print(f"    size       : {human_bytes(product.size_bytes)}")
            print(f"    instrument : {product.instrument or 'unknown'}")
            if product.label:
                for key in (
                    "lines",
                    "samples",
                    "pixel_resolution",
                    "start_date_time",
                    "processing_level",
                    "upper_left_latitude",
                    "upper_left_longitude",
                    "lower_right_latitude",
                    "lower_right_longitude",
                    "sun_azimuth",
                ):
                    if key in product.label:
                        print(f"    {key:<11}: {product.label[key]}")
            else:
                print("    label      : none found (no geographic metadata)")
            for note in product.notes:
                print(f"    note       : {note}")

    print(f"\nNAV ancillary: {len(inventory['nav_files'])} file(s)")
    for nav in inventory["nav_files"]:
        print(f"  {nav.split('!')[-1]}")
    print(f"SPICE kernels: {len(inventory['spice_files'])} file(s)")

    print("\nCAPABILITIES")
    capabilities = dm.get_capabilities()
    for key in ("source_image", "reference_image", "navigation", "dem", "spice", "registration"):
        print(f"  {key:<16}: {capabilities[key]}")
    for message in capabilities["messages"]:
        print(f"  ! {message}")
    return inventory


def report_geometry(dm: DataManager) -> None:
    nav_ref = dm.get_navigation_ref()
    if not nav_ref:
        print("\nNo geometry grid available.")
        return

    archive, member = nav_ref.split("!", 1)
    rows = read_geometry_grid(Path(archive), member)
    summary = geometry_grid_summary(rows)
    print("\n--- source geometry grid (NAV) ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(
        "  NOTE: sampled lat/lon per (pixel, scan). This is not a camera model and\n"
        "        does not provide the precise geometry SPICE would."
    )


def prepare_source_crop(dm: DataManager, max_dimension: int) -> Path:
    """Write a working-resolution copy of the source browse image."""
    source_ref = dm.get_source_image_ref()
    if source_ref is None:
        raise SystemExit("No source image discovered; nothing to prepare.")

    out_dir = dm.get_working_registration_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    image, scale = load_image(source_ref, max_dimension)
    out_path = out_dir / "source_working.png"
    cv2.imwrite(str(out_path), image)

    meta = {
        "source_ref": source_ref,
        "decimation": scale,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "note": "Derived from the PDS4 browse product. data/raw is unmodified.",
    }
    (out_dir / "source_working.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_path} ({image.shape[1]}x{image.shape[0]}, decimation {scale:.4f})")
    return out_path


def run_pair_registration(dm: DataManager, max_dimension: int, output_root: Path) -> int:
    capabilities = dm.get_capabilities()
    if not capabilities["registration"]:
        print("\nRegistration blocked: need both a source and a reference image.")
        return 1

    config = get_config("registration")
    config["max_dimension_px"] = max_dimension

    print("\n--- registration: source vs reference ---")
    outcome = run_registration(
        capabilities["source_ref"],
        capabilities["reference_ref"],
        "ch2_ohrc_vs_lronac",
        config,
    )
    for stage in outcome.stages:
        print(f"  [{stage.status:>7}] {stage.label:<32} {stage.detail}")

    metrics = outcome.metrics
    print(f"\n  candidate matches : {metrics['n_candidate_matches']}")
    print(f"  uniform selected  : {metrics['n_uniform_selected']}")
    print(f"  verified inliers  : {metrics['n_verified_inliers']}")
    print(f"  inlier ratio      : {metrics['inlier_ratio']}")
    print(f"  rmse_px           : {metrics['rmse_px']}")
    print(f"  spatial coverage  : {metrics['spatial_coverage']}")
    print(f"  QUALITY PASSED    : {metrics['quality_passed']}")
    for reason in metrics.get("quality_reasons", []):
        print(f"    - {reason}")

    written = write_registration_products(
        outcome, output_root, capabilities["source_ref"], capabilities["reference_ref"]
    )
    print(f"  products -> {written['scene_dir']}")
    return 0


def run_self_check(dm: DataManager, max_dimension: int, output_root: Path) -> int:
    """
    Controlled validation on real OHRC imagery with a known transform.

    This is the only accuracy figure in the project that can be verified,
    because the applied transform is known exactly.
    """
    source_ref = dm.get_source_image_ref()
    if source_ref is None:
        print("\nNo source image discovered; cannot run the self-check.")
        return 1

    config = get_config("registration")
    config["max_dimension_px"] = max_dimension

    base, _ = load_image(source_ref, max_dimension)
    from lunar_platform.preprocessing.normalization import to_uint8

    base = to_uint8(base)
    warped, truth = apply_known_transform(base)
    warped = apply_illumination_change(warped)

    work_dir = dm.get_working_registration_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = work_dir / "selfcheck_fixed.png"
    moving_path = work_dir / "selfcheck_moving.png"
    cv2.imwrite(str(fixed_path), base)
    cv2.imwrite(str(moving_path), warped)

    print("\n--- controlled validation: real OHRC tile, known transform ---")
    print("  applied: rotation 7.0 deg, scale 0.82, translation (14, -9), gamma 1.7")

    outcome = run_registration(str(moving_path), str(fixed_path), "ohrc_self_check", config)
    for stage in outcome.stages:
        print(f"  [{stage.status:>7}] {stage.label:<32} {stage.detail}")

    metrics = outcome.metrics
    print(f"\n  candidate matches : {metrics['n_candidate_matches']}")
    print(f"  uniform selected  : {metrics['n_uniform_selected']}")
    print(f"  verified inliers  : {metrics['n_verified_inliers']}")
    print(f"  inlier ratio      : {metrics['inlier_ratio']}")
    print(f"  rmse_px           : {metrics['rmse_px']}")
    print(f"  spatial coverage  : {metrics['spatial_coverage']}")
    print(f"  QUALITY PASSED    : {metrics['quality_passed']}")
    for reason in metrics.get("quality_reasons", []):
        print(f"    - {reason}")

    if outcome.transform is not None:
        # The pipeline solves moving -> fixed; `truth` was applied fixed -> moving.
        # Comparing them directly would score a correct answer as a large error.
        truth_inverse = np.linalg.inv(truth)
        error = transform_error(outcome.transform, truth_inverse, base.shape[1], base.shape[0])
        metrics["known_transform_error"] = error
        print("\n  recovered vs known transform (displacement over a grid):")
        for key, value in error.items():
            print(f"    {key}: {value}")

    written = write_registration_products(outcome, output_root, str(moving_path), str(fixed_path))
    print(f"  products -> {written['scene_dir']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/registration"))
    parser.add_argument("--max-dimension", type=int, default=2048,
                        help="Working pixel budget per image side (default 2048)")
    parser.add_argument("--geometry", action="store_true", help="Summarise the NAV geometry grid")
    parser.add_argument("--prepare-source", action="store_true",
                        help="Write a working copy of the source browse image")
    parser.add_argument("--register", action="store_true",
                        help="Run source-vs-reference registration")
    parser.add_argument("--self-check", action="store_true",
                        help="Run controlled validation with a known transform")
    args = parser.parse_args()

    dm = DataManager(base_dir=str(args.data_dir))
    report_inventory(dm)

    if args.geometry:
        report_geometry(dm)

    status = 0
    if args.prepare_source:
        prepare_source_crop(dm, args.max_dimension)
    if args.register:
        status |= run_pair_registration(dm, args.max_dimension, args.outputs)
    if args.self_check:
        status |= run_self_check(dm, args.max_dimension, args.outputs)

    if not any([args.geometry, args.prepare_source, args.register, args.self_check]):
        print("\nReport only. Nothing was written.")
        print("Add --geometry, --prepare-source, --register or --self-check to do work.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
