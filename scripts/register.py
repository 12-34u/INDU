#!/usr/bin/env python3
"""
Register a source image against a reference image.

    python scripts/register.py --source <ref> --reference <ref> [options]

The problem statement asks for a generic solution, so this takes any pair GDAL
can open and does not assume the Chandrayaan-2 / LRO products this repository
happens to carry. Run with no arguments and it falls back to whatever it finds
under data/raw, which is the common case while developing.

Anything the pipeline can measure about a pair it reports, whether or not the
run passed: a registration that fails is a result, and the numbers that show
why it failed are the useful part. Nothing under data/raw is written to.

Examples
--------
    # Whatever is under data/raw
    python scripts/register.py

    # An explicit pair, with the ground scales supplied
    python scripts/register.py --source a.tif --reference b.tif \
        --source-gsd 2.85 --reference-gsd 1.30

    # A controlled check: warp an image by a known transform and recover it
    python scripts/register.py --self-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lunar_platform.core.data_manager import DataManager
from lunar_platform.preprocessing.illumination import REPRESENTATIONS
from lunar_platform.preprocessing.normalization import to_uint8
from lunar_platform.registration.pipeline import load_image, run_registration
from lunar_platform.registration.products import write_registration_products
from lunar_platform.registration.validation import (
    apply_illumination_change,
    apply_known_transform,
    transform_error,
)
from lunar_platform.utils.config import get_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", help="Moving image. Any GDAL-readable reference.")
    parser.add_argument("--reference", help="Fixed image. Any GDAL-readable reference.")
    parser.add_argument("--scene-id", default=None, help="Name for the output directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/registration"))

    parser.add_argument(
        "--source-gsd",
        type=float,
        help="Metres per pixel of the source, when it cannot be derived from the product",
    )
    parser.add_argument("--reference-gsd", type=float, help="Metres per pixel of the reference")
    parser.add_argument(
        "--max-dimension", type=int, default=1600, help="Working pixel budget per image side"
    )
    parser.add_argument(
        "--representation",
        choices=sorted(REPRESENTATIONS),
        help="Illumination representation. Omit to use the configured default.",
    )
    parser.add_argument(
        "--compare-representations",
        action="store_true",
        help="Run every representation and report which found the most inliers",
    )
    parser.add_argument(
        "--model",
        choices=("similarity", "affine", "homography"),
        help="Geometric model to verify with",
    )
    parser.add_argument(
        "--no-prior",
        action="store_true",
        help="Disable the scale and overlap prior (matches the pre-prior behaviour)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Register the source against a known warp of itself, where truth is exact",
    )
    parser.add_argument("--json", action="store_true", help="Emit metrics as JSON only")
    return parser


def resolve_pair(args, manager: DataManager) -> tuple[str, str]:
    """The pair to register: what was asked for, else what data/raw holds."""
    if args.source and args.reference:
        return args.source, args.reference

    capabilities = manager.get_capabilities()
    source = args.source or capabilities.get("source_ref")
    reference = args.reference or capabilities.get("reference_ref")

    if not source or not reference:
        raise SystemExit(
            "Need a source and a reference. Pass --source and --reference, or place "
            "products under data/raw. Discovered: "
            f"source={source or 'none'}, reference={reference or 'none'}."
        )
    return source, reference


def build_config(args) -> dict:
    config = get_config("registration") or {}
    config["max_dimension_px"] = args.max_dimension
    config["use_geometric_prior"] = not args.no_prior
    if args.representation:
        config["illumination_representation"] = args.representation
    if args.model:
        config.setdefault("geometry", {})["default_model"] = args.model
    if args.source_gsd:
        config["source_gsd_m"] = args.source_gsd
    if args.reference_gsd:
        config["reference_gsd_m"] = args.reference_gsd
    return config


def make_self_check_pair(source_ref: str, manager: DataManager, max_dimension: int):
    """
    Warp a real tile by a known transform, so accuracy has a truth to measure.

    This is the only figure in the project that can be verified without a
    camera model: the applied transform is known exactly, so the recovered one
    can be compared against it rather than against its own residuals.
    """
    base, _ = load_image(source_ref, max_dimension)
    base = to_uint8(base)
    warped, truth = apply_known_transform(base)
    warped = apply_illumination_change(warped)

    work = manager.get_working_registration_dir()
    work.mkdir(parents=True, exist_ok=True)
    fixed = work / "selfcheck_fixed.png"
    moving = work / "selfcheck_moving.png"
    cv2.imwrite(str(fixed), base)
    cv2.imwrite(str(moving), warped)
    return str(moving), str(fixed), truth, base


def report(outcome, verbose: bool = True) -> None:
    if not verbose:
        return
    print(f"\n--- registration: {outcome.scene_id} ---")
    for stage in outcome.stages:
        print(f"  [{stage.status:>7}] {stage.label:<34} {stage.detail}")

    metrics = outcome.metrics
    print()
    for key in (
        "n_candidate_matches",
        "n_uniform_selected",
        "n_verified_inliers",
        "inlier_ratio",
        "rmse_px",
        "median_error_px",
        "max_error_px",
        "spatial_coverage",
        "n_subpixel_refined",
    ):
        if key in metrics:
            print(f"  {key:<22} {metrics[key]}")

    spread = metrics.get("inlier_distribution")
    if spread:
        print(
            f"  {'distribution':<22} {spread['occupied_cells']}/{spread['total_cells']} cells, "
            f"gini {spread['gini']}"
        )

    print(f"\n  QUALITY PASSED        {metrics.get('quality_passed')}")
    for reason in metrics.get("quality_reasons", []):
        print(f"    - {reason}")


def compare_representations(source_ref, reference_ref, config, scene_id) -> None:
    """
    Measure every representation on this pair instead of trusting one.

    Which representation survives a given illumination difference is a property
    of the pair, not something that can be decided in advance.
    """
    print("\n--- illumination representations, measured on this pair ---")
    print(f"  {'representation':<26}{'matches':>9}{'inliers':>9}{'ratio':>8}{'rmse_px':>10}")

    results = []
    for name in sorted(REPRESENTATIONS):
        trial = dict(config)
        trial["illumination_representation"] = name
        outcome = run_registration(source_ref, reference_ref, f"{scene_id}_{name}", trial)
        m = outcome.metrics
        results.append((name, m.get("n_verified_inliers", 0)))
        print(
            f"  {name:<26}{m.get('n_candidate_matches', 0):>9}"
            f"{m.get('n_verified_inliers', 0):>9}{m.get('inlier_ratio', 0):>8.3f}"
            f"{(m.get('rmse_px') if m.get('rmse_px') is not None else float('nan')):>10.3f}"
        )

    best = max(results, key=lambda r: r[1])
    print(f"\n  best by verified inliers: {best[0]} ({best[1]})")


def main() -> int:
    args = build_parser().parse_args()
    manager = DataManager()
    config = build_config(args)

    truth = None
    base = None
    if args.self_check:
        discovered_source = resolve_pair(args, manager)[0]
        source_ref, reference_ref, truth, base = make_self_check_pair(
            discovered_source, manager, args.max_dimension
        )
        scene_id = args.scene_id or "self_check"
    else:
        source_ref, reference_ref = resolve_pair(args, manager)
        scene_id = args.scene_id or "registration"

    if args.compare_representations:
        compare_representations(source_ref, reference_ref, config, scene_id)
        return 0

    outcome = run_registration(
        source_ref, reference_ref, scene_id, config, data_manager=manager
    )

    if truth is not None and outcome.transform is not None:
        # The pipeline solves moving -> fixed; the truth was applied
        # fixed -> moving, so comparing them directly would score a correct
        # answer as a large error.
        error = transform_error(
            outcome.transform, np.linalg.inv(truth), base.shape[1], base.shape[0]
        )
        outcome.metrics["known_transform_error"] = error

    report(outcome, verbose=not args.json)

    written = write_registration_products(outcome, args.out, source_ref, reference_ref)

    if args.json:
        print(json.dumps(written["manifest"], indent=2, default=str))
    else:
        known = outcome.metrics.get("known_transform_error")
        if known:
            limit = float(config.get("max_accuracy_rmse_px", 1.0))
            achieved = float(known.get("rmse_px", float("inf")))
            print("\n  ACCURACY vs KNOWN transform (displacement over a grid):")
            for key, value in known.items():
                print(f"    {key:<12} {value}")
            verdict = "YES" if achieved <= limit else "NO"
            print(f"\n    SUB-PIXEL ({limit} px): {verdict}  -  achieved {achieved:.3f} px RMSE")
            print(
                "    This is registration accuracy against truth. The rmse_px reported\n"
                "    above it is the residual of the matched points, which carries match\n"
                "    noise as well as model error and is a different quantity."
            )
        print(f"\n  products -> {written['scene_dir']}")

    return 0 if outcome.metrics.get("quality_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
