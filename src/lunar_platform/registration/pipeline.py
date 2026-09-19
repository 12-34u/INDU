"""
Registration pipeline: source image against reference image.

    load -> normalise -> illumination representation -> multi-scale match
         -> uniform spatial selection -> RANSAC -> sub-pixel refinement
         -> transform -> registered image -> metrics + match-point product

Every stage is measured and reported. The pipeline does not decide whether a
result is scientifically acceptable; it produces numbers and lets the
configured thresholds in configs/registration.yaml say whether they pass.

DEM and SPICE are not used. Without a camera model the estimated transform
relates two image planes, not two positions on the Moon, and the metrics are
image-space residuals rather than ground accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

from ..preprocessing.illumination import apply_representation
from ..preprocessing.normalization import to_uint8, valid_data_fraction
from .distribution import distribution_metrics, select_uniform_matches
from .ground_scale import GroundScale, plan_common_scale, resolve_ground_scale
from .overlap import compute_overlap_window
from .geometry import calculate_spatial_coverage, compute_reprojection_rmse
from .ransac import estimate_transform
from .refine import refine_matches
from .sift import SiftMatcher


@dataclass
class RegistrationStage:
    key: str
    label: str
    status: str  # "ok" | "failed" | "skipped"
    detail: str
    duration_ms: float


@dataclass
class RegistrationOutcome:
    scene_id: str
    succeeded: bool
    transform: Optional[np.ndarray]
    model_type: str
    matches: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    stages: list[RegistrationStage] = field(default_factory=list)
    source_image: Optional[np.ndarray] = None
    reference_image: Optional[np.ndarray] = None
    registered_image: Optional[np.ndarray] = None
    notes: list[str] = field(default_factory=list)


class _Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._start) * 1000.0
        return False


def _raster_shape(ref: str) -> tuple[int, int]:
    """(height, width) of a raster, without reading any pixels."""
    with rasterio.open(ref) as src:
        return src.height, src.width


def load_image(
    ref: str,
    max_dimension: int = 2048,
    window: tuple[int, int, int, int] | None = None,
    out_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """
    Read an image at a decimated resolution, optionally from a window.

    Decimation is driven by pixel budget rather than by the longest side.
    These products are orbital strips - the reference is 2532 x 52224, a 20:1
    aspect ratio - and constraining the long axis alone would crush the short
    axis to a few dozen pixels and destroy every feature in it.

    window is (col_off, row_off, width, height) in source pixels.

    out_shape is (height, width) and overrides the pixel budget entirely. It is
    how a caller brings two products to a COMMON ground scale: the budget
    computes a factor per image independently, which leaves a pair at different
    metres-per-pixel and no scale-sensitive matcher can recover from that.

    Pixels carrying the raster's declared nodata value are returned as NaN.
    This matters more than it sounds: LROC NAC uses -32768, and about 1% of a
    typical window carries it. Left in, that value IS the 1st percentile, so a
    percentile stretch runs from -32768 to a few hundred and maps every real
    surface value to near-white. The image survives every shape and validity
    check and arrives at the matcher as a blank rectangle.

    Returns the array and the decimation factor applied.
    """
    with rasterio.open(ref) as src:
        if window is not None:
            col_off, row_off, win_w, win_h = window
            col_off = max(0, min(col_off, src.width - 1))
            row_off = max(0, min(row_off, src.height - 1))
            win_w = max(1, min(win_w, src.width - col_off))
            win_h = max(1, min(win_h, src.height - row_off))
            read_window = rasterio.windows.Window(col_off, row_off, win_w, win_h)
            height, width = win_h, win_w
        else:
            read_window = None
            height, width = src.height, src.width

        if out_shape is not None:
            out_h, out_w = int(out_shape[0]), int(out_shape[1])
            scale = float(out_w) / max(width, 1)
        else:
            budget = float(max_dimension) ** 2
            scale = min(1.0, float(np.sqrt(budget / max(width * height, 1))))
            out_h = max(8, int(height * scale))
            out_w = max(8, int(width * scale))

        data = src.read(
            1,
            window=read_window,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
        )
        nodata = src.nodata

    if nodata is not None and np.issubdtype(data.dtype, np.number):
        blank = data == nodata
        if blank.any():
            data = data.astype(np.float32)
            data[blank] = np.nan

    return data, scale


def run_registration(
    source_ref: str,
    reference_ref: str,
    scene_id: str,
    config: dict | None = None,
    data_manager=None,
) -> RegistrationOutcome:
    config = config or {}
    representation = config.get("illumination_representation", "clahe")
    model_type = config.get("geometry", {}).get("default_model", "similarity")
    ransac_threshold = float(config.get("geometry", {}).get("ransac_threshold", 3.0))
    max_dimension = int(config.get("max_dimension_px", 2048))

    grid = config.get("match_distribution", {})
    grid_cols = int(grid.get("grid_cols", 8))
    grid_rows = int(grid.get("grid_rows", 8))
    max_per_cell = int(grid.get("max_per_cell", 4))

    stages: list[RegistrationStage] = []
    notes: list[str] = []
    use_prior = bool(config.get("use_geometric_prior", True))

    # ------------------------------------------------------------------
    # Geometric prior: where the two products share ground, and at what
    # common scale. Both are computed before a single feature is extracted,
    # because a matcher cannot recover from a pair that is misaligned in
    # scale or barely overlapping - it just returns nothing.
    # ------------------------------------------------------------------
    with _Timer() as t:
        source_shape_full = _raster_shape(source_ref)
        reference_shape_full = _raster_shape(reference_ref)

        overlap = None
        if use_prior:
            overlap = compute_overlap_window(
                source_ref, reference_ref, reference_shape_full, data_manager
            )

        reference_window = overlap.as_tuple if overlap else None
        source_window = overlap.source_window if overlap else None

        source_shape = (
            (source_window[3], source_window[2]) if source_window else source_shape_full
        )
        reference_shape = (
            (reference_window[3], reference_window[2]) if reference_window else reference_shape_full
        )

        source_gsd = resolve_ground_scale(
            source_ref,
            explicit_gsd_m=config.get("source_gsd_m"),
            data_manager=data_manager,
        ) if use_prior else GroundScale(None, "disabled", "geometric prior disabled")
        reference_gsd = resolve_ground_scale(
            reference_ref,
            explicit_gsd_m=config.get("reference_gsd_m"),
            data_manager=data_manager,
        ) if use_prior else GroundScale(None, "disabled", "geometric prior disabled")

        plan = plan_common_scale(
            source_ref,
            reference_ref,
            source_shape,
            reference_shape,
            max_dimension,
            source_gsd,
            reference_gsd,
        )
    stages.append(
        RegistrationStage(
            "prior",
            "Geometric prior",
            "ok" if (overlap or plan.equalised) else "skipped",
            (
                (overlap.note + " " if overlap else "No overlap prior available. ")
                + plan.note
            ),
            round(t.ms, 2),
        )
    )
    if overlap is None and use_prior:
        notes.append(
            "No overlap prior: the products' geometry could not place one in the other, "
            "so the whole reference is searched."
        )

    with _Timer() as t:
        source_raw, source_scale = load_image(
            source_ref, max_dimension, window=source_window, out_shape=plan.source_out_shape
        )
        reference_raw, reference_scale = load_image(
            reference_ref,
            max_dimension,
            window=reference_window,
            out_shape=plan.reference_out_shape,
        )
    stages.append(
        RegistrationStage(
            "load",
            "Images loaded",
            "ok",
            f"source {source_raw.shape[1]}x{source_raw.shape[0]} (decimated {source_scale:.3f}), "
            f"reference {reference_raw.shape[1]}x{reference_raw.shape[0]} (decimated {reference_scale:.3f})"
            + (f", working {plan.working_gsd_m:.2f} m/px" if plan.working_gsd_m else ""),
            round(t.ms, 2),
        )
    )

    with _Timer() as t:
        source_8 = to_uint8(source_raw)
        reference_8 = to_uint8(reference_raw)
        # Measured on the arrays as read, so nodata is counted as missing
        # rather than as the black it becomes after the stretch.
        source_valid = valid_data_fraction(source_raw)
        reference_valid = valid_data_fraction(reference_raw)
    stages.append(
        RegistrationStage(
            "normalize",
            "Radiometric normalisation",
            "ok",
            f"valid data source {source_valid:.1%}, reference {reference_valid:.1%}",
            round(t.ms, 2),
        )
    )

    with _Timer() as t:
        source_prepared = apply_representation(source_8, representation)
        reference_prepared = apply_representation(reference_8, representation)
    stages.append(
        RegistrationStage(
            "illumination",
            "Illumination handling",
            "ok",
            f"representation '{representation}' applied to both images",
            round(t.ms, 2),
        )
    )

    with _Timer() as t:
        matcher = SiftMatcher(
            nfeatures=int(config.get("nfeatures", 8000)),
            ratio_threshold=float(config.get("ratio_threshold", 0.75)),
        )
        pts_source, pts_reference, scores, matcher_name = matcher.match(
            source_prepared, reference_prepared
        )
    n_candidates = len(pts_source)
    stages.append(
        RegistrationStage(
            "matching",
            "Feature matching",
            "ok" if n_candidates else "failed",
            f"{n_candidates} candidate matches from {matcher_name}",
            round(t.ms, 2),
        )
    )

    if n_candidates < 3:
        notes.append(
            "Too few candidate matches to attempt geometric verification. "
            "With no camera model there is no guarantee the two products overlap."
        )
        return RegistrationOutcome(
            scene_id=scene_id,
            succeeded=False,
            transform=None,
            model_type=model_type,
            metrics=_empty_metrics(n_candidates, representation, model_type),
            stages=stages,
            source_image=source_8,
            reference_image=reference_8,
            notes=notes,
        )

    with _Timer() as t:
        keep = select_uniform_matches(
            pts_source,
            pts_reference,
            scores,
            source_prepared.shape[1],
            source_prepared.shape[0],
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            max_per_cell=max_per_cell,
        )
        pts_source_u = pts_source[keep]
        pts_reference_u = pts_reference[keep]
        scores_u = scores[keep]
    stages.append(
        RegistrationStage(
            "distribution",
            "Uniform spatial selection",
            "ok",
            f"{len(keep)} of {n_candidates} kept ({grid_cols}x{grid_rows} grid, "
            f"max {max_per_cell}/cell)",
            round(t.ms, 2),
        )
    )

    with _Timer() as t:
        transform, inlier_mask = estimate_transform(
            pts_source_u, pts_reference_u, model_type, ransac_threshold
        )
        n_inliers = int(np.count_nonzero(inlier_mask))
    stages.append(
        RegistrationStage(
            "ransac",
            "Geometric verification",
            "ok" if n_inliers >= 3 else "failed",
            f"{n_inliers} verified inliers of {len(pts_source_u)} ({model_type}, "
            f"threshold {ransac_threshold} px)",
            round(t.ms, 2),
        )
    )

    refine_scores = np.zeros(len(pts_source_u))
    n_refined = 0
    n_inliers_before_refine = n_inliers
    if n_inliers >= 3 and config.get("subpixel_refinement", True):
        with _Timer() as t:
            refined_ref, refine_scores_in = refine_matches(
                source_prepared,
                reference_prepared,
                pts_source_u[inlier_mask],
                pts_reference_u[inlier_mask],
            )
            moved = np.linalg.norm(refined_ref - pts_reference_u[inlier_mask], axis=1)
            n_refined = int(np.count_nonzero(moved > 1e-6))

            pts_reference_u = pts_reference_u.copy()
            pts_reference_u[inlier_mask] = refined_ref
            refine_scores[inlier_mask] = refine_scores_in

            # Re-estimate on refined correspondences.
            transform, inlier_mask = estimate_transform(
                pts_source_u, pts_reference_u, model_type, ransac_threshold
            )
            n_inliers = int(np.count_nonzero(inlier_mask))
        stages.append(
            RegistrationStage(
                "refine",
                "Sub-pixel refinement",
                "ok",
                f"{n_refined} of {n_inliers_before_refine} inliers refined by local NCC; "
                f"{int(inlier_mask.sum())} inliers after re-estimation",
                round(t.ms, 2),
            )
        )
    else:
        stages.append(
            RegistrationStage(
                "refine",
                "Sub-pixel refinement",
                "skipped",
                "too few inliers to refine" if n_inliers < 3 else "disabled by config",
                0.0,
            )
        )

    quality_passed, quality_reasons = _evaluate_quality_gate(
        n_candidates, n_inliers, len(pts_source_u), transform, inlier_mask,
        pts_source_u, pts_reference_u, source_prepared.shape, model_type, config
    )
    succeeded = quality_passed
    if not quality_passed:
        notes.extend(quality_reasons)

    metrics = _compute_metrics(
        pts_source_u,
        pts_reference_u,
        inlier_mask,
        transform,
        source_prepared.shape,
        n_candidates,
        representation,
        model_type,
        grid_cols,
        grid_rows,
    )
    metrics["source_decimation"] = round(source_scale, 5)
    metrics["reference_decimation"] = round(reference_scale, 5)
    metrics["n_subpixel_refined"] = n_refined
    metrics["quality_passed"] = quality_passed
    metrics["quality_reasons"] = quality_reasons

    registered = None
    if succeeded:
        with _Timer() as t:
            registered = cv2.warpPerspective(
                source_8,
                transform,
                (reference_8.shape[1], reference_8.shape[0]),
                flags=cv2.INTER_LINEAR,
            )
        stages.append(
            RegistrationStage(
                "warp",
                "Registered image produced",
                "ok",
                f"source warped into reference frame {registered.shape[1]}x{registered.shape[0]}",
                round(t.ms, 2),
            )
        )
    else:
        stages.append(
            RegistrationStage(
                "warp",
                "Registered image produced",
                "skipped",
                "no verified transform",
                0.0,
            )
        )
        notes.append(
            "Geometric verification did not converge. Reported values are "
            "measurements of this attempt, not a registered result."
        )

    matches = _build_match_records(
        pts_source_u, pts_reference_u, scores_u, refine_scores, inlier_mask, transform
    )

    return RegistrationOutcome(
        scene_id=scene_id,
        succeeded=succeeded,
        transform=transform if succeeded else None,
        model_type=model_type,
        matches=matches,
        metrics=metrics,
        stages=stages,
        source_image=source_8,
        reference_image=reference_8,
        registered_image=registered,
        notes=notes,
    )


# Minimum correspondences each model needs. A fit using only this many passes
# through its own points almost exactly, producing a near-zero RMSE that says
# nothing about whether the images are aligned.
MODEL_MIN_SAMPLES = {"similarity": 2, "affine": 3, "homography": 4}
DEGENERACY_FACTOR = 3


def _evaluate_quality_gate(
    n_candidates: int,
    n_inliers: int,
    n_selected: int,
    transform: np.ndarray,
    inlier_mask: np.ndarray,
    pts_source: np.ndarray,
    pts_reference: np.ndarray,
    source_shape: tuple,
    model_type: str,
    config: dict,
) -> tuple[bool, list[str]]:
    """
    Apply the configured thresholds, plus a guard against degenerate fits.

    Returns (passed, reasons_it_failed). Thresholds come from
    configs/registration.yaml and are not scientifically validated values.
    """
    reasons: list[str] = []

    min_matches = int(config.get("min_matches", 30))
    min_inlier_ratio = float(config.get("min_inlier_ratio", 0.35))
    max_rmse = float(config.get("max_rmse_px", 5.0))
    min_coverage = float(config.get("min_spatial_coverage", 0.30))

    minimum_sample = MODEL_MIN_SAMPLES.get(model_type, 4)
    degeneracy_floor = minimum_sample * DEGENERACY_FACTOR

    if n_inliers < degeneracy_floor:
        reasons.append(
            f"Only {n_inliers} verified inliers for a {model_type} model that needs "
            f"{minimum_sample}; below the degeneracy floor of {degeneracy_floor}, so the "
            "fit is unconstrained and its residuals are not meaningful."
        )

    if n_candidates < min_matches:
        reasons.append(f"{n_candidates} candidate matches is below the configured minimum of {min_matches}.")

    inlier_ratio = (n_inliers / n_selected) if n_selected else 0.0
    if inlier_ratio < min_inlier_ratio:
        reasons.append(
            f"Inlier ratio {inlier_ratio:.3f} is below the configured minimum of {min_inlier_ratio}."
        )

    if n_inliers:
        height, width = source_shape[:2]
        coverage = calculate_spatial_coverage(pts_source[inlier_mask], width, height)
        if coverage < min_coverage:
            reasons.append(
                f"Inlier spatial coverage {coverage:.3f} is below the configured minimum "
                f"of {min_coverage}; the transform is constrained in only part of the frame."
            )
        rmse = compute_reprojection_rmse(
            pts_source[inlier_mask], pts_reference[inlier_mask], transform
        )
        if rmse > max_rmse:
            reasons.append(f"RMSE {rmse:.3f} px exceeds the configured maximum of {max_rmse}.")
    else:
        reasons.append("No verified inliers.")

    return (len(reasons) == 0), reasons


def _empty_metrics(n_candidates: int, representation: str, model_type: str) -> dict:
    return {
        "n_candidate_matches": int(n_candidates),
        "n_uniform_selected": 0,
        "n_verified_inliers": 0,
        "inlier_ratio": 0.0,
        "rmse_px": None,
        "median_error_px": None,
        "max_error_px": None,
        "spatial_coverage": 0.0,
        "illumination_representation": representation,
        "model_type": model_type,
        "quality_passed": False,
        "quality_reasons": ["No verified inliers."],
        "source_decimation": None,
        "reference_decimation": None,
        "n_subpixel_refined": 0,
    }


def _compute_metrics(
    pts_source: np.ndarray,
    pts_reference: np.ndarray,
    inlier_mask: np.ndarray,
    transform: np.ndarray,
    source_shape: tuple,
    n_candidates: int,
    representation: str,
    model_type: str,
    grid_cols: int,
    grid_rows: int,
) -> dict:
    height, width = source_shape[:2]
    n_selected = len(pts_source)
    n_inliers = int(np.count_nonzero(inlier_mask))

    metrics = _empty_metrics(n_candidates, representation, model_type)
    metrics["n_uniform_selected"] = n_selected
    metrics["n_verified_inliers"] = n_inliers
    metrics["inlier_ratio"] = round(n_inliers / n_selected, 4) if n_selected else 0.0

    if n_inliers:
        inlier_source = pts_source[inlier_mask]
        inlier_reference = pts_reference[inlier_mask]

        homogeneous = np.hstack([inlier_source, np.ones((n_inliers, 1))])
        projected = (transform @ homogeneous.T).T
        projected = projected[:, :2] / projected[:, 2:]
        errors = np.linalg.norm(projected - inlier_reference, axis=1)

        metrics["rmse_px"] = round(
            float(compute_reprojection_rmse(inlier_source, inlier_reference, transform)), 4
        )
        metrics["median_error_px"] = round(float(np.median(errors)), 4)
        metrics["max_error_px"] = round(float(errors.max()), 4)
        metrics["spatial_coverage"] = round(
            float(calculate_spatial_coverage(inlier_source, width, height)), 4
        )
        metrics["inlier_distribution"] = distribution_metrics(
            inlier_source, width, height, grid_cols, grid_rows
        )

    metrics["candidate_distribution"] = distribution_metrics(
        pts_source, width, height, grid_cols, grid_rows
    )
    return metrics


def _build_match_records(
    pts_source: np.ndarray,
    pts_reference: np.ndarray,
    scores: np.ndarray,
    refine_scores: np.ndarray,
    inlier_mask: np.ndarray,
    transform: np.ndarray,
) -> list[dict]:
    records: list[dict] = []
    if len(pts_source) == 0:
        return records

    homogeneous = np.hstack([pts_source, np.ones((len(pts_source), 1))])
    projected = (transform @ homogeneous.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = projected[:, :2] / projected[:, 2:]
    errors = np.linalg.norm(projected - pts_reference, axis=1)

    for i in range(len(pts_source)):
        records.append(
            {
                "index": i,
                "source_x": round(float(pts_source[i, 0]), 3),
                "source_y": round(float(pts_source[i, 1]), 3),
                "reference_x": round(float(pts_reference[i, 0]), 3),
                "reference_y": round(float(pts_reference[i, 1]), 3),
                "error_px": round(float(errors[i]), 4) if np.isfinite(errors[i]) else None,
                "inlier": bool(inlier_mask[i]),
                "match_score": round(float(scores[i]), 5),
                "refine_correlation": round(float(refine_scores[i]), 5),
            }
        )
    return records
