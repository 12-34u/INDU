"""
End-to-end perception and localisation pipeline.

    observation -> quadrants -> points -> crater candidates -> match -> position

Every stage below actually runs; nothing is replayed from the ground truth used
to render the observation. The rendering pose is passed in only so the recovered
position can be scored against it, and it is never consulted by the detector,
the aggregator or the matcher.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..crater_detection.base import PointDetector, detect_all
from ..crater_mapping.aggregation import aggregate_points
from ..crater_mapping.models import (
    CraterCandidate,
    DetectedPoint,
    LocalizationResult,
    PipelineStage,
    ReferenceCrater,
)
from ..map.reference_map import craters_in_radius
from ..observation.quadrants import QuadrantView, split_quadrants
from ..observation.real_view import render_real_observation
from ..observation.synthetic_view import render_observation
from ..simulation.camera import NadirCamera, RoverPose
from .matching import match_constellation


@dataclass
class PerceptionResult:
    image: np.ndarray
    quadrants: list[QuadrantView]
    points: list[DetectedPoint]
    candidates: list[CraterCandidate]
    localization: LocalizationResult
    visible_truth: list[dict] = field(default_factory=list)
    stages: list[PipelineStage] = field(default_factory=list)
    detector_name: str = ""
    detector_is_synthetic: bool = True
    # Real-mode only. Empty in the synthetic path, which has no basemap and
    # fixes its position from crater correspondences rather than image texture.
    basemap: dict = field(default_factory=dict)
    correspondences: list[dict] = field(default_factory=list)
    geodetic: dict = field(default_factory=dict)


class _Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._start) * 1000.0
        return False


def run_pipeline(
    pose: RoverPose,
    camera: NadirCamera,
    reference_craters: list[ReferenceCrater],
    detector: PointDetector,
    is_synthetic: bool = True,
) -> PerceptionResult:
    stages: list[PipelineStage] = []

    with _Timer() as t:
        image, visible_truth = render_observation(pose, camera, reference_craters)
    stages.append(
        PipelineStage(
            key="observation",
            label="Image received",
            status="ok",
            detail=f"{camera.width_px}x{camera.height_px} px at {camera.gsd_m} m/px (synthetic)",
            duration_ms=round(t.ms, 2),
        )
    )

    with _Timer() as t:
        quadrants = split_quadrants(image)
    stages.append(
        PipelineStage(
            key="quadrants",
            label="Quadrants generated",
            status="ok",
            detail=f"{len(quadrants)} quadrants, {quadrants[0].width}x{quadrants[0].height} px each",
            duration_ms=round(t.ms, 2),
        )
    )

    with _Timer() as t:
        points = detect_all(detector, quadrants)
    per_quadrant = {q.name: sum(1 for p in points if p.quadrant == q.name) for q in quadrants}
    stages.append(
        PipelineStage(
            key="points",
            label="Points detected",
            status="ok" if points else "failed",
            detail=" ".join(f"{k}:{v}" for k, v in per_quadrant.items()) or "no points found",
            duration_ms=round(t.ms, 2),
        )
    )

    with _Timer() as t:
        candidates = aggregate_points(points, camera, pose.heading_deg)
    stages.append(
        PipelineStage(
            key="aggregation",
            label="Points aggregated",
            status="ok" if candidates else "failed",
            detail=f"{len(points)} points into {len(candidates)} crater candidates",
            duration_ms=round(t.ms, 2),
        )
    )

    # Only craters the camera could actually see are eligible; scoring against
    # the whole sector would understate how hard the match is.
    in_range = craters_in_radius(reference_craters, pose.x_m, pose.y_m, camera.max_range_m)

    with _Timer() as t:
        solution = match_constellation(candidates, reference_craters)
    matched = len(solution.matches)
    stages.append(
        PipelineStage(
            key="matching",
            label="Craters matched to reference map",
            status="ok" if matched else "failed",
            detail=f"{matched}/{len(candidates)} candidates matched from {solution.n_votes} pose votes",
            duration_ms=round(t.ms, 2),
        )
    )

    error_m = None
    if solution.estimated_x_m is not None:
        error_m = float(
            np.hypot(solution.estimated_x_m - pose.x_m, solution.estimated_y_m - pose.y_m)
        )

    if solution.estimated_x_m is None:
        note = "Match failed: too few consistent crater correspondences to fix a position."
    elif is_synthetic:
        note = "Demo localization. Synthetic observation, error measured against the rendering pose."
    else:
        note = "Demo localization. No ground truth available outside synthetic mode."

    stages.append(
        PipelineStage(
            key="localization",
            label="Rover position estimated",
            status="ok" if solution.estimated_x_m is not None else "failed",
            detail=(
                f"error {error_m:.2f} m against ground truth"
                if error_m is not None
                else "no position fix"
            ),
            duration_ms=0.0,
        )
    )

    localization = LocalizationResult(
        estimated_x_m=solution.estimated_x_m,
        estimated_y_m=solution.estimated_y_m,
        ground_truth_x_m=pose.x_m if is_synthetic else None,
        ground_truth_y_m=pose.y_m if is_synthetic else None,
        position_error_m=error_m if is_synthetic else None,
        heading_deg=pose.heading_deg,
        matches=solution.matches,
        n_candidates=len(candidates),
        n_reference_in_range=len(in_range),
        match_confidence=round(solution.inlier_ratio, 3) if solution.matches else None,
        method="constellation-translation-vote",
        is_synthetic=is_synthetic,
        note=note,
    )

    return PerceptionResult(
        image=image,
        quadrants=quadrants,
        points=points,
        candidates=candidates,
        localization=localization,
        visible_truth=visible_truth,
        stages=stages,
        detector_name=detector.name,
        detector_is_synthetic=detector.is_synthetic,
    )


def craters_to_candidates(
    craters, camera: NadirCamera, heading_deg: float, gsd_m: float
) -> list[CraterCandidate]:
    """
    Express shadow-pair craters as candidates the rest of the system understands.

    The offsets are where the crater sits relative to the rover, which is all
    the matcher and the UI ever need. Fit statistics that a circle fit would
    supply do not exist here, so the fields that would carry them report what
    this detector actually measured rather than a plausible-looking number.
    """
    candidates: list[CraterCandidate] = []
    pose = RoverPose(0.0, 0.0, heading_deg)

    for index, crater in enumerate(craters, start=1):
        dx, dy = camera.image_to_offset(pose, crater.center_u, crater.center_v)
        # Score is an unbounded alignment-times-size product; squashing it
        # keeps the UI's 0-1 confidence meaningful without implying it is a
        # detection probability.
        confidence = float(crater.score / (crater.score + 8.0))
        candidates.append(
            CraterCandidate(
                id=f"C{index:02d}",
                center_u=round(float(crater.center_u), 2),
                center_v=round(float(crater.center_v), 2),
                radius_px=round(float(crater.radius_px), 2),
                diameter_px=round(float(crater.radius_px * 2.0), 2),
                n_points=int(crater.n_points),
                fit_rmse_px=0.0,
                confidence=round(confidence, 3),
                source_quadrants=_quadrants_spanned(crater, camera),
                offset_dx_m=round(float(dx), 3),
                offset_dy_m=round(float(dy), 3),
                diameter_m=round(float(crater.radius_px * 2.0 * gsd_m), 2),
                provenance="detected",
            )
        )
    return candidates


def _quadrants_spanned(crater, camera: NadirCamera) -> list:
    """Which quadrants the crater's shadow and highlight actually fall in."""
    mid_x, mid_y = camera.width_px / 2.0, camera.height_px / 2.0
    names = set()
    for u, v in (
        (crater.shadow_u, crater.shadow_v),
        (crater.highlight_u, crater.highlight_v),
        (crater.center_u, crater.center_v),
    ):
        if not (0 <= u < camera.width_px and 0 <= v < camera.height_px):
            continue
        if v < mid_y:
            names.add("Q1" if u < mid_x else "Q2")
        else:
            names.add("Q3" if u < mid_x else "Q4")
    return sorted(names)


def associate_with_reference(
    candidates: list[CraterCandidate],
    references: list[ReferenceCrater],
    estimated_x_m: float,
    estimated_y_m: float,
    tolerance_m: float = 40.0,
    diameter_tolerance: float = 0.6,
):
    """
    Tie observed craters to mapped craters once the position is known.

    This runs AFTER the fix and plays no part in producing it, which is the
    opposite of the synthetic path where the correspondences are what yields
    the position. It exists so the crater map can show which observed crater is
    which mapped crater; reading it as evidence for the fix would be circular.

    The tolerance is deliberately tight and size has to agree as well as
    position. Relaxing it does raise the number of associations, but on a
    surface this densely cratered a loose tolerance pairs a crater with
    whichever neighbour happens to be nearby, which looks like agreement and
    is not. Expect a minority of observed craters to associate: a detection
    repeats between two views of the same ground only about a third of the
    time.
    """
    from ..crater_mapping.models import CraterMatch

    if not candidates or not references:
        return []

    taken: set[str] = set()
    matches = []
    for candidate in candidates:
        predicted_x = estimated_x_m + candidate.offset_dx_m
        predicted_y = estimated_y_m + candidate.offset_dy_m

        best, best_distance = None, tolerance_m
        for reference in references:
            if reference.id in taken:
                continue
            if reference.diameter_m > 0 and candidate.diameter_m > 0:
                size_error = abs(candidate.diameter_m - reference.diameter_m) / reference.diameter_m
                if size_error > diameter_tolerance:
                    continue
            distance = float(np.hypot(reference.x_m - predicted_x, reference.y_m - predicted_y))
            if distance < best_distance:
                best, best_distance = reference, distance

        if best is not None:
            taken.add(best.id)
            matches.append(
                CraterMatch(
                    candidate_id=candidate.id,
                    reference_id=best.id,
                    residual_m=round(best_distance, 3),
                )
            )
    return matches


def run_real_pipeline(
    pose: RoverPose,
    camera: NadirCamera,
    basemap,
    reference_craters: list[ReferenceCrater],
    detector,
    localizer,
) -> PerceptionResult:
    """
    The same pipeline over real Chandrayaan-2 imagery.

        real observation -> quadrants -> illumination points -> craters
                         -> register against basemap -> position

    Two stages differ from the synthetic path, both because the synthetic
    versions do not survive contact with this data:

    * craters come from shadow/highlight pairing rather than from fitting
      circles to bright rims, because at this sun angle there are no closed
      bright rims to fit;
    * the position comes from registering the observation against the basemap
      rather than from matching a crater constellation, because real crater
      detections repeat between two views only about a third of the time,
      which is too few consistent correspondences to fix a position.

    The pose is ground truth in the strict sense: the observation was cut from
    the basemap at exactly that pose, so the reported error is measured, not
    estimated. What it measures is this pipeline against the basemap, which is
    not the same as accuracy against the Moon - that would need the camera
    geometry this project does not carry.
    """
    stages: list[PipelineStage] = []

    with _Timer() as t:
        observation = render_real_observation(pose, camera, basemap)
    image = observation.image
    stages.append(
        PipelineStage(
            key="observation",
            label="Image received",
            status="ok" if observation.valid_fraction > 0.5 else "failed",
            detail=observation.detail(),
            duration_ms=round(t.ms, 2),
        )
    )

    with _Timer() as t:
        quadrants = split_quadrants(image)
    stages.append(
        PipelineStage(
            key="quadrants",
            label="Quadrants generated",
            status="ok",
            detail=f"{len(quadrants)} quadrants, {quadrants[0].width}x{quadrants[0].height} px each",
            duration_ms=round(t.ms, 2),
        )
    )

    with _Timer() as t:
        points = detect_all(detector, quadrants)
    per_quadrant = {q.name: sum(1 for p in points if p.quadrant == q.name) for q in quadrants}
    stages.append(
        PipelineStage(
            key="points",
            label="Shadow and highlight outlines detected",
            status="ok" if points else "failed",
            detail=" ".join(f"{k}:{v}" for k, v in per_quadrant.items()) or "no points found",
            duration_ms=round(t.ms, 2),
        )
    )

    with _Timer() as t:
        found = detector.detect_craters(image, observation.sun_direction_deg)
        candidates = craters_to_candidates(found, camera, pose.heading_deg, camera.gsd_m)
    stages.append(
        PipelineStage(
            key="aggregation",
            label="Craters paired from shadow and highlight",
            status="ok" if candidates else "failed",
            detail=(
                f"{len(candidates)} craters, sun bearing "
                f"{observation.sun_direction_deg:.0f} deg in frame"
            ),
            duration_ms=round(t.ms, 2),
        )
    )

    in_range = craters_in_radius(reference_craters, pose.x_m, pose.y_m, camera.max_range_m)

    with _Timer() as t:
        fix = localizer.locate(
            image, camera, basemap, pose.heading_deg, observation.valid_mask
        )
    stages.append(
        PipelineStage(
            key="matching",
            label="Observation registered to basemap",
            status="ok" if fix.succeeded else "failed",
            detail=(
                f"{fix.n_inliers}/{fix.n_matches} inlier features, "
                f"reprojection rmse {fix.reprojection_rmse_px} px"
                if fix.succeeded
                else (fix.rejected_reason or "registration failed")
            ),
            duration_ms=round(t.ms, 2),
        )
    )

    error_m = None
    if fix.succeeded:
        error_m = float(np.hypot(fix.estimated_x_m - pose.x_m, fix.estimated_y_m - pose.y_m))

    matches = (
        associate_with_reference(
            candidates, reference_craters, fix.estimated_x_m, fix.estimated_y_m
        )
        if fix.succeeded
        else []
    )

    stages.append(
        PipelineStage(
            key="localization",
            label="Rover position estimated",
            status="ok" if fix.succeeded else "failed",
            detail=(
                f"error {error_m:.2f} m against the pose the observation was cut from"
                if error_m is not None
                else "no position fix"
            ),
            duration_ms=0.0,
        )
    )

    if not fix.succeeded:
        note = fix.rejected_reason or "Registration failed; no position fix."
    else:
        note = (
            "Real Chandrayaan-2 OHRC imagery. Position recovered by registering the "
            "observation against the sector basemap; the error is measured against the "
            "pose the observation was cut from, which is exact. It scores this pipeline "
            "against the basemap, not against the Moon."
        )

    localization = LocalizationResult(
        estimated_x_m=fix.estimated_x_m,
        estimated_y_m=fix.estimated_y_m,
        ground_truth_x_m=pose.x_m,
        ground_truth_y_m=pose.y_m,
        position_error_m=error_m,
        heading_deg=pose.heading_deg,
        matches=matches,
        n_candidates=len(candidates),
        n_reference_in_range=len(in_range),
        match_confidence=round(fix.inlier_ratio, 3) if fix.succeeded else None,
        method=localizer.name,
        is_synthetic=False,
        note=note,
    )

    geodetic: dict = {}
    truth_lonlat = basemap.lonlat_at(pose.x_m, pose.y_m)
    if truth_lonlat:
        geodetic["ground_truth"] = {
            "longitude": round(truth_lonlat[0], 6),
            "latitude": round(truth_lonlat[1], 6),
        }
    if fix.succeeded:
        estimated_lonlat = basemap.lonlat_at(fix.estimated_x_m, fix.estimated_y_m)
        if estimated_lonlat:
            geodetic["estimated"] = {
                "longitude": round(estimated_lonlat[0], 6),
                "latitude": round(estimated_lonlat[1], 6),
            }
    if geodetic:
        geodetic["note"] = (
            "Interpolated from the bundle's NAV geometry grid, which is sampled "
            "lat/lon rather than a camera model."
        )

    return PerceptionResult(
        image=image,
        quadrants=quadrants,
        points=points,
        candidates=candidates,
        localization=localization,
        visible_truth=[],
        stages=stages,
        detector_name=detector.name,
        detector_is_synthetic=detector.is_synthetic,
        basemap=basemap.describe(),
        correspondences=fix.correspondences,
        geodetic=geodetic,
    )
