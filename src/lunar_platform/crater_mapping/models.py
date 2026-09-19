"""Data structures shared by the crater perception and localisation pipeline."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Quadrant = Literal["Q1", "Q2", "Q3", "Q4"]

# Every object carries where it came from, so the UI can state plainly which
# parts of a result are synthetic rather than implying measured performance.
Provenance = Literal["synthetic", "catalog", "detected"]


class ReferenceCrater(BaseModel):
    """A crater in the reference map, positioned in sector metres (y down)."""

    id: str
    x_m: float
    y_m: float
    diameter_m: float
    provenance: Provenance = "synthetic"


class DetectedPoint(BaseModel):
    """A single point returned by the point detector, in observation pixels."""

    u: float
    v: float
    strength: float
    quadrant: Quadrant


class CraterCandidate(BaseModel):
    """
    A crater aggregated from detected points.

    Image-space fields come straight from the fit. The offset fields are the
    candidate expressed as a sector-frame displacement from the rover, which is
    what the matcher consumes.
    """

    id: str
    center_u: float
    center_v: float
    radius_px: float
    diameter_px: float
    n_points: int
    fit_rmse_px: float
    confidence: float
    source_quadrants: list[Quadrant]
    offset_dx_m: float
    offset_dy_m: float
    diameter_m: float
    provenance: Provenance = "detected"


class CraterMatch(BaseModel):
    """A correspondence between an observed candidate and a reference crater."""

    candidate_id: str
    reference_id: str
    residual_m: float


class PipelineStage(BaseModel):
    """One step of the pipeline, with a real measured duration."""

    key: str
    label: str
    status: Literal["ok", "failed", "skipped"]
    detail: str
    duration_ms: float


class LocalizationResult(BaseModel):
    estimated_x_m: Optional[float]
    estimated_y_m: Optional[float]
    ground_truth_x_m: Optional[float]
    ground_truth_y_m: Optional[float]
    position_error_m: Optional[float]
    heading_deg: float
    matches: list[CraterMatch]
    n_candidates: int
    n_reference_in_range: int
    match_confidence: Optional[float]
    method: str
    is_synthetic: bool
    note: str
