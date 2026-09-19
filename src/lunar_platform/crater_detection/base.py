"""
Point detector interface.

The pipeline depends on this Protocol, never on a concrete detector, so the
current classical implementation can be swapped for a learned crater model
without touching the aggregation, matching or localisation stages.

A detector receives one quadrant at a time and returns points in FULL image
coordinates (use QuadrantView.to_full). Detecting per quadrant is the
processing strategy this project was handed; it also keeps thresholds local,
which helps when illumination varies across the frame.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..crater_mapping.models import DetectedPoint
from ..observation.quadrants import QuadrantView


@runtime_checkable
class PointDetector(Protocol):
    name: str
    is_synthetic: bool

    def detect(self, quadrant: QuadrantView) -> list[DetectedPoint]:
        """Return candidate crater-rim points found in this quadrant."""
        ...


def detect_all(detector: PointDetector, quadrants: list[QuadrantView]) -> list[DetectedPoint]:
    """Run a detector over every quadrant independently."""
    points: list[DetectedPoint] = []
    for quadrant in quadrants:
        points.extend(detector.detect(quadrant))
    return points
