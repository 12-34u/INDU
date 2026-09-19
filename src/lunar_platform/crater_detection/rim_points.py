"""
Classical crater-rim point detector.

The algorithm itself is real image processing - morphological top-hat to
isolate bright rim structure, threshold, contour extraction, then arc-length
sampling along each contour. It has only ever been exercised against the
synthetic observations in observation.synthetic_view, so it is reported as
is_synthetic=True: the code is genuine, the validation is not.

Replacing it with a trained crater detector means implementing
crater_detection.base.PointDetector and nothing else.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..crater_mapping.models import DetectedPoint
from ..observation.quadrants import QuadrantView


class RimPointDetector:
    name = "classical-rim-tophat"
    is_synthetic = True

    def __init__(
        self,
        tophat_radius_px: int = 15,
        threshold_percentile: float = 92.0,
        min_response: float = 28.0,
        min_contour_points: int = 8,
        sample_spacing_px: float = 7.0,
        max_points_per_quadrant: int = 220,
    ):
        self.tophat_radius_px = tophat_radius_px
        self.threshold_percentile = threshold_percentile
        # A percentile alone always passes a fixed share of pixels, so a
        # quadrant of bare regolith would return a full quota of points drawn
        # from grain. The absolute floor is what makes an empty quadrant empty.
        self.min_response = min_response
        self.min_contour_points = min_contour_points
        self.sample_spacing_px = sample_spacing_px
        self.max_points_per_quadrant = max_points_per_quadrant

    def detect(self, quadrant: QuadrantView) -> list[DetectedPoint]:
        img = quadrant.image
        if img.size == 0:
            return []

        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Top-hat keeps structures brighter than their surroundings and smaller
        # than the kernel, which is what a crater rim is against regolith.
        ksize = self.tophat_radius_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

        if float(tophat.max()) <= 0:
            return []

        cutoff = max(
            float(np.percentile(tophat, self.threshold_percentile)), self.min_response
        )
        if float(tophat.max()) < cutoff:
            return []
        mask = (tophat >= cutoff).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        points: list[DetectedPoint] = []
        for contour in contours:
            pts = contour.reshape(-1, 2)
            if len(pts) < self.min_contour_points:
                continue

            # Walk the contour and drop a point every sample_spacing_px.
            step = max(1, int(round(self.sample_spacing_px)))
            for idx in range(0, len(pts), step):
                u_local, v_local = float(pts[idx][0]), float(pts[idx][1])
                strength = float(tophat[int(v_local), int(u_local)])
                if strength < cutoff:
                    continue
                u_full, v_full = quadrant.to_full(u_local, v_local)
                points.append(
                    DetectedPoint(
                        u=u_full,
                        v=v_full,
                        strength=round(strength, 2),
                        quadrant=quadrant.name,
                    )
                )

        if len(points) > self.max_points_per_quadrant:
            points.sort(key=lambda p: p.strength, reverse=True)
            points = points[: self.max_points_per_quadrant]

        return points
