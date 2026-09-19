"""
Crater detection for real low-sun lunar imagery.

crater_detection.rim_points looks for a bright annulus, which is what a crater
looks like in observation.synthetic_view. It is not what a crater looks like in
this OHRC strip. The sun sits low over the southern highlands here, so a crater
reads as a dark bowl with a lit crescent on the sunward side and no closed rim
at all - run the top-hat detector on it and every quadrant returns its cap of
points drawn from regolith texture, and no circle fits.

This detector uses the illumination instead of fighting it. A crater is the one
structure on this surface that reliably produces a shadow and a highlight of
similar size, adjacent, separated along the sun bearing. Finding those pairs is
both a better model of the physics and far more selective than thresholding
brightness.

    illumination residual -> shadow and highlight blobs -> pair along the sun
    bearing -> crater centre midway between them

The sun bearing is a parameter rather than a constant because it rotates in the
frame as the rover turns; observation.real_view computes it per observation.

Scope, stated plainly: these are detections, not a catalog. They locate real
craters in real imagery and are good enough to drive a hazard map, but they
have not been scored against a reference crater database, and their diameters
come from shadow geometry rather than from a fitted rim.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..crater_mapping.models import DetectedPoint, Quadrant
from ..observation.quadrants import QuadrantView

# Scale of the illumination trend removed before thresholding. Large enough to
# leave craters intact, small enough to flatten the strip's along-track
# brightness drift.
BACKGROUND_SIGMA_PX = 21.0


@dataclass
class ShadowPairCrater:
    """A crater located from its shadow/highlight pair, in image pixels."""

    center_u: float
    center_v: float
    radius_px: float
    score: float
    shadow_u: float
    shadow_v: float
    highlight_u: float
    highlight_v: float
    n_points: int


def _blobs(mask: np.ndarray, min_area: int, max_area: int) -> list[tuple[float, float, float, int]]:
    """Connected components as (centroid_u, centroid_v, equivalent_radius, area)."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            out.append(
                (float(centroids[i][0]), float(centroids[i][1]), float(np.sqrt(area / np.pi)), area)
            )
    return out


class ShadowPairDetector:
    """
    Pairs shadows with highlights along the sun bearing.

    Defaults were chosen by measuring how often a detection reappears at the
    same ground point in a second, rotated view of the same terrain, since a
    landmark that does not repeat is useless downstream.
    """

    name = "shadow-highlight-pairing"
    is_synthetic = False

    def __init__(
        self,
        sun_direction_deg: float = 340.0,
        pre_blur_px: float = 1.2,
        threshold_sigma: float = 1.0,
        min_blob_area_px: int = 25,
        max_blob_area_px: int = 6000,
        bearing_tolerance_deg: float = 40.0,
        max_size_ratio: float = 3.0,
        max_craters: int = 400,
        contour_sample_spacing_px: float = 5.0,
    ):
        self.sun_direction_deg = sun_direction_deg
        self.pre_blur_px = pre_blur_px
        self.threshold_sigma = threshold_sigma
        self.min_blob_area_px = min_blob_area_px
        self.max_blob_area_px = max_blob_area_px
        self.bearing_tolerance_deg = bearing_tolerance_deg
        self.max_size_ratio = max_size_ratio
        self.max_craters = max_craters
        self.contour_sample_spacing_px = contour_sample_spacing_px

    # -- masks ---------------------------------------------------------------

    def illumination_masks(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shadow and highlight masks from the local illumination residual."""
        data = image.astype(np.float32)
        if self.pre_blur_px > 0:
            data = cv2.GaussianBlur(data, (0, 0), self.pre_blur_px)

        residual = data - cv2.GaussianBlur(data, (0, 0), BACKGROUND_SIGMA_PX)
        spread = float(residual.std())
        if spread <= 1e-6:
            empty = np.zeros(image.shape[:2], np.uint8)
            return empty, empty

        cutoff = self.threshold_sigma * spread
        kernel = np.ones((3, 3), np.uint8)

        def clean(mask: np.ndarray) -> np.ndarray:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        shadow = clean(((residual < -cutoff) * 255).astype(np.uint8))
        highlight = clean(((residual > cutoff) * 255).astype(np.uint8))
        return shadow, highlight

    # -- craters -------------------------------------------------------------

    def detect_craters(
        self, image: np.ndarray, sun_direction_deg: float | None = None
    ) -> list[ShadowPairCrater]:
        """
        Find craters in a full frame.

        Pairing needs the whole frame: a crater's shadow and its highlight
        routinely fall in different quadrants, so this deliberately does not
        work quadrant by quadrant.
        """
        if image.size == 0:
            return []

        bearing = self.sun_direction_deg if sun_direction_deg is None else sun_direction_deg
        shadow_mask, highlight_mask = self.illumination_masks(image)
        shadows = _blobs(shadow_mask, self.min_blob_area_px, self.max_blob_area_px)
        highlights = _blobs(highlight_mask, self.min_blob_area_px, self.max_blob_area_px)
        if not shadows or not highlights:
            return []

        theta = np.radians(bearing)
        sun_vector = np.array([np.cos(theta), np.sin(theta)])
        min_cosine = np.cos(np.radians(self.bearing_tolerance_deg))

        found: list[ShadowPairCrater] = []
        for su, sv, sr, s_area in shadows:
            best = None
            for hu, hv, hr, h_area in highlights:
                delta = np.array([hu - su, hv - sv])
                separation = float(np.linalg.norm(delta))
                # Too close and the pair is one noisy blob split in two; too
                # far and they belong to different craters.
                if separation < 2.0 or separation > 8.0 * max(sr, 2.0):
                    continue
                if float(delta @ sun_vector) / separation < min_cosine:
                    continue
                ratio = hr / max(sr, 1e-6)
                if not (1.0 / self.max_size_ratio <= ratio <= self.max_size_ratio):
                    continue

                alignment = float(delta @ sun_vector) / separation
                symmetry = min(sr, hr) / max(sr, hr)
                # Bigger pairs are more reliable landmarks, so size is part of
                # the score rather than a separate filter.
                score = alignment * symmetry * min(sr, hr)
                if best is None or score > best[0]:
                    best = (score, hu, hv, hr, separation, s_area + h_area)

            if best is None:
                continue
            score, hu, hv, hr, separation, area = best
            found.append(
                ShadowPairCrater(
                    center_u=float((su + hu) / 2.0),
                    center_v=float((sv + hv) / 2.0),
                    radius_px=float(max(separation / 2.0, (sr + hr) / 2.0)),
                    score=float(score),
                    shadow_u=su, shadow_v=sv,
                    highlight_u=float(hu), highlight_v=float(hv),
                    n_points=int(area),
                )
            )

        found.sort(key=lambda c: -c.score)
        return self._suppress_overlaps(found)[: self.max_craters]

    @staticmethod
    def _suppress_overlaps(
        craters: list[ShadowPairCrater], overlap: float = 0.7
    ) -> list[ShadowPairCrater]:
        """Keep the strongest of any group of near-concentric detections."""
        kept: list[ShadowPairCrater] = []
        for crater in craters:
            if all(
                np.hypot(crater.center_u - k.center_u, crater.center_v - k.center_v)
                > overlap * max(crater.radius_px, k.radius_px)
                for k in kept
            ):
                kept.append(crater)
        return kept

    # -- PointDetector protocol ----------------------------------------------

    def detect(self, quadrant: QuadrantView) -> list[DetectedPoint]:
        """
        Sample the shadow and highlight outlines in this quadrant.

        This satisfies crater_detection.base.PointDetector so the per-quadrant
        stage stays meaningful on real imagery. These are the boundaries the
        crater pairing is built from, reported where they were found; the
        craters themselves come from detect_craters over the whole frame.
        """
        image = quadrant.image
        if image.size == 0:
            return []

        shadow_mask, highlight_mask = self.illumination_masks(image)
        step = max(1, int(round(self.contour_sample_spacing_px)))
        points: list[DetectedPoint] = []
        name: Quadrant = quadrant.name

        for mask, strength in ((shadow_mask, -1.0), (highlight_mask, 1.0)):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                pts = contour.reshape(-1, 2)
                if cv2.contourArea(contour) < self.min_blob_area_px:
                    continue
                for index in range(0, len(pts), step):
                    u_full, v_full = quadrant.to_full(float(pts[index][0]), float(pts[index][1]))
                    points.append(
                        DetectedPoint(
                            u=u_full,
                            v=v_full,
                            # Sign carries which structure the point outlines,
                            # so the UI can tell shadow from highlight.
                            strength=round(strength * float(len(pts)), 2),
                            quadrant=name,
                        )
                    )
        return points
