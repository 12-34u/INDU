"""
Sub-pixel refinement of verified correspondences.

SIFT keypoints already carry sub-pixel centres, but the reference-side location
is only as good as the match. This refines each inlier's reference position by
local normalised cross-correlation around the point predicted by the current
transform, then fits a quadratic to the correlation peak.

Refinement is applied only to RANSAC inliers. Running it on outliers would pull
them onto whatever texture happens to be nearby and manufacture agreement.
"""

from __future__ import annotations

import cv2
import numpy as np


def _quadratic_peak_offset(response: np.ndarray, peak: tuple[int, int]) -> tuple[float, float]:
    """Fit a parabola through the correlation peak and its neighbours."""
    y, x = peak
    if not (0 < y < response.shape[0] - 1 and 0 < x < response.shape[1] - 1):
        return 0.0, 0.0

    dx_denominator = response[y, x - 1] - 2 * response[y, x] + response[y, x + 1]
    dy_denominator = response[y - 1, x] - 2 * response[y, x] + response[y + 1, x]

    dx = (
        0.5 * (response[y, x - 1] - response[y, x + 1]) / dx_denominator
        if abs(dx_denominator) > 1e-9
        else 0.0
    )
    dy = (
        0.5 * (response[y - 1, x] - response[y + 1, x]) / dy_denominator
        if abs(dy_denominator) > 1e-9
        else 0.0
    )

    # A parabola fit that lands outside the centre cell means the peak was not
    # well formed; ignore it rather than trust an extrapolation.
    if abs(dx) > 1.0 or abs(dy) > 1.0:
        return 0.0, 0.0
    return float(dx), float(dy)


def refine_matches(
    source: np.ndarray,
    reference: np.ndarray,
    pts_source: np.ndarray,
    pts_reference: np.ndarray,
    patch_radius: int = 12,
    search_radius: int = 4,
    min_correlation: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns refined reference points and the correlation score for each.

    A point whose peak correlation falls below min_correlation keeps its
    original position and is reported with its measured score, so the caller
    can see how many refinements actually succeeded.
    """
    refined = pts_reference.copy().astype(np.float64)
    scores = np.zeros(len(pts_reference), dtype=np.float64)

    h_src, w_src = source.shape[:2]
    h_ref, w_ref = reference.shape[:2]

    for i, ((sx, sy), (rx, ry)) in enumerate(zip(pts_source, pts_reference)):
        sx_i, sy_i, rx_i, ry_i = int(round(sx)), int(round(sy)), int(round(rx)), int(round(ry))

        if not (
            patch_radius <= sx_i < w_src - patch_radius
            and patch_radius <= sy_i < h_src - patch_radius
        ):
            continue

        margin = patch_radius + search_radius
        if not (
            margin <= rx_i < w_ref - margin and margin <= ry_i < h_ref - margin
        ):
            continue

        template = source[
            sy_i - patch_radius : sy_i + patch_radius + 1,
            sx_i - patch_radius : sx_i + patch_radius + 1,
        ].astype(np.float32)
        window = reference[
            ry_i - margin : ry_i + margin + 1,
            rx_i - margin : rx_i + margin + 1,
        ].astype(np.float32)

        if template.std() < 1e-3 or window.std() < 1e-3:
            continue

        response = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        scores[i] = float(max_val)

        if max_val < min_correlation:
            continue

        peak_x, peak_y = max_loc
        sub_dx, sub_dy = _quadratic_peak_offset(response, (peak_y, peak_x))

        refined[i, 0] = rx_i + (peak_x - search_radius) + sub_dx
        refined[i, 1] = ry_i + (peak_y - search_radius) + sub_dy

    return refined, scores
