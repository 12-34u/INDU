"""
Point aggregation: detected rim points -> crater candidates.

Craters overlap in a real frame, so rim points cannot simply be clustered by
proximity - one connected blob of points routinely spans two or three rims, and
a single circle fitted through it describes none of them. Instead this extracts
circles one at a time with RANSAC and removes each circle's inliers before
looking for the next, which separates overlapping rims and ignores the spurious
points the detector inevitably returns.

Confidence comes from two measured properties of the accepted fit: how tightly
its inliers sit on the circle, and how much of the circumference they span. A
candidate built from a short arc scores low. It is not a detection probability.
"""

from __future__ import annotations

import numpy as np

from ..simulation.camera import NadirCamera, RoverPose
from .models import CraterCandidate, DetectedPoint, Quadrant

AGGREGATION_SEED = 11


def _circumcircle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[float, float, float] | None:
    """Exact circle through three points, or None if they are near-collinear."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None

    a_sq = ax * ax + ay * ay
    b_sq = bx * bx + by * by
    c_sq = cx * cx + cy * cy

    ux = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / d
    uy = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / d
    return float(ux), float(uy), float(np.hypot(ax - ux, ay - uy))


def _refine_circle(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic least-squares circle fit (Kasa). Returns (cu, cv, radius, rmse)."""
    A = np.column_stack([2.0 * u, 2.0 * v, np.ones_like(u)])
    d = u**2 + v**2
    solution, *_ = np.linalg.lstsq(A, d, rcond=None)
    a, b, c = solution

    radius_sq = c + a**2 + b**2
    if radius_sq <= 0:
        return float(a), float(b), 0.0, float("inf")

    radius = float(np.sqrt(radius_sq))
    residuals = np.hypot(u - a, v - b) - radius
    return float(a), float(b), radius, float(np.sqrt(np.mean(residuals**2)))


def _angular_coverage(u: np.ndarray, v: np.ndarray, cu: float, cv: float, bins: int = 16) -> float:
    angles = np.arctan2(v - cv, u - cu)
    idx = ((angles + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    return float(len(np.unique(idx)) / bins)


def aggregate_points(
    points: list[DetectedPoint],
    camera: NadirCamera,
    heading_deg: float,
    inlier_tolerance_px: float = 3.0,
    min_points: int = 26,
    min_radius_px: float = 14.0,
    # Bounded by the largest crater the reference population contains; without
    # this, RANSAC happily fits huge circles through rim points that belong to
    # several different craters.
    max_radius_px: float = 75.0,
    max_fit_rmse_px: float = 2.2,
    min_coverage: float = 0.6,
    iterations: int = 900,
    max_craters: int = 14,
    neighbour_radius_px: float = 320.0,
    seed: int = AGGREGATION_SEED,
) -> list[CraterCandidate]:
    """Extract crater candidates from rim points by sequential RANSAC circle fitting."""
    if len(points) < min_points:
        return []

    coords = np.array([[p.u, p.v] for p in points], dtype=np.float64)
    quadrants = [p.quadrant for p in points]
    remaining = np.arange(len(points))

    rng = np.random.default_rng(seed)
    pose = RoverPose(0.0, 0.0, heading_deg)
    candidates: list[CraterCandidate] = []

    while len(remaining) >= min_points and len(candidates) < max_craters:
        pts = coords[remaining]
        best_inliers: np.ndarray | None = None
        best_count = 0

        for _ in range(iterations):
            anchor = int(rng.integers(len(pts)))
            # Sample the other two points near the anchor: three points drawn
            # from across the frame almost never share a rim.
            near = np.flatnonzero(
                np.hypot(pts[:, 0] - pts[anchor, 0], pts[:, 1] - pts[anchor, 1]) <= neighbour_radius_px
            )
            if len(near) < 3:
                continue

            pick = rng.choice(near, size=2, replace=False)
            circle = _circumcircle(pts[anchor], pts[pick[0]], pts[pick[1]])
            if circle is None:
                continue

            cu, cv, radius = circle
            if not (min_radius_px <= radius <= max_radius_px):
                continue

            distances = np.hypot(pts[:, 0] - cu, pts[:, 1] - cv)
            inliers = np.flatnonzero(np.abs(distances - radius) <= inlier_tolerance_px)
            if len(inliers) > best_count:
                best_count = len(inliers)
                best_inliers = inliers

        if best_inliers is None or best_count < min_points:
            break

        # Refit on the consensus set, then re-select inliers for the refined circle.
        cu, cv, radius, _ = _refine_circle(pts[best_inliers, 0], pts[best_inliers, 1])
        if not (min_radius_px <= radius <= max_radius_px):
            remaining = np.delete(remaining, best_inliers)
            continue

        distances = np.hypot(pts[:, 0] - cu, pts[:, 1] - cv)
        final_inliers = np.flatnonzero(np.abs(distances - radius) <= inlier_tolerance_px)
        if len(final_inliers) < min_points:
            remaining = np.delete(remaining, best_inliers)
            continue

        u_in = pts[final_inliers, 0]
        v_in = pts[final_inliers, 1]
        cu, cv, radius, rmse = _refine_circle(u_in, v_in)
        coverage = _angular_coverage(u_in, v_in, cu, cv)

        accepted = (
            min_radius_px <= radius <= max_radius_px
            and rmse <= max_fit_rmse_px
            and coverage >= min_coverage
        )

        if accepted:
            fit_quality = 1.0 / (1.0 + rmse / 2.0)
            confidence = float(np.clip(coverage * fit_quality, 0.0, 1.0))
            source: list[Quadrant] = sorted({quadrants[remaining[i]] for i in final_inliers})
            dx, dy = camera.image_to_offset(pose, cu, cv)

            candidates.append(
                CraterCandidate(
                    id=f"C{len(candidates) + 1:02d}",
                    center_u=round(float(cu), 2),
                    center_v=round(float(cv), 2),
                    radius_px=round(radius, 2),
                    diameter_px=round(radius * 2.0, 2),
                    n_points=int(len(final_inliers)),
                    fit_rmse_px=round(rmse, 3),
                    confidence=round(confidence, 3),
                    source_quadrants=source,
                    offset_dx_m=round(float(dx), 3),
                    offset_dy_m=round(float(dy), 3),
                    diameter_m=round(float(radius * 2.0 * camera.gsd_m), 2),
                )
            )

        remaining = np.delete(remaining, final_inliers)

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    candidates = _suppress_duplicates(candidates)
    for i, candidate in enumerate(candidates, start=1):
        candidate.id = f"C{i:02d}"
    return candidates


def _suppress_duplicates(
    candidates: list[CraterCandidate],
    centre_tolerance: float = 0.25,
    radius_tolerance: float = 0.35,
) -> list[CraterCandidate]:
    """
    Drop concentric duplicates, keeping the highest-confidence fit.

    A crater rim has width, so its inner and outer edges can each satisfy the
    circle fit and be emitted as separate candidates. Tolerances are relative to
    radius because that spacing scales with crater size.
    """
    kept: list[CraterCandidate] = []
    for candidate in candidates:
        duplicate = False
        for existing in kept:
            centre_gap = np.hypot(
                candidate.center_u - existing.center_u, candidate.center_v - existing.center_v
            )
            radius_gap = abs(candidate.radius_px - existing.radius_px)
            if (
                centre_gap <= centre_tolerance * existing.radius_px
                and radius_gap <= radius_tolerance * existing.radius_px
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept
