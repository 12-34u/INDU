"""
Localisation by registering a real observation against the sector basemap.

Crater-constellation matching (localization.matching) recovers a position from
a handful of landmarks. That works on the synthetic observations it was built
for, where every crater is a clean bright annulus. It does not survive real
OHRC imagery: at this sun angle a crater is a dark bowl with a lit crescent,
its apparent centre moves with illumination, and the detections repeat between
two views of the same ground only about a third of the time - too few
consistent correspondences to fix a position.

Registering the images directly does survive it, because it uses all the
texture in the frame rather than a dozen fitted circles. This is the same
approach as registration.pipeline, scoped to the observation-vs-basemap case
and kept fast enough to run per frame by caching the basemap's features.

The estimate is checked before it is returned. The transform's rotation must
agree with the rover's known heading and its scale with the known resolution
ratio; a fix that disagrees is a wrong answer dressed as a confident one, and
is rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..observation.basemap import SectorBasemap
from ..simulation.camera import NadirCamera


@dataclass
class ImageFix:
    """Outcome of one observation-to-basemap registration."""

    estimated_x_m: Optional[float]
    estimated_y_m: Optional[float]
    n_keypoints: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_rmse_px: Optional[float] = None
    recovered_heading_deg: Optional[float] = None
    heading_residual_deg: Optional[float] = None
    scale_residual: Optional[float] = None
    rejected_reason: Optional[str] = None
    correspondences: list[dict] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.estimated_x_m is not None


def _angle_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def heading_from_transform(matrix: np.ndarray) -> float:
    """
    Recover the rover heading implied by an observation-to-basemap transform.

    Inverts the forward model in observation.real_view: the basemap-to-image
    affine is a rotation by atan2(-cos h, -sin h) scaled by the resolution
    ratio, so the heading falls straight out of the estimated rotation.
    """
    psi = math.atan2(matrix[1, 0], matrix[0, 0])
    phi = -psi
    heading = math.degrees(math.atan2(-math.cos(phi), -math.sin(phi))) % 360.0
    # A heading a hair below zero comes back as 359.999... from the modulo,
    # which is the same bearing reported as the opposite end of the range.
    return 0.0 if heading > 360.0 - 1e-9 else heading


class BasemapLocalizer:
    """
    SIFT-based localiser against a cached basemap.

    The basemap's keypoints are computed once and reused: they are the
    expensive half, and the basemap does not change between observations.
    """

    name = "sift-basemap-registration"
    is_synthetic = False

    def __init__(
        self,
        n_features: int = 4000,
        ratio_test: float = 0.75,
        ransac_threshold_px: float = 3.0,
        min_inliers: int = 8,
        max_heading_residual_deg: float = 5.0,
        max_scale_residual: float = 0.10,
    ):
        self.n_features = n_features
        self.ratio_test = ratio_test
        self.ransac_threshold_px = ransac_threshold_px
        self.min_inliers = min_inliers
        self.max_heading_residual_deg = max_heading_residual_deg
        self.max_scale_residual = max_scale_residual

        self._sift = cv2.SIFT_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher()
        # Local contrast equalisation; the strip's brightness drifts along
        # track, and raw intensity would make features depend on where in the
        # strip the tile was cut from.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._basemap_cache: dict[tuple, tuple] = {}

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return self._clahe.apply(image)

    def basemap_features(self, basemap: SectorBasemap) -> tuple:
        key = (id(basemap), basemap.width_px, basemap.height_px, basemap.origin_scan)
        cached = self._basemap_cache.get(key)
        if cached is not None:
            return cached

        keypoints, descriptors = self._sift.detectAndCompute(
            self._prepare(basemap.image), None
        )
        # Bound the cache: a session switches between a handful of basemaps.
        if len(self._basemap_cache) > 4:
            self._basemap_cache.clear()
        self._basemap_cache[key] = (keypoints, descriptors)
        return keypoints, descriptors

    def locate(
        self,
        observation: np.ndarray,
        camera: NadirCamera,
        basemap: SectorBasemap,
        heading_deg: float,
        valid_mask: Optional[np.ndarray] = None,
        max_correspondences: int = 400,
    ) -> ImageFix:
        """Register one observation against the basemap and fix the rover."""
        base_kp, base_desc = self.basemap_features(basemap)
        if base_desc is None or len(base_kp) < self.min_inliers:
            return ImageFix(None, None, rejected_reason="Basemap has too few features.")

        mask = None
        if valid_mask is not None and not valid_mask.all():
            # Keypoints straddling the border sit on a fabricated edge, not on
            # terrain, and would match nothing real.
            mask = cv2.erode((valid_mask * 255).astype(np.uint8), np.ones((9, 9), np.uint8))

        obs_kp, obs_desc = self._sift.detectAndCompute(self._prepare(observation), mask)
        if obs_desc is None or len(obs_kp) < self.min_inliers:
            return ImageFix(
                None, None, n_keypoints=len(obs_kp or []),
                rejected_reason="Observation has too few features to register.",
            )

        pairs = self._matcher.knnMatch(obs_desc, base_desc, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < self.ratio_test * n.distance]
        if len(good) < self.min_inliers:
            return ImageFix(
                None, None, n_keypoints=len(obs_kp), n_matches=len(good),
                rejected_reason=f"Only {len(good)} distinctive matches; need {self.min_inliers}.",
            )

        src = np.float32([obs_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([base_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold_px
        )
        if matrix is None or inlier_mask is None:
            return ImageFix(
                None, None, n_keypoints=len(obs_kp), n_matches=len(good),
                rejected_reason="RANSAC found no consistent transform.",
            )

        inliers = inlier_mask.ravel().astype(bool)
        n_inliers = int(inliers.sum())
        inlier_ratio = float(n_inliers / max(len(good), 1))

        projected = (matrix @ np.hstack([src.reshape(-1, 2), np.ones((len(src), 1))]).T).T
        residuals = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)[inliers]
        rmse = float(np.sqrt(np.mean(residuals**2))) if n_inliers else None

        recovered_heading = heading_from_transform(matrix)
        heading_residual = _angle_difference_deg(recovered_heading, heading_deg)
        estimated_scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
        expected_scale = camera.gsd_m / basemap.gsd_m
        scale_residual = abs(estimated_scale - expected_scale) / max(expected_scale, 1e-9)

        fix = ImageFix(
            None, None,
            n_keypoints=len(obs_kp),
            n_matches=len(good),
            n_inliers=n_inliers,
            inlier_ratio=round(inlier_ratio, 4),
            reprojection_rmse_px=round(rmse, 3) if rmse is not None else None,
            recovered_heading_deg=round(recovered_heading, 2),
            heading_residual_deg=round(heading_residual, 2),
            scale_residual=round(scale_residual, 4),
        )

        if n_inliers < self.min_inliers:
            fix.rejected_reason = f"Only {n_inliers} inliers; need {self.min_inliers}."
            return fix
        if heading_residual > self.max_heading_residual_deg:
            fix.rejected_reason = (
                f"Transform implies heading {recovered_heading:.1f} deg against a known "
                f"{heading_deg:.1f} deg; the match is not geometrically consistent."
            )
            return fix
        if scale_residual > self.max_scale_residual:
            fix.rejected_reason = (
                f"Transform scale is off by {scale_residual * 100:.0f}%; the match does not "
                "respect the known resolution ratio."
            )
            return fix

        centre = matrix @ np.array([camera.width_px / 2.0, camera.height_px / 2.0, 1.0])
        fix.estimated_x_m, fix.estimated_y_m = basemap.pixel_to_sector(centre[0], centre[1])

        kept = np.flatnonzero(inliers)[:max_correspondences]
        fix.correspondences = [
            {
                "u": round(float(src[i, 0, 0]), 2),
                "v": round(float(src[i, 0, 1]), 2),
                "basemap_u": round(float(dst[i, 0, 0]), 2),
                "basemap_v": round(float(dst[i, 0, 1]), 2),
            }
            for i in kept
        ]
        return fix
