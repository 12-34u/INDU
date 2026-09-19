"""
Rover observation cut from real OHRC imagery.

REAL: unlike observation.synthetic_view, nothing here is drawn. The rover's
view is a single resampling of the sector basemap at the requested pose, so
every crater, boulder and shadow in it is the mission's own pixels.

The pose used to cut the tile is exact, which is what makes this a simulation
worth scoring: the position localisation recovers can be compared against the
pose the observation was cut from, and the comparison is a real measurement of
the pipeline rather than a replay of annotations.

Two honesty constraints are enforced rather than glossed:

* one resampling only. Cutting and then straightening the tile would blur it
  twice and quietly degrade everything downstream.
* the frame is not padded. Where the view falls off the edge of the imagery
  the pixels are marked invalid and reported, because reflected or tiled
  regolith is fabricated terrain that a detector would happily find craters in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..simulation.camera import NadirCamera, RoverPose
from .basemap import SectorBasemap


@dataclass
class RealObservation:
    """A real observation plus the provenance needed to judge it."""

    image: np.ndarray
    valid_mask: np.ndarray
    valid_fraction: float
    resample_factor: float
    sun_direction_deg: float
    gsd_m: float
    source_gsd_m: float

    @property
    def is_upsampled(self) -> bool:
        return self.resample_factor > 1.05

    def detail(self) -> str:
        scale = (
            f"upsampled {self.resample_factor:.1f}x from {self.source_gsd_m:.2f} m/px"
            if self.is_upsampled
            else f"resampled from {self.source_gsd_m:.2f} m/px"
        )
        coverage = (
            "" if self.valid_fraction > 0.999 else f", {self.valid_fraction * 100:.0f}% within imagery"
        )
        return f"{self.image.shape[1]}x{self.image.shape[0]} px at {self.gsd_m} m/px, real OHRC ({scale}{coverage})"


def basemap_to_image_transform(
    pose: RoverPose, camera: NadirCamera, basemap: SectorBasemap
) -> np.ndarray:
    """
    The 2x3 affine taking basemap pixels to observation pixels.

    Derived from simulation.camera's convention so the two cannot drift apart:
    forward is up the frame, right is across it, and the rover sits at the
    centre. Composing scale and rotation into one matrix is what keeps the cut
    to a single resampling.
    """
    theta = math.radians(pose.heading_deg)
    ratio = basemap.gsd_m / camera.gsd_m
    sin_t, cos_t = math.sin(theta), math.cos(theta)

    return np.array(
        [
            [
                -ratio * sin_t,
                ratio * cos_t,
                camera.width_px / 2.0 + (pose.x_m * sin_t - pose.y_m * cos_t) / camera.gsd_m,
            ],
            [
                -ratio * cos_t,
                -ratio * sin_t,
                camera.height_px / 2.0 + (pose.x_m * cos_t + pose.y_m * sin_t) / camera.gsd_m,
            ],
        ],
        dtype=np.float64,
    )


def render_real_observation(
    pose: RoverPose, camera: NadirCamera, basemap: SectorBasemap
) -> RealObservation:
    """Cut the rover's view out of the real basemap at the given pose."""
    transform = basemap_to_image_transform(pose, camera, basemap)
    size = (camera.width_px, camera.height_px)

    # Shrinking needs area averaging or the fine structure aliases into
    # false rim-like texture; magnifying needs interpolation.
    resample_factor = basemap.gsd_m / camera.gsd_m
    interpolation = cv2.INTER_AREA if resample_factor < 1.0 else cv2.INTER_LINEAR

    image = cv2.warpAffine(
        basemap.image,
        transform,
        size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Warp a solid field the same way to find which pixels came from real data.
    valid = cv2.warpAffine(
        np.full(basemap.image.shape[:2], 255, np.uint8),
        transform,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_mask = valid > 0

    return RealObservation(
        image=image,
        valid_mask=valid_mask,
        valid_fraction=float(valid_mask.mean()),
        resample_factor=float(resample_factor),
        # The rover's heading turns the scene under the camera, so the sun sits
        # at a different bearing in the frame than it does in the basemap.
        sun_direction_deg=float((basemap.sun_direction_deg - pose.heading_deg) % 360.0),
        gsd_m=camera.gsd_m,
        source_gsd_m=basemap.gsd_m,
    )


def native_camera(basemap: SectorBasemap, size_px: int = 256) -> NadirCamera:
    """
    A camera sampling at the basemap's own resolution.

    Asking for a finer grid than the browse product carries would magnify
    rather than reveal: the detail is not in the data.
    """
    return NadirCamera(width_px=size_px, height_px=size_px, gsd_m=round(basemap.gsd_m, 4))
