"""
Camera model relating sector coordinates to rover observation pixels.

The MVP models the rover observation as a nadir (straight-down) view centred on
the rover. This is deliberately the simplest invertible model that still makes
localisation a real geometric problem: a perspective/oblique model would need
camera intrinsics and attitude that the project does not yet carry.

Sector frame: x to the right, y downward, both in metres from the sector's
top-left corner. This matches the crater convention used by
hazards.hazard_map.calculate_crater_hazard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RoverPose:
    """Ground-truth or estimated rover pose in sector metres."""

    x_m: float
    y_m: float
    heading_deg: float = 0.0


@dataclass(frozen=True)
class NadirCamera:
    """
    A square nadir view of the surface.

    width_px/height_px is the rendered observation size; gsd_m is the ground
    distance one observation pixel covers, so the footprint is
    width_px * gsd_m by height_px * gsd_m metres.
    """

    width_px: int = 512
    height_px: int = 512
    gsd_m: float = 0.5

    @property
    def footprint_m(self) -> tuple[float, float]:
        return self.width_px * self.gsd_m, self.height_px * self.gsd_m

    @property
    def max_range_m(self) -> float:
        w, h = self.footprint_m
        return math.hypot(w, h) / 2.0

    def world_to_image(self, pose: RoverPose, x_m: float, y_m: float) -> tuple[float, float]:
        """Project a sector point into observation pixel coordinates."""
        theta = math.radians(pose.heading_deg)
        dx = x_m - pose.x_m
        dy = y_m - pose.y_m

        forward = dx * math.cos(theta) + dy * math.sin(theta)
        right = -dx * math.sin(theta) + dy * math.cos(theta)

        u = self.width_px / 2.0 + right / self.gsd_m
        v = self.height_px / 2.0 - forward / self.gsd_m
        return u, v

    def image_to_offset(self, pose: RoverPose, u: float, v: float) -> tuple[float, float]:
        """
        Convert an observation pixel back to a sector-frame offset in metres
        relative to the rover. Only the heading is used, so this is available
        before the rover position is known - which is what makes localisation
        from the observation possible.
        """
        theta = math.radians(pose.heading_deg)
        right = (u - self.width_px / 2.0) * self.gsd_m
        forward = (self.height_px / 2.0 - v) * self.gsd_m

        dx = forward * math.cos(theta) - right * math.sin(theta)
        dy = forward * math.sin(theta) + right * math.cos(theta)
        return dx, dy

    def is_inside(self, u: float, v: float, margin_px: float = 0.0) -> bool:
        return (
            margin_px <= u < self.width_px - margin_px
            and margin_px <= v < self.height_px - margin_px
        )
