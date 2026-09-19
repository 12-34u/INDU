"""
Synthetic rover observation.

SYNTHETIC: this renders what the rover would see given a known pose and the
reference crater map. It is a stand-in for real Chandrayaan-2 / rover imagery
and carries no radiometric or photometric fidelity.

Its purpose is to close the loop for an objective demo: because the pose used
to render is ground truth, the position that localisation recovers from the
rendered image can be scored against it. The detector downstream runs on this
raster like it would on a real one, so the detection and aggregation stages are
genuine image processing, not replayed annotations.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..crater_mapping.models import ReferenceCrater
from ..simulation.camera import NadirCamera, RoverPose

OBSERVATION_SEED = 7


def render_observation(
    pose: RoverPose,
    camera: NadirCamera,
    craters: list[ReferenceCrater],
    sun_azimuth_deg: float = 135.0,
    noise_sigma: float = 6.0,
    seed: int = OBSERVATION_SEED,
) -> tuple[np.ndarray, list[dict]]:
    """
    Render the observation and report which craters are actually in frame.

    Returns the 8-bit image and the ground-truth projection of every crater
    that landed inside it. The projections are returned for scoring and for
    the UI only; the detector never sees them.
    """
    h, w = camera.height_px, camera.width_px
    rng = np.random.default_rng(seed)

    # Regolith background: mid-grey with low-frequency shading plus grain.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (
        118.0
        + 9.0 * np.sin(xx / 61.0 + 0.6)
        + 7.0 * np.cos(yy / 47.0 - 0.3)
    )
    image = base + rng.normal(0.0, noise_sigma, size=(h, w)).astype(np.float32)

    sun = np.radians(sun_azimuth_deg)
    lit_offset = np.array([np.cos(sun), np.sin(sun)])

    visible: list[dict] = []
    for crater in craters:
        u, v = camera.world_to_image(pose, crater.x_m, crater.y_m)
        radius_px = (crater.diameter_m / 2.0) / camera.gsd_m

        # Keep craters whose rim intersects the frame at all.
        if not (-radius_px <= u < w + radius_px and -radius_px <= v < h + radius_px):
            continue
        if radius_px < 3.0:
            continue

        layer = np.zeros((h, w), dtype=np.float32)
        centre = (int(round(u)), int(round(v)))
        r = int(round(radius_px))

        # Bowl interior sits slightly darker than the surrounding regolith.
        cv2.circle(layer, centre, r, -16.0, thickness=-1, lineType=cv2.LINE_AA)
        # Rim: a bright annulus, the feature the detector keys on.
        rim_thickness = max(2, int(round(radius_px * 0.22)))
        cv2.circle(layer, centre, r, 52.0, thickness=rim_thickness, lineType=cv2.LINE_AA)
        image += layer

        # Directional shading: brighten the sun-facing rim, shadow the far side.
        shadow = np.zeros((h, w), dtype=np.float32)
        shadow_centre = (
            int(round(u - lit_offset[0] * radius_px * 0.35)),
            int(round(v - lit_offset[1] * radius_px * 0.35)),
        )
        cv2.circle(shadow, shadow_centre, max(1, int(r * 0.75)), -22.0, -1, cv2.LINE_AA)
        image += shadow

        if camera.is_inside(u, v):
            visible.append(
                {
                    "reference_id": crater.id,
                    "u": float(u),
                    "v": float(v),
                    "radius_px": float(radius_px),
                    "diameter_m": float(crater.diameter_m),
                    "x_m": float(crater.x_m),
                    "y_m": float(crater.y_m),
                }
            )

    return np.clip(image, 0, 255).astype(np.uint8), visible
