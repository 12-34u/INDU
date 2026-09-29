"""
Reference crater map for the sector.

This is the catalog that localisation matches an observation against. It is
seeded from the sector's own crater catalog (the same file that drives the
hazard map, so the two never disagree) and topped up with deterministic
synthetic craters, because matching a constellation needs more landmarks than
the handful the demo catalog carries.

Every generated crater is marked provenance="synthetic". Nothing here is a
real lunar crater catalog.
"""

from __future__ import annotations

import numpy as np

from ..crater_mapping.models import ReferenceCrater

# Fixed so the reference map is identical across processes and reloads;
# localisation results must be reproducible between runs.
REFERENCE_SEED = 20240917

# Landmarks per square kilometre.
#
# This is a DENSITY rather than a count because the matcher's requirement is
# local: match_constellation needs min_inliers distinct candidates agreeing on
# one position, and a candidate only exists for a crater inside the current
# observation frame. What decides whether a fix is possible is therefore how
# many landmarks fall in one frame, not how many exist in the sector.
#
# A fixed count silently fails that test as the sector grows. 130 craters is
# 15 per frame over the demo's 1 km sector and every pose can be fixed; the
# same 130 over a 2.4 km sector is 2.5 per frame, below the floor of 4, and
# localisation then fails almost everywhere for a reason that looks like a
# matcher fault and is not one.
#
# 130/km² is the demo's density, which is the density the aggregation and
# matching thresholds were tuned against.
REFERENCE_DENSITY_PER_KM2 = 130.0


def build_reference_map(
    width_m: float,
    height_m: float,
    catalog_craters: list[dict] | None = None,
    n_synthetic: int | None = None,
    density_per_km2: float = REFERENCE_DENSITY_PER_KM2,
    # Kept small relative to the observation footprint: craters spanning half
    # the frame overlap so heavily that their rims cannot be separated.
    min_diameter_m: float = 20.0,
    max_diameter_m: float = 60.0,
    min_separation_m: float = 46.0,
    seed: int = REFERENCE_SEED,
) -> list[ReferenceCrater]:
    """
    Build the sector reference map.

    Catalog craters are kept verbatim; synthetic craters fill the rest of the
    sector under a minimum-separation constraint so the constellation matcher
    is not fed degenerate overlapping landmarks.

    How many synthetic craters that is follows from the sector's area and
    density_per_km2, so a larger sector gets proportionally more landmarks and
    the number visible in one observation stays roughly constant. Passing
    n_synthetic overrides the density with an explicit count.
    """
    if n_synthetic is None:
        n_synthetic = int(round(density_per_km2 * (width_m * height_m) / 1e6))

    craters: list[ReferenceCrater] = []
    placed: list[tuple[float, float]] = []

    for i, c in enumerate(catalog_craters or []):
        crater = ReferenceCrater(
            id=c.get("id") or f"CAT{i:03d}",
            x_m=float(c["x"]),
            y_m=float(c["y"]),
            diameter_m=float(c["diameter_m"]),
            provenance="catalog",
        )
        craters.append(crater)
        placed.append((crater.x_m, crater.y_m))

    rng = np.random.default_rng(seed)
    margin = max_diameter_m / 2.0
    attempts = 0
    max_attempts = n_synthetic * 80

    while len(craters) - len(catalog_craters or []) < n_synthetic and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(margin, width_m - margin))
        y = float(rng.uniform(margin, height_m - margin))

        if any((x - px) ** 2 + (y - py) ** 2 < min_separation_m**2 for px, py in placed):
            continue

        craters.append(
            ReferenceCrater(
                id=f"SYN{len(craters):03d}",
                x_m=x,
                y_m=y,
                diameter_m=float(rng.uniform(min_diameter_m, max_diameter_m)),
                provenance="synthetic",
            )
        )
        placed.append((x, y))

    return craters


def craters_in_radius(
    craters: list[ReferenceCrater], x_m: float, y_m: float, radius_m: float
) -> list[ReferenceCrater]:
    return [
        c for c in craters if (c.x_m - x_m) ** 2 + (c.y_m - y_m) ** 2 <= radius_m**2
    ]


def to_hazard_dicts(craters: list[ReferenceCrater]) -> list[dict]:
    """Convert to the dict shape calculate_crater_hazard expects."""
    return [
        {"id": c.id, "x": c.x_m, "y": c.y_m, "diameter_m": c.diameter_m} for c in craters
    ]


def build_reference_map_from_basemap(
    basemap,
    detector,
    min_diameter_m: float = 20.0,
    max_craters: int = 400,
) -> list[ReferenceCrater]:
    """
    Build the landmark map by detecting craters in the real sector basemap.

    Unlike build_reference_map above, nothing here is generated: every landmark
    is a crater found in Chandrayaan-2 imagery, positioned in sector metres and
    marked provenance="detected". It is still not a published crater catalog -
    these detections have not been scored against one - but it is measured from
    the data rather than invented to give the matcher something to chew on.

    Craters below min_diameter_m are dropped: near the browse product's
    resolution a detection is a few pixels across and does not survive being
    re-observed from another angle.
    """
    craters = detector.detect_craters(basemap.image, basemap.sun_direction_deg)

    out: list[ReferenceCrater] = []
    for crater in craters:
        x_m, y_m = basemap.pixel_to_sector(crater.center_u, crater.center_v)
        diameter_m = 2.0 * crater.radius_px * basemap.gsd_m
        if diameter_m < min_diameter_m:
            continue
        out.append(
            ReferenceCrater(
                id=f"D{len(out):03d}",
                x_m=round(float(x_m), 2),
                y_m=round(float(y_m), 2),
                diameter_m=round(float(diameter_m), 2),
                provenance="detected",
            )
        )
        if len(out) >= max_craters:
            break
    return out
