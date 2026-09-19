"""
Traversability derived from the navigation cost map.

This is a presentation layer over navigation.cost_map, deliberately not a second
cost model: the surface the UI shades is the same array A* searches, so what an
operator sees and what the planner does can never drift apart.

Traversability is reported as 1 - normalised cost, so 1 is easy ground and 0 is
the worst passable cell. Cells A* treats as impassable are reported separately
rather than being folded into the 0 end of the scale.
"""

from __future__ import annotations

import numpy as np

from ..navigation.cost_map import generate_cost_map


def compute_traversability(
    slope: np.ndarray,
    roughness: np.ndarray,
    hazard: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns (traversability, cost_map, stats).

    traversability is [0, 1] with impassable cells set to 0.
    cost_map is the raw array used by the planner, with np.inf preserved.
    """
    cost_map = generate_cost_map(slope, roughness, hazard, config)
    impassable = ~np.isfinite(cost_map)

    finite = cost_map[~impassable]
    if finite.size == 0:
        return np.zeros_like(cost_map), cost_map, {
            "impassable_fraction": 1.0,
            "mean_cost": None,
            "min_cost": None,
            "max_cost": None,
            "high_cost_fraction": 0.0,
        }

    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo if hi > lo else 1.0

    normalised = np.clip((cost_map - lo) / span, 0.0, 1.0)
    traversability = 1.0 - normalised
    traversability[impassable] = 0.0

    # "High cost" is the worst quartile of the passable range, a fixed
    # definition so the metric means the same thing between runs.
    high_cost_threshold = lo + 0.75 * span
    high_cost = np.count_nonzero(finite >= high_cost_threshold)

    stats = {
        "impassable_fraction": float(impassable.mean()),
        "mean_cost": float(finite.mean()),
        "min_cost": lo,
        "max_cost": hi,
        "high_cost_fraction": float(high_cost / finite.size),
        "high_cost_threshold": float(high_cost_threshold),
    }
    return traversability.astype(np.float32), cost_map, stats
