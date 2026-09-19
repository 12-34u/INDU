"""Metrics measured along a planned route."""

from __future__ import annotations

import numpy as np


def route_metrics(
    path: list[tuple[int, int]],
    slope: np.ndarray,
    cost_map: np.ndarray,
    gsd_m: float,
    high_cost_threshold: float | None = None,
) -> dict:
    """
    Measure a planned route against the terrain it crosses.

    Every value is read off the route the planner returned; none is estimated.
    Length accounts for diagonal steps being longer than orthogonal ones.
    """
    if not path:
        return {
            "n_waypoints": 0,
            "length_m": 0.0,
            "max_slope_deg": None,
            "mean_slope_deg": None,
            "mean_cost": None,
            "high_cost_cells": 0,
        }

    xs = np.array([p[0] for p in path], dtype=int)
    ys = np.array([p[1] for p in path], dtype=int)

    steps = np.hypot(np.diff(xs), np.diff(ys))
    length_m = float(steps.sum() * gsd_m)

    slopes = slope[ys, xs]
    costs = cost_map[ys, xs]
    finite_costs = costs[np.isfinite(costs)]

    high_cost_cells = 0
    if high_cost_threshold is not None and finite_costs.size:
        high_cost_cells = int(np.count_nonzero(finite_costs >= high_cost_threshold))

    return {
        "n_waypoints": len(path),
        "length_m": round(length_m, 2),
        "max_slope_deg": round(float(slopes.max()), 2),
        "mean_slope_deg": round(float(slopes.mean()), 2),
        "mean_cost": round(float(finite_costs.mean()), 4) if finite_costs.size else None,
        "high_cost_cells": high_cost_cells,
    }
