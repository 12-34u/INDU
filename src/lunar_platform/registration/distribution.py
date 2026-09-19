"""
Spatial distribution of match points.

Feature matchers concentrate their strongest matches wherever texture is
richest, which on lunar imagery means a handful of fresh craters. A transform
fitted from one such cluster is well constrained there and extrapolates badly
everywhere else, and its RMSE looks excellent while doing so.

This enforces coverage instead: the source image is divided into a grid and
only the best few matches per cell are kept, so the surviving set spans the
frame. Selection is by match score, so within a cell the strongest still win.
"""

from __future__ import annotations

import numpy as np


def select_uniform_matches(
    pts_source: np.ndarray,
    pts_reference: np.ndarray,
    scores: np.ndarray,
    image_width: int,
    image_height: int,
    grid_cols: int = 8,
    grid_rows: int = 8,
    max_per_cell: int = 4,
) -> np.ndarray:
    """
    Return indices of matches kept after per-cell capping.

    Cells are defined on the source image. Returned indices are sorted so the
    caller can reorder all parallel arrays consistently.
    """
    n = len(pts_source)
    if n == 0:
        return np.empty(0, dtype=int)
    if grid_cols < 1 or grid_rows < 1 or max_per_cell < 1:
        raise ValueError("grid_cols, grid_rows and max_per_cell must all be >= 1")

    cell_w = max(image_width / grid_cols, 1e-6)
    cell_h = max(image_height / grid_rows, 1e-6)

    col = np.clip((pts_source[:, 0] / cell_w).astype(int), 0, grid_cols - 1)
    row = np.clip((pts_source[:, 1] / cell_h).astype(int), 0, grid_rows - 1)
    cell_id = row * grid_cols + col

    keep: list[int] = []
    for cell in np.unique(cell_id):
        members = np.flatnonzero(cell_id == cell)
        if len(members) > max_per_cell:
            # Strongest first, then cap.
            members = members[np.argsort(scores[members])[::-1][:max_per_cell]]
        keep.extend(members.tolist())

    return np.array(sorted(keep), dtype=int)


def occupancy_grid(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    grid_cols: int = 8,
    grid_rows: int = 8,
) -> np.ndarray:
    """Count of points per grid cell, for reporting coverage in the UI."""
    counts = np.zeros((grid_rows, grid_cols), dtype=int)
    if len(points) == 0:
        return counts

    cell_w = max(image_width / grid_cols, 1e-6)
    cell_h = max(image_height / grid_rows, 1e-6)

    for x, y in points:
        if 0 <= x < image_width and 0 <= y < image_height:
            counts[
                min(int(y / cell_h), grid_rows - 1),
                min(int(x / cell_w), grid_cols - 1),
            ] += 1
    return counts


def distribution_metrics(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    grid_cols: int = 8,
    grid_rows: int = 8,
) -> dict:
    """
    Coverage and evenness of a point set.

    occupied_fraction answers "how much of the frame is constrained".
    gini answers "is it evenly constrained" - 0 is perfectly even, and a value
    near 1 means the points are piled into a few cells even if many cells are
    technically occupied.
    """
    counts = occupancy_grid(points, image_width, image_height, grid_cols, grid_rows)
    total_cells = grid_cols * grid_rows
    occupied = int(np.count_nonzero(counts))

    flat = np.sort(counts.ravel().astype(float))
    total = flat.sum()
    if total > 0:
        index = np.arange(1, flat.size + 1)
        gini = float(
            (2.0 * np.sum(index * flat)) / (flat.size * total) - (flat.size + 1) / flat.size
        )
    else:
        gini = 0.0

    return {
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "occupied_cells": occupied,
        "total_cells": total_cells,
        "occupied_fraction": round(occupied / total_cells, 4),
        "max_points_in_cell": int(counts.max()) if counts.size else 0,
        "gini": round(gini, 4),
    }
