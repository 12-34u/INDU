"""
Four-quadrant decomposition of a rover observation.

Each quadrant is carried as a view plus its origin, so a detector can run on
the sub-image independently while every point it returns can be mapped back to
full-image coordinates without the caller tracking offsets.

    +-----+-----+
    | Q1  | Q2  |
    +-----+-----+
    | Q3  | Q4  |
    +-----+-----+
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..crater_mapping.models import Quadrant

QUADRANT_ORDER: tuple[Quadrant, ...] = ("Q1", "Q2", "Q3", "Q4")


@dataclass(frozen=True)
class QuadrantView:
    name: Quadrant
    image: np.ndarray
    x_off: int
    y_off: int

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def to_full(self, u: float, v: float) -> tuple[float, float]:
        """Map a point in this quadrant back to full-image coordinates."""
        return u + self.x_off, v + self.y_off

    def bounds(self) -> dict:
        return {
            "name": self.name,
            "x": self.x_off,
            "y": self.y_off,
            "width": self.width,
            "height": self.height,
        }


def split_quadrants(image: np.ndarray) -> list[QuadrantView]:
    """Split an observation into Q1..Q4. Odd sizes put the extra row/column in Q3/Q4."""
    h, w = image.shape[:2]
    mid_y, mid_x = h // 2, w // 2

    return [
        QuadrantView("Q1", image[:mid_y, :mid_x], 0, 0),
        QuadrantView("Q2", image[:mid_y, mid_x:], mid_x, 0),
        QuadrantView("Q3", image[mid_y:, :mid_x], 0, mid_y),
        QuadrantView("Q4", image[mid_y:, mid_x:], mid_x, mid_y),
    ]
