"""
Choosing which views are worth matching against.

Matching is expensive and, between badly-chosen views, futile. This ranks the
library before any feature is extracted, on the three properties that actually
decided the outcome when this project tried to register its one cross-
instrument pair:

* **ground shared** — by polygon intersection, not bounding box. The products
  here are long narrow ribbons crossing at an angle: their boxes overlap
  completely while the ground they share is a diagonal band across part of
  each.
* **illumination** — scored through shadow length rather than raw elevation
  difference. Two degrees of sun against seven is a 3.4x difference in shadow
  length and the images do not look alike; the same five-degree gap at forty
  degrees is barely 1.2x and hardly matters.
* **scale** — the ratio of ground sample distances, since a large ratio has to
  be resampled away before matching and costs detail doing it.

Illumination is weighted highest deliberately. It is the property that made the
one pair in this archive unmatchable, and no amount of overlap compensates for
a scene lit from a different angle at a different elevation.

Nothing here reads pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .models import Footprint, Illumination, SceneLibrary, SceneView

# Weights over the three scores. Illumination leads because it is what fails.
WEIGHT_OVERLAP = 0.35
WEIGHT_ILLUMINATION = 0.45
WEIGHT_SCALE = 0.20

# Below this shared fraction there is not enough common ground to bother.
MIN_OVERLAP = 0.02


@dataclass
class RetrievalQuery:
    """What we are trying to place."""

    footprint: Footprint
    illumination: Illumination = field(default_factory=Illumination)
    gsd_m: Optional[float] = None
    sensor: Optional[str] = None


@dataclass
class RetrievalCandidate:
    """One library view, scored against a query, with its reasoning."""

    view: SceneView
    overlap: float
    illumination_score: float
    scale_score: float
    score: float
    shadow_ratio: Optional[float]
    azimuth_difference_deg: Optional[float]
    scale_ratio: Optional[float]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "view_id": self.view.view_id,
            "sensor": self.view.sensor,
            "score": round(self.score, 4),
            "overlap": round(self.overlap, 4),
            "illumination_score": round(self.illumination_score, 4),
            "scale_score": round(self.scale_score, 4),
            "shadow_ratio": round(self.shadow_ratio, 3) if self.shadow_ratio else None,
            "azimuth_difference_deg": (
                round(self.azimuth_difference_deg, 2)
                if self.azimuth_difference_deg is not None
                else None
            ),
            "scale_ratio": round(self.scale_ratio, 3) if self.scale_ratio else None,
            "sun_elevation_deg": self.view.illumination.sun_elevation_deg,
            "gsd_m": self.view.gsd_m,
            "reasons": list(self.reasons),
        }


def illumination_similarity(
    query: Illumination, candidate: Illumination
) -> tuple[float, Optional[float], Optional[float]]:
    """
    How alike two scenes will look, from 0 to 1.

    Returns the score with the shadow ratio and azimuth difference that
    produced it, so a ranking can always explain itself.

    Unknown illumination scores in the middle rather than at either extreme: a
    view that has not said how it was lit should neither be promoted over one
    that matches nor discarded below one that does not.
    """
    if not (query.known and candidate.known):
        return 0.5, None, None

    ratio = query.shadow_length_ratio(candidate) or 1.0
    # 1.0 when shadows are the same length, falling away as they diverge.
    length_score = 1.0 / ratio

    azimuth = query.azimuth_difference_deg(candidate)
    if azimuth is None:
        direction_score = 1.0
    else:
        # Shadows pointing the same way matter as much as being the same
        # length; opposed lighting inverts the relief entirely.
        direction_score = 0.5 + 0.5 * math.cos(math.radians(azimuth))

    return float(length_score * direction_score), ratio, azimuth


def scale_similarity(query_gsd: Optional[float], candidate_gsd: Optional[float]):
    """1.0 at equal sampling, falling as the ratio grows."""
    if not query_gsd or not candidate_gsd:
        return 0.5, None
    ratio = max(query_gsd, candidate_gsd) / min(query_gsd, candidate_gsd)
    return float(1.0 / ratio), float(ratio)


def score_view(query: RetrievalQuery, view: SceneView) -> RetrievalCandidate:
    """Score one view, recording why."""
    reasons: list[str] = []

    overlap = (
        query.footprint.polygon_overlap(view.footprint)
        if query.footprint.corners and view.footprint.corners
        else 0.0
    )
    if overlap <= 0 and query.footprint.known and view.footprint.known:
        # Footprints may be missing corners; fall back to a centre distance so
        # a view is not discarded merely for being described less fully.
        separation = query.footprint.separation_m(view.footprint)
        if separation is not None and separation < 50_000:
            overlap = 0.0
            reasons.append(
                f"No polygon overlap; centres are {separation / 1000:.1f} km apart."
            )

    illumination_score, shadow_ratio, azimuth = illumination_similarity(
        query.illumination, view.illumination
    )
    scale_score, scale_ratio = scale_similarity(query.gsd_m, view.gsd_m)

    if shadow_ratio is not None:
        if shadow_ratio > 2.0:
            reasons.append(
                f"Shadows differ by {shadow_ratio:.1f}x "
                f"({query.illumination.sun_elevation_deg:.1f}° against "
                f"{view.illumination.sun_elevation_deg:.1f}° sun elevation); "
                "the same ground will not look the same."
            )
        elif shadow_ratio < 1.3:
            reasons.append(f"Similarly lit: shadows within {shadow_ratio:.2f}x.")
    else:
        reasons.append("Illumination unknown for one side; scored neutrally.")

    if scale_ratio and scale_ratio > 2.0:
        reasons.append(
            f"Ground sampling differs by {scale_ratio:.1f}x; the finer image must "
            "be resampled down to match."
        )

    score = (
        WEIGHT_OVERLAP * overlap
        + WEIGHT_ILLUMINATION * illumination_score
        + WEIGHT_SCALE * scale_score
    )

    return RetrievalCandidate(
        view=view,
        overlap=overlap,
        illumination_score=illumination_score,
        scale_score=scale_score,
        score=score,
        shadow_ratio=shadow_ratio,
        azimuth_difference_deg=azimuth,
        scale_ratio=scale_ratio,
        reasons=reasons,
    )


def retrieve(
    library: SceneLibrary,
    query: RetrievalQuery,
    k: int = 5,
    min_overlap: float = MIN_OVERLAP,
    exclude_view_ids: Optional[set] = None,
) -> list[RetrievalCandidate]:
    """
    The k best views to try matching against, best first.

    Views sharing no ground are dropped rather than ranked low: there is
    nothing to match, and leaving them in the list invites something
    downstream to try.
    """
    excluded = exclude_view_ids or set()
    candidates = [
        score_view(query, view)
        for view in library.views
        if view.view_id not in excluded
    ]

    usable = [c for c in candidates if c.overlap >= min_overlap]
    usable.sort(key=lambda c: c.score, reverse=True)
    return usable[:k]


def query_from_view(view: SceneView) -> RetrievalQuery:
    """Treat an existing view as the thing to be placed. Useful for self-tests."""
    return RetrievalQuery(
        footprint=view.footprint,
        illumination=view.illumination,
        gsd_m=view.gsd_m,
        sensor=view.sensor,
    )
