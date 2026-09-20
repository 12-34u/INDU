"""
Placing an image by agreement between several views rather than one pairing.

A single registration reports its own residuals, and residuals only say how
well a transform explains the matches that produced it. They cannot say the
matches were the right ones. A confidently wrong fix has small residuals; that
is what makes it confidently wrong.

Several views cannot conspire in the same way. Each is a different image, taken
at a different time under different lighting, and a wrong correspondence in one
has no reason to agree with a wrong correspondence in another. So a position
several views independently land on is evidence of a different kind from one
position with a tight fit.

Two honesty rules follow, and are enforced below:

* a lone contributing view yields a position but no agreement, and is reported
  that way rather than given a confidence it has not earned;
* views that disagree are reported, not discarded quietly. A split vote is a
  result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MOON_RADIUS_M = 1737400.0

# Two views are treated as agreeing when their positions fall within this of
# each other. Wide enough to absorb per-view geolocation error, far below the
# scale at which two genuinely different answers would both look right.
DEFAULT_AGREEMENT_M = 250.0


@dataclass
class ViewFix:
    """What one view had to say about where the query is."""

    view_id: str
    sensor: str
    succeeded: bool
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    rmse_px: Optional[float] = None
    reason: str = ""
    agreed: Optional[bool] = None
    offset_from_consensus_m: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "sensor": self.sensor,
            "succeeded": self.succeeded,
            "longitude": round(self.longitude, 6) if self.longitude is not None else None,
            "latitude": round(self.latitude, 6) if self.latitude is not None else None,
            "n_inliers": self.n_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "rmse_px": self.rmse_px,
            "agreed": self.agreed,
            "offset_from_consensus_m": (
                round(self.offset_from_consensus_m, 1)
                if self.offset_from_consensus_m is not None
                else None
            ),
            "reason": self.reason,
        }


@dataclass
class ConsensusFix:
    """A position several views agree on, with the dissent recorded."""

    longitude: Optional[float]
    latitude: Optional[float]
    n_attempted: int
    n_registered: int
    n_agreeing: int
    spread_m: Optional[float]
    confidence: float
    cross_checked: bool
    note: str
    contributions: list[ViewFix] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.longitude is not None

    def to_dict(self) -> dict:
        return {
            "longitude": round(self.longitude, 6) if self.longitude is not None else None,
            "latitude": round(self.latitude, 6) if self.latitude is not None else None,
            "n_attempted": self.n_attempted,
            "n_registered": self.n_registered,
            "n_agreeing": self.n_agreeing,
            "spread_m": round(self.spread_m, 1) if self.spread_m is not None else None,
            "confidence": round(self.confidence, 3),
            "cross_checked": self.cross_checked,
            "note": self.note,
            "contributions": [c.to_dict() for c in self.contributions],
        }


def great_circle_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return float(2 * MOON_RADIUS_M * math.asin(math.sqrt(min(1.0, a))))


def _mean_lonlat(points: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Average positions as directions, not as numbers.

    Averaging longitudes arithmetically is wrong near the meridian and wrong
    again near a pole, where a small ground distance spans a large change in
    longitude. This project's sector is at 71 degrees south, so that is not a
    hypothetical.
    """
    vectors = []
    for longitude, latitude in points:
        lon, lat = math.radians(longitude), math.radians(latitude)
        vectors.append(
            (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
        )
    mean = np.mean(np.array(vectors), axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        return points[0]
    mean = mean / norm
    return (
        float(math.degrees(math.atan2(mean[1], mean[0]))),
        float(math.degrees(math.asin(float(np.clip(mean[2], -1.0, 1.0))))),
    )


def consensus_position(
    fixes: list[ViewFix],
    agreement_m: float = DEFAULT_AGREEMENT_M,
) -> ConsensusFix:
    """
    Combine per-view positions into one, keeping the largest agreeing group.

    The winner is the position with the most support rather than the mean of
    everything: a single bad registration should not drag the answer toward
    itself, and averaging all of them lets it.
    """
    attempted = len(fixes)
    placed = [f for f in fixes if f.succeeded and f.longitude is not None]

    if not placed:
        reasons = "; ".join(f.reason for f in fixes if f.reason) or "no views registered"
        return ConsensusFix(
            None, None, attempted, 0, 0, None, 0.0, False,
            f"No view produced a position. {reasons}",
            contributions=fixes,
        )

    # Support: how many other views land within the agreement distance.
    supports = []
    for candidate in placed:
        supports.append(
            sum(
                1
                for other in placed
                if great_circle_m(
                    candidate.longitude, candidate.latitude, other.longitude, other.latitude
                )
                <= agreement_m
            )
        )

    seed = placed[int(np.argmax(supports))]
    cluster = [
        f
        for f in placed
        if great_circle_m(seed.longitude, seed.latitude, f.longitude, f.latitude) <= agreement_m
    ]

    longitude, latitude = _mean_lonlat([(f.longitude, f.latitude) for f in cluster])

    offsets = [great_circle_m(longitude, latitude, f.longitude, f.latitude) for f in cluster]
    spread = float(np.sqrt(np.mean(np.square(offsets)))) if offsets else 0.0

    cluster_ids = {f.view_id for f in cluster}
    for fix in fixes:
        if fix.succeeded and fix.longitude is not None:
            fix.agreed = fix.view_id in cluster_ids
            fix.offset_from_consensus_m = great_circle_m(
                longitude, latitude, fix.longitude, fix.latitude
            )

    cross_checked = len(cluster) >= 2
    dissenting = len(placed) - len(cluster)

    if not cross_checked:
        confidence = 0.25
        note = (
            "One view produced a position, so there is nothing to check it against. "
            "The position may be right; it is simply uncorroborated."
        )
    else:
        # Agreement among those that registered, tightened by how closely.
        agreement_fraction = len(cluster) / len(placed)
        tightness = 1.0 / (1.0 + spread / max(agreement_m, 1e-6))
        confidence = float(min(0.95, agreement_fraction * tightness))
        note = (
            f"{len(cluster)} of {len(placed)} views that registered agree to within "
            f"{spread:.0f} m."
        )
        if dissenting:
            note += (
                f" {dissenting} disagreed and {'is' if dissenting == 1 else 'are'} "
                "reported rather than dropped."
            )

    return ConsensusFix(
        longitude=longitude,
        latitude=latitude,
        n_attempted=attempted,
        n_registered=len(placed),
        n_agreeing=len(cluster),
        spread_m=spread,
        confidence=confidence,
        cross_checked=cross_checked,
        note=note,
        contributions=fixes,
    )


def _project_query_centre(outcome, geolocator) -> Optional[tuple[float, float]]:
    """
    Where the query's centre lands on the ground, via one registered view.

    Three coordinate systems have to be crossed and none of them can be
    skipped: registration works on decimated, windowed copies, so its transform
    is in working pixels; the view's geometry is defined on its own native
    pixels; only then is there a longitude and latitude.
    """
    if outcome.transform is None or outcome.source_image is None:
        return None

    height, width = outcome.source_image.shape[:2]
    centre = np.array([width / 2.0, height / 2.0, 1.0])

    projected = outcome.transform @ centre
    if projected.shape[0] == 3:
        if abs(projected[2]) < 1e-12:
            return None
        projected = projected[:2] / projected[2]
    else:
        projected = projected[:2]

    frame = outcome.reference_frame
    column, row = (
        frame.to_native(float(projected[0]), float(projected[1]))
        if frame
        else (float(projected[0]), float(projected[1]))
    )
    return geolocator(column, row)


def localize_against_library(
    query_ref: str,
    query_footprint,
    query_illumination,
    query_gsd_m: Optional[float],
    library,
    manager,
    k: int = 3,
    config: Optional[dict] = None,
    agreement_m: float = DEFAULT_AGREEMENT_M,
    exclude_view_ids: Optional[set] = None,
) -> ConsensusFix:
    """
    Place an image by registering it against the best few views and combining.

    Retrieval decides what is worth trying; each attempt is made independently
    so one failure does not end the run; consensus decides what the answers add
    up to. A view that cannot be geolocated is attempted anyway and reported as
    unable to contribute, because that is a property of the library rather than
    of the query.
    """
    from ..registration.pipeline import run_registration
    from ..scene.geolocate import geolocator_for
    from ..scene.retrieval import RetrievalQuery, retrieve

    query = RetrievalQuery(
        footprint=query_footprint,
        illumination=query_illumination,
        gsd_m=query_gsd_m,
    )
    candidates = retrieve(library, query, k=k, exclude_view_ids=exclude_view_ids)

    if not candidates:
        return ConsensusFix(
            None, None, 0, 0, 0, None, 0.0, False,
            "No view in the library shares ground with the query.",
        )

    fixes: list[ViewFix] = []
    for candidate in candidates:
        view = candidate.view
        fix = ViewFix(view_id=view.view_id, sensor=view.sensor, succeeded=False)

        geolocator = geolocator_for(view, manager)
        if geolocator is None:
            fix.reason = "View has no geometry, so it cannot turn a match into a position."
            fixes.append(fix)
            continue

        try:
            outcome = run_registration(
                query_ref,
                view.source_ref,
                f"multiview_{view.view_id.replace(':', '_')}",
                dict(config or {}),
                data_manager=manager,
            )
        except Exception as exc:
            fix.reason = f"Registration raised: {exc}"
            fixes.append(fix)
            continue

        metrics = outcome.metrics or {}
        fix.n_inliers = int(metrics.get("n_verified_inliers", 0))
        fix.inlier_ratio = float(metrics.get("inlier_ratio", 0.0))
        fix.rmse_px = metrics.get("rmse_px")

        if not outcome.succeeded:
            reasons = metrics.get("quality_reasons") or outcome.notes
            fix.reason = reasons[0] if reasons else "Registration did not pass its gates."
            fixes.append(fix)
            continue

        ground = _project_query_centre(outcome, geolocator)
        if ground is None:
            fix.reason = "Registered, but the query centre could not be placed on the ground."
            fixes.append(fix)
            continue

        fix.succeeded = True
        fix.longitude, fix.latitude = ground
        fix.reason = f"{fix.n_inliers} inliers, rmse {fix.rmse_px} px"
        fixes.append(fix)

    return consensus_position(fixes, agreement_m=agreement_m)
