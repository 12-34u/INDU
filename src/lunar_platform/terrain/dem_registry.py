"""
Choosing the DEM that covers the ground you are standing on.

The project now holds two elevation products in different projections: a LOLA
polar stereographic tile covering 60S to the pole, and an SLDEM2015 simple
cylindrical tile covering 0-60N over part of the near side. Neither covers the
other's ground, and a DEM of the wrong region is the worst kind of wrong - it
reads cleanly, fills every cell and yields plausible terrain.

So nothing here picks a DEM by name or by configuration. Every discovered
product is asked whether it contains the point, and one that does not is not
offered. If none does, that is reported rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DemCandidate:
    """A discovered DEM and what it can be asked."""

    product_id: str
    path: Path
    kind: str  # "polar" | "equirectangular"
    posting_m: float
    dem: object

    def covers(self, longitude: float, latitude: float) -> bool:
        return bool(self.dem.covers(longitude, latitude))

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "path": str(self.path),
            "kind": self.kind,
            "posting_m": round(self.posting_m, 2),
        }


def discover_dems(dem_root: Path) -> list[DemCandidate]:
    """
    Every readable DEM under a directory, in either supported projection.

    A product that cannot be opened is skipped rather than raising: one broken
    tile should not make the others unavailable.
    """
    dem_root = Path(dem_root)
    found: list[DemCandidate] = []
    if not dem_root.exists():
        return found

    from .equirect_dem import EquirectangularDEM
    from .lola_dem import LolaPolarDEM

    # Polar products, identified by the PDS label beside their raster.
    for label in sorted(dem_root.rglob("*.lbl")) + sorted(dem_root.rglob("*.LBL")):
        raster = label.with_suffix(".img")
        if not raster.exists():
            raster = label.with_suffix(".IMG")
        if not raster.exists():
            continue
        try:
            dem = LolaPolarDEM(label)
            found.append(
                DemCandidate(dem.product_id, label, "polar", dem.pixel_size_m, dem)
            )
        except Exception:
            continue

    # Simple cylindrical tiles, identified by their sidecar.
    for sidecar in sorted(dem_root.rglob("*.json")):
        raster = sidecar.with_suffix(".IMG")
        if not raster.exists():
            raster = sidecar.with_suffix(".img")
        if not raster.exists():
            continue
        try:
            dem = EquirectangularDEM(raster, sidecar)
            found.append(
                DemCandidate(
                    dem.product_id, raster, "equirectangular", dem.pixel_size_m, dem
                )
            )
        except Exception:
            continue

    return found


def select_dem(
    candidates: list[DemCandidate], longitude: float, latitude: float
) -> Optional[DemCandidate]:
    """
    The finest-posted DEM that actually contains the point, or None.

    Finest wins because between two products that both cover the ground, the
    one with more detail is strictly better; coverage is the hard constraint
    and resolution is the preference.
    """
    covering = [c for c in candidates if c.covers(longitude, latitude)]
    if not covering:
        return None
    return min(covering, key=lambda c: c.posting_m)


def describe_selection(
    candidates: list[DemCandidate], longitude: float, latitude: float
) -> dict:
    """What was available, what was chosen, and why the rest were not."""
    chosen = select_dem(candidates, longitude, latitude)
    return {
        "point": {"longitude": round(longitude, 5), "latitude": round(latitude, 5)},
        "selected": chosen.to_dict() if chosen else None,
        "available": [
            {**c.to_dict(), "covers_point": c.covers(longitude, latitude)}
            for c in candidates
        ],
        "note": (
            "Chosen by coverage first, then finest posting. A DEM that does not "
            "contain the point is never used: it would read cleanly and describe "
            "the wrong ground."
        ),
    }
