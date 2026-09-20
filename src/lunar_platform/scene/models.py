"""
What the system knows about one image, and the collection of them per sector.

Registration between two arbitrary products fails more often than it succeeds,
and the reason is usually knowable in advance: they were lit differently, or
they barely overlap, or they are at incomparable scales. All three are
properties of the images that can be recorded once and consulted before any
feature is extracted.

So a view is not just pixels. It carries where it looks, how it was lit, and
how finely it samples the ground, because those are what decide whether two
views can be matched at all.

Nothing here reads image data. A library of several hundred views has to be
cheap to load and cheap to search; the pixels are fetched only once a view has
been chosen.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Mean lunar radius, for turning angles into ground distance.
MOON_RADIUS_M = 1737400.0


@dataclass
class Illumination:
    """
    Lighting geometry at a point on the ground, in degrees.

    Recorded per view because it is the single strongest predictor of whether
    two images of the same ground will match. Near the terminator the surface
    is mostly cast shadow, and shadow moves with the sun: two views of
    identical terrain at 2 and 7 degrees of sun elevation do not look alike.
    """

    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    incidence_deg: Optional[float] = None
    emission_deg: Optional[float] = None
    phase_deg: Optional[float] = None
    source: str = "unknown"  # "spice" | "label" | "unknown"

    @property
    def known(self) -> bool:
        return self.sun_elevation_deg is not None

    def shadow_length_ratio(self, other: "Illumination") -> Optional[float]:
        """
        How much longer shadows are in one view than the other.

        A feature of height h casts h/tan(elevation). This ratio is a more
        honest measure of appearance difference than the elevation difference
        itself: 2 degrees against 7 is a factor of 3.4, while 40 against 45 is
        barely 1.2, and the same 5 degree gap means very different things.
        """
        if not (self.known and other.known):
            return None
        a = math.tan(math.radians(max(self.sun_elevation_deg, 0.05)))
        b = math.tan(math.radians(max(other.sun_elevation_deg, 0.05)))
        if a <= 0 or b <= 0:
            return None
        longer, shorter = max(1 / a, 1 / b), min(1 / a, 1 / b)
        return float(longer / shorter)

    def azimuth_difference_deg(self, other: "Illumination") -> Optional[float]:
        if self.sun_azimuth_deg is None or other.sun_azimuth_deg is None:
            return None
        delta = abs(self.sun_azimuth_deg - other.sun_azimuth_deg) % 360.0
        return float(min(delta, 360.0 - delta))


@dataclass
class Footprint:
    """Ground covered by a view, as corner coordinates plus a centre."""

    corners: list[tuple[float, float]] = field(default_factory=list)  # (lon, lat)
    centre_longitude: Optional[float] = None
    centre_latitude: Optional[float] = None

    @property
    def known(self) -> bool:
        return self.centre_longitude is not None and self.centre_latitude is not None

    def bounds(self) -> Optional[tuple[float, float, float, float]]:
        """(min_lon, min_lat, max_lon, max_lat), or None without corners."""
        if not self.corners:
            return None
        lons = [c[0] for c in self.corners]
        lats = [c[1] for c in self.corners]
        return min(lons), min(lats), max(lons), max(lats)

    def separation_m(self, other: "Footprint") -> Optional[float]:
        """Great-circle distance between the two centres."""
        if not (self.known and other.known):
            return None
        phi1 = math.radians(self.centre_latitude)
        phi2 = math.radians(other.centre_latitude)
        dphi = phi2 - phi1
        dlam = math.radians(other.centre_longitude - self.centre_longitude)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        return float(2 * MOON_RADIUS_M * math.asin(math.sqrt(min(1.0, a))))

    def _local_xy(self, reference_latitude: float) -> list[tuple[float, float]]:
        """
        Corners in a local metric frame, so areas mean something.

        Degrees of longitude shrink with latitude, and near 71 degrees south a
        degree of longitude is a third of a degree of latitude. Computing an
        overlap area in raw degrees would badly distort any footprint away from
        the equator, which is every footprint this project handles.
        """
        scale = math.cos(math.radians(reference_latitude))
        return [
            (lon * scale * MOON_RADIUS_M * math.pi / 180.0,
             lat * MOON_RADIUS_M * math.pi / 180.0)
            for lon, lat in self.corners
        ]

    @staticmethod
    def _clip(subject: list, clipper: list) -> list:
        """Sutherland-Hodgman polygon clipping. Both must be convex."""

        def inside(point, a, b):
            return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0

        def intersect(p1, p2, a, b):
            x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
            x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denominator) < 1e-12:
                return p2
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        output = list(subject)
        for i in range(len(clipper)):
            a, b = clipper[i], clipper[(i + 1) % len(clipper)]
            current, output = output, []
            if not current:
                break
            previous = current[-1]
            for point in current:
                if inside(point, a, b):
                    if not inside(previous, a, b):
                        output.append(intersect(previous, point, a, b))
                    output.append(point)
                elif inside(previous, a, b):
                    output.append(intersect(previous, point, a, b))
                previous = point
        return output

    @staticmethod
    def _area(polygon: list) -> float:
        """Shoelace area, unsigned."""
        if len(polygon) < 3:
            return 0.0
        total = 0.0
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    def _ensure_ccw(self, polygon: list) -> list:
        signed = 0.0
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            signed += x1 * y2 - x2 * y1
        return polygon if signed >= 0 else polygon[::-1]

    def polygon_overlap(self, other: "Footprint") -> float:
        """
        Fraction of this footprint's area genuinely shared with another.

        Unlike bounds_overlap this respects the shape. It matters here: the
        OHRC strip and the NAC frame are long narrow ribbons crossing at an
        angle, so their bounding boxes overlap completely while the ground they
        actually share is a diagonal band across part of each.
        """
        if len(self.corners) < 3 or len(other.corners) < 3:
            return 0.0

        reference = self.centre_latitude if self.known else self.corners[0][1]
        subject = self._ensure_ccw(self._local_xy(reference))
        clipper = self._ensure_ccw(other._local_xy(reference))

        own_area = self._area(subject)
        if own_area <= 0:
            return 0.0
        return float(min(1.0, self._area(self._clip(subject, clipper)) / own_area))

    def bounds_overlap(self, other: "Footprint") -> float:
        """
        Fraction of this view's bounding box shared with another's.

        A bounding-box test, and honest about being one: two narrow strips
        crossing at an angle can share a box while sharing little ground. It is
        used to shortlist, never to conclude.
        """
        a, b = self.bounds(), other.bounds()
        if a is None or b is None:
            return 0.0

        lon_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        lat_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        if lon_overlap <= 0 or lat_overlap <= 0:
            return 0.0

        own = max((a[2] - a[0]) * (a[3] - a[1]), 1e-12)
        return float(min(1.0, (lon_overlap * lat_overlap) / own))


@dataclass
class SceneView:
    """One image in the library, described well enough to be chosen without reading it."""

    view_id: str
    sector_id: str
    sensor: str
    product_id: str
    # How to read the pixels when the view is actually selected. A GDAL
    # reference, so a member inside an unextracted archive is addressable.
    source_ref: str
    gsd_m: Optional[float] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    acquired_utc: Optional[str] = None
    footprint: Footprint = field(default_factory=Footprint)
    illumination: Illumination = field(default_factory=Illumination)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["footprint"]["corners"] = [list(c) for c in self.footprint.corners]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SceneView":
        footprint = Footprint(
            corners=[tuple(c) for c in data.get("footprint", {}).get("corners", [])],
            centre_longitude=data.get("footprint", {}).get("centre_longitude"),
            centre_latitude=data.get("footprint", {}).get("centre_latitude"),
        )
        illumination = Illumination(**(data.get("illumination") or {}))
        known = {f for f in cls.__dataclass_fields__ if f not in ("footprint", "illumination")}
        return cls(
            footprint=footprint,
            illumination=illumination,
            **{k: v for k, v in data.items() if k in known},
        )


@dataclass
class SceneLibrary:
    """The views available for one sector."""

    sector_id: str
    views: list[SceneView] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.views)

    def add(self, view: SceneView) -> None:
        self.views = [v for v in self.views if v.view_id != view.view_id]
        self.views.append(view)

    def by_id(self, view_id: str) -> Optional[SceneView]:
        return next((v for v in self.views if v.view_id == view_id), None)

    def by_sensor(self, sensor: str) -> list[SceneView]:
        return [v for v in self.views if v.sensor.upper() == sensor.upper()]

    def describe(self) -> dict:
        lit = [v for v in self.views if v.illumination.known]
        elevations = [v.illumination.sun_elevation_deg for v in lit]
        return {
            "sector_id": self.sector_id,
            "n_views": len(self.views),
            "sensors": sorted({v.sensor for v in self.views}),
            "n_georeferenced": sum(1 for v in self.views if v.footprint.known),
            "n_with_illumination": len(lit),
            "sun_elevation_range_deg": (
                [round(min(elevations), 2), round(max(elevations), 2)] if elevations else None
            ),
            "gsd_range_m": (
                [
                    round(min(v.gsd_m for v in self.views if v.gsd_m), 3),
                    round(max(v.gsd_m for v in self.views if v.gsd_m), 3),
                ]
                if any(v.gsd_m for v in self.views)
                else None
            ),
        }

    # -- persistence ---------------------------------------------------------

    def save(self, root: Path) -> Path:
        """Write to <root>/<sector_id>/library.json. Derived, so never under data/raw."""
        directory = Path(root) / self.sector_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "library.json"
        path.write_text(
            json.dumps(
                {
                    "sector_id": self.sector_id,
                    "views": [v.to_dict() for v in self.views],
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, root: Path, sector_id: str) -> "SceneLibrary":
        path = Path(root) / sector_id / "library.json"
        if not path.exists():
            return cls(sector_id=sector_id)
        data = json.loads(path.read_text())
        return cls(
            sector_id=data.get("sector_id", sector_id),
            views=[SceneView.from_dict(v) for v in data.get("views", [])],
        )
