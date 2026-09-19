"""
In-place readers for the Chandrayaan-2 OHRC bundle.

data/raw is an immutable archive, so nothing here extracts, moves or rewrites
anything. The browse product and the NAV geometry grid are both small and are
read straight out of the ZIP central directory; the 1.1 GB full-resolution
image member is never touched.

Two things come out of this module:

* the browse raster - a 10x decimation of the full image, which is the only
  image member cheap enough to read at runtime;
* the NAV geometry grid - sampled longitude/latitude on a regular
  (pixel, scan) lattice, which is the only geolocation this product carries.

The grid is not a camera model. It is sampled ground truth from the mission's
own processing, interpolated between nodes, and it is what lets a position in
the simulation be reported as real selenographic coordinates instead of bare
sector metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import zipfile

import cv2
import numpy as np

from .raw_inventory import read_geometry_grid

# Mean lunar radius (IAU). Used only to turn grid node spacing into metres.
MOON_RADIUS_M = 1737400.0

# Browse products are published as a 10x decimation of the full image. The
# value is checked against the label dimensions when both are available.
NOMINAL_BROWSE_DECIMATION = 10.0

_browse_cache: dict[tuple, np.ndarray] = {}
_grid_cache: dict[tuple, "GeometryGrid"] = {}


def _cache_key(archive: Path, member: str) -> tuple:
    stat = archive.stat()
    return (str(archive), member, stat.st_mtime_ns, stat.st_size)


def load_browse_image(archive: Optional[Path], member: Optional[str] = None) -> np.ndarray:
    """
    Decode the browse raster without extracting anything.

    Reads a member of the bundle when given an archive, or a file on disk when
    given a path and no member - data/raw sometimes holds an extracted copy of
    a bundle alongside the bundle itself. Returns a 2D uint8 array, cached
    because decoding costs a couple of hundred milliseconds and every
    observation reads the same raster.
    """
    if archive is None:
        raise ValueError("load_browse_image needs an archive or a file path")

    path = Path(archive)
    if member is None:
        key = _cache_key(path, "")
        cached = _browse_cache.get(key)
        if cached is not None:
            return cached
        payload = path.read_bytes()
        member = path.name
    else:
        key = _cache_key(path, member)
        cached = _browse_cache.get(key)
        if cached is not None:
            return cached
        with zipfile.ZipFile(path) as bundle:
            payload = bundle.read(member)

    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode browse product {member}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _browse_cache[key] = image
    return image


def great_circle_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Surface distance on a spherical Moon, in metres. Accepts arrays."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * MOON_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class GeometryGrid:
    """
    Longitude/latitude sampled on a regular (pixel, scan) lattice.

    `pixels` and `scans` are the lattice coordinates in FULL-resolution image
    space. `lon`/`lat` are (n_scans, n_pixels) arrays. Lookups bilinearly
    interpolate between nodes and clamp outside the lattice rather than
    extrapolating, because extrapolated geometry would be fabricated.
    """

    pixels: np.ndarray
    scans: np.ndarray
    lon: np.ndarray
    lat: np.ndarray

    @property
    def max_pixel(self) -> int:
        return int(self.pixels[-1])

    @property
    def max_scan(self) -> int:
        return int(self.scans[-1])

    def lonlat_at(self, pixel: float, scan: float) -> tuple[float, float]:
        """Interpolate (longitude, latitude) at a full-resolution image point."""
        lon, lat = self.lonlat_at_many(np.array([pixel], float), np.array([scan], float))
        return float(lon[0]), float(lat[0])

    def lonlat_at_many(
        self, pixel: np.ndarray, scan: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised bilinear interpolation over the lattice."""
        px = np.clip(np.asarray(pixel, float), self.pixels[0], self.pixels[-1])
        sc = np.clip(np.asarray(scan, float), self.scans[0], self.scans[-1])

        j = np.clip(np.searchsorted(self.pixels, px) - 1, 0, len(self.pixels) - 2)
        i = np.clip(np.searchsorted(self.scans, sc) - 1, 0, len(self.scans) - 2)

        x0, x1 = self.pixels[j], self.pixels[j + 1]
        y0, y1 = self.scans[i], self.scans[i + 1]
        tx = np.where(x1 > x0, (px - x0) / np.maximum(x1 - x0, 1e-9), 0.0)
        ty = np.where(y1 > y0, (sc - y0) / np.maximum(y1 - y0, 1e-9), 0.0)

        def bilinear(grid: np.ndarray) -> np.ndarray:
            g00 = grid[i, j]
            g01 = grid[i, j + 1]
            g10 = grid[i + 1, j]
            g11 = grid[i + 1, j + 1]
            return (
                g00 * (1 - tx) * (1 - ty)
                + g01 * tx * (1 - ty)
                + g10 * (1 - tx) * ty
                + g11 * tx * ty
            )

        return bilinear(self.lon), bilinear(self.lat)

    def ground_sampling_m(self, pixel: float, scan: float) -> tuple[float, float]:
        """
        Measured metres per FULL-resolution pixel near a point, as
        (across_track, along_track).

        This is derived from the grid rather than taken from the label's
        nominal pixel_resolution, because the two disagree: the label states a
        nominal value while the grid records the geometry actually achieved.
        """
        step_p = float(self.pixels[1] - self.pixels[0])
        step_s = float(self.scans[1] - self.scans[0])

        lon0, lat0 = self.lonlat_at(pixel, scan)
        lon_p, lat_p = self.lonlat_at(pixel + step_p, scan)
        lon_s, lat_s = self.lonlat_at(pixel, scan + step_s)

        across = float(great_circle_m(lat0, lon0, lat_p, lon_p)) / step_p
        along = float(great_circle_m(lat0, lon0, lat_s, lon_s)) / step_s
        return across, along

    def nearest_image_point(self, longitude: float, latitude: float) -> tuple[float, float]:
        """
        Full-resolution (pixel, scan) whose grid node is closest to a lon/lat.

        Node resolution only - the lattice is ~100 px apart, so this places a
        target to within roughly half a node, which is ample for anchoring a
        sector. It does not invert the geometry.
        """
        distance = great_circle_m(self.lat, self.lon, latitude, longitude)
        i, j = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
        return float(self.pixels[j]), float(self.scans[i])

    def contains(self, longitude: float, latitude: float, tolerance_m: float = 5000.0) -> bool:
        distance = great_circle_m(self.lat, self.lon, latitude, longitude)
        return bool(np.nanmin(distance) <= tolerance_m)


def _read_loose_geometry_grid(path: Path) -> list[dict]:
    """Read a NAV grid CSV that sits on disk rather than inside a bundle."""
    import csv

    rows: list[dict] = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "longitude": float(row["Longitude"]),
                        "latitude": float(row["Latitude"]),
                        "pixel": int(row["Pixel"]),
                        "scan": int(row["Scan"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def load_geometry_grid(archive: Path, member: Optional[str] = None) -> GeometryGrid:
    """
    Read the NAV grid in place and reshape it into a lattice.

    Accepts a bundle member or a loose CSV, matching load_browse_image.
    """
    archive = Path(archive)
    key = _cache_key(archive, member or "")
    cached = _grid_cache.get(key)
    if cached is not None:
        return cached

    rows = (
        read_geometry_grid(archive, member)
        if member is not None
        else _read_loose_geometry_grid(archive)
    )
    if not rows:
        raise ValueError(f"Geometry grid {member or archive} is empty")

    pixels = np.array(sorted({r["pixel"] for r in rows}), dtype=float)
    scans = np.array(sorted({r["scan"] for r in rows}), dtype=float)
    pixel_index = {int(v): i for i, v in enumerate(pixels)}
    scan_index = {int(v): i for i, v in enumerate(scans)}

    lon = np.full((len(scans), len(pixels)), np.nan)
    lat = np.full_like(lon, np.nan)
    for row in rows:
        i = scan_index[row["scan"]]
        j = pixel_index[row["pixel"]]
        lon[i, j] = row["longitude"]
        lat[i, j] = row["latitude"]

    # A partially populated lattice would interpolate through NaNs and produce
    # silently wrong coordinates, so refuse it rather than patch it.
    if np.isnan(lon).any() or np.isnan(lat).any():
        missing = int(np.isnan(lon).sum())
        raise ValueError(
            f"Geometry grid {member or archive} is not a complete lattice: "
            f"{missing} nodes missing"
        )

    grid = GeometryGrid(pixels=pixels, scans=scans, lon=lon, lat=lat)
    _grid_cache[key] = grid
    return grid


def browse_decimation(
    browse_shape: tuple[int, int], label: Optional[dict] = None
) -> float:
    """
    How many full-resolution pixels one browse pixel spans.

    Taken from the label's stated full dimensions when they are present, so a
    bundle published at a different decimation is handled correctly, and only
    falling back to the nominal 10x when the label is silent.
    """
    if label:
        lines, samples = label.get("lines"), label.get("samples")
        if lines and samples:
            height, width = browse_shape[:2]
            if height > 0 and width > 0:
                return float(np.mean([float(lines) / height, float(samples) / width]))
    return NOMINAL_BROWSE_DECIMATION
