"""
Simple-cylindrical DEM tiles, read straight from a flat binary raster.

terrain/lola_dem.py handles the LOLA polar products, which carry a PDS label
that GDAL can parse. SLDEM2015 tiles are distributed as bare rasters: no
label, no georeferencing, nothing but the filename to say what ground they
cover. So the geometry is recorded in a sidecar JSON next to the file, written
once and verified, rather than inferred at read time from a filename that could
be anything.

Two things about these products are worth stating because getting either wrong
produces a DEM that reads cleanly and is silently wrong:

* the values are in KILOMETRES relative to a 1737.4 km datum, not metres. A
  patch a few kilometres across shows about 0.03 units of relief, which is
  35 m as kilometres and 3 cm as metres. Only one of those is a real surface.
* the tile covers a latitude and longitude window. SLDEM2015 as a family spans
  60S to 60N, so it can never cover a polar sector however many tiles are
  fetched. `covers` is what callers should ask before sampling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .lola_dem import SectorElevation

# Bytes per sample, by the PDS sample type these products use.
DTYPE_BY_BITS = {32: "<f4", 16: "<i2"}


@dataclass
class EquirectSpec:
    """Georeferencing for a simple-cylindrical tile, from its sidecar."""

    lines: int
    line_samples: int
    minimum_latitude: float
    maximum_latitude: float
    westernmost_longitude: float
    easternmost_longitude: float
    sample_bits: int
    scaling_factor_to_metres: float
    offset_m: float
    product: str
    map_resolution_ppd: float

    @property
    def dtype(self) -> str:
        if self.sample_bits not in DTYPE_BY_BITS:
            raise ValueError(f"Unsupported sample_bits {self.sample_bits}")
        return DTYPE_BY_BITS[self.sample_bits]

    @property
    def expected_bytes(self) -> int:
        return self.lines * self.line_samples * (self.sample_bits // 8)

    @property
    def degrees_per_pixel(self) -> float:
        return 1.0 / self.map_resolution_ppd


class EquirectangularDEM:
    """A simple-cylindrical DEM tile addressed by longitude and latitude."""

    def __init__(self, raster_path: Path, spec_path: Optional[Path] = None):
        self.raster_path = Path(raster_path)
        if not self.raster_path.exists():
            raise FileNotFoundError(f"DEM raster not found: {self.raster_path}")

        spec_path = Path(spec_path) if spec_path else self.raster_path.with_suffix(".json")
        if not spec_path.exists():
            raise FileNotFoundError(
                f"{self.raster_path.name} has no PDS label and no sidecar at "
                f"{spec_path.name}. Its georeferencing has to be stated somewhere; "
                "guessing it from the filename would be a silent source of error."
            )

        raw = json.loads(Path(spec_path).read_text())
        self.spec = EquirectSpec(
            lines=int(raw["lines"]),
            line_samples=int(raw["line_samples"]),
            minimum_latitude=float(raw["minimum_latitude"]),
            maximum_latitude=float(raw["maximum_latitude"]),
            westernmost_longitude=float(raw["westernmost_longitude"]),
            easternmost_longitude=float(raw["easternmost_longitude"]),
            sample_bits=int(raw.get("sample_bits", 32)),
            scaling_factor_to_metres=float(raw.get("scaling_factor_to_metres", 1.0)),
            offset_m=float(raw.get("offset_m", 0.0)),
            product=str(raw.get("product", self.raster_path.stem)),
            map_resolution_ppd=float(raw["map_resolution_ppd"]),
        )

        actual = self.raster_path.stat().st_size
        if actual != self.spec.expected_bytes:
            raise ValueError(
                f"{self.raster_path.name} is {actual} bytes but its sidecar describes "
                f"{self.spec.expected_bytes}. One of the two is wrong, and sampling "
                "either way would return the wrong ground."
            )

    @property
    def product_id(self) -> str:
        return self.spec.product

    @property
    def pixel_size_m(self) -> float:
        """Ground sampling at the equator, where this projection is truthful."""
        metres_per_degree = 1737400.0 * np.pi / 180.0
        return float(self.spec.degrees_per_pixel * metres_per_degree)

    # -- coverage ------------------------------------------------------------

    @staticmethod
    def _wrap_longitude(longitude: np.ndarray) -> np.ndarray:
        return np.mod(np.asarray(longitude, float), 360.0)

    def covers(self, longitude: float, latitude: float) -> bool:
        """Whether a point falls inside this tile. Ask before sampling."""
        spec = self.spec
        lon = float(self._wrap_longitude(np.array([longitude]))[0])
        west = spec.westernmost_longitude % 360.0
        east = spec.easternmost_longitude % 360.0
        if east <= west:  # tile crosses the prime meridian
            inside_lon = lon >= west or lon <= east
        else:
            inside_lon = west <= lon <= east
        return bool(inside_lon and spec.minimum_latitude <= latitude <= spec.maximum_latitude)

    # -- sampling ------------------------------------------------------------

    def lonlat_to_pixel(
        self, longitude: np.ndarray, latitude: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fractional column/row for longitude and latitude arrays."""
        spec = self.spec
        lon = self._wrap_longitude(longitude)
        west = spec.westernmost_longitude % 360.0

        # Offset east of the tile's western edge, wrapped so a tile spanning the
        # prime meridian still yields increasing columns.
        east_offset = np.mod(lon - west, 360.0)
        cols = east_offset * spec.map_resolution_ppd
        # Row 0 is the northernmost line.
        rows = (spec.maximum_latitude - np.asarray(latitude, float)) * spec.map_resolution_ppd
        return cols, rows

    def read_window(self, col_off: int, row_off: int, width: int, height: int) -> np.ndarray:
        """
        Read a rectangular window, in metres, without loading the whole tile.

        These tiles are ~1.8 GB; only the rows a sector touches are read.
        """
        spec = self.spec
        col_off = max(0, min(col_off, spec.line_samples - 1))
        row_off = max(0, min(row_off, spec.lines - 1))
        width = max(1, min(width, spec.line_samples - col_off))
        height = max(1, min(height, spec.lines - row_off))

        itemsize = spec.sample_bits // 8
        out = np.empty((height, width), dtype=np.float64)
        with open(self.raster_path, "rb") as handle:
            for i in range(height):
                offset = ((row_off + i) * spec.line_samples + col_off) * itemsize
                handle.seek(offset)
                out[i] = np.frombuffer(handle.read(width * itemsize), dtype=spec.dtype)

        return out * spec.scaling_factor_to_metres + spec.offset_m

    def sample_sector(self, basemap, output_px: Optional[int] = None) -> SectorElevation:
        """Sample this tile onto a sector basemap's grid, in metres."""
        if basemap.grid is None:
            raise ValueError(
                "The sector basemap is not georeferenced, so the DEM cannot be aligned "
                "to it. A NAV geometry grid is required."
            )

        size = int(output_px or basemap.width_px)
        gsd = basemap.width_m / size

        centres = (np.arange(size) + 0.5) * (basemap.width_px / size)
        grid_u, grid_v = np.meshgrid(centres, centres)
        pixel = basemap.origin_pixel + grid_u * basemap.decimation
        scan = basemap.origin_scan + grid_v * basemap.decimation
        longitude, latitude = basemap.grid.lonlat_at_many(pixel.ravel(), scan.ravel())

        if not self.covers(float(np.mean(longitude)), float(np.mean(latitude))):
            raise ValueError(
                f"The sector centre ({np.mean(latitude):.3f}, {np.mean(longitude):.3f}) "
                f"falls outside {self.product_id}, which covers latitudes "
                f"{self.spec.minimum_latitude} to {self.spec.maximum_latitude} and "
                f"longitudes {self.spec.westernmost_longitude} to "
                f"{self.spec.easternmost_longitude}. Sampling it would return the wrong "
                "ground while looking perfectly reasonable."
            )

        cols, rows = self.lonlat_to_pixel(longitude, latitude)
        col_min = int(np.floor(cols.min())) - 4
        row_min = int(np.floor(rows.min())) - 4
        col_max = int(np.ceil(cols.max())) + 4
        row_max = int(np.ceil(rows.max())) + 4

        patch = self.read_window(col_min, row_min, col_max - col_min, row_max - row_min)

        from .lola_dem import _bilinear

        elevation = _bilinear(
            patch, cols - max(col_min, 0), rows - max(row_min, 0)
        ).reshape(size, size)

        valid = np.isfinite(elevation)
        notes: list[str] = []
        if self.pixel_size_m > gsd * 1.5:
            notes.append(
                f"Elevation is posted at about {self.pixel_size_m:.0f} m at the equator "
                f"and resampled to {gsd:.2f} m to share the imagery's grid. The detail "
                "is not real at the finer spacing."
            )

        return SectorElevation(
            elevation_m=elevation.astype(np.float32),
            gsd_m=gsd,
            source_gsd_m=self.pixel_size_m,
            product=self.product_id,
            valid_fraction=float(valid.mean()),
            notes=notes,
        )
