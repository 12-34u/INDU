"""
Real lunar elevation for the sector, from the LOLA polar DEM.

The Chandrayaan-2 bundle carries no elevation, so before this the terrain under
the simulation was generated. This samples LOLA's south polar DEM instead, so
the slope and roughness layers - and therefore the routes planned on them -
describe the actual surface.

Two things are worth stating plainly rather than leaving for someone to
discover:

* The DEM is posted at 60 m, while the OHRC imagery is about 2.85 m. Elevation
  is therefore roughly twenty times coarser than the picture drawn over it. It
  resolves the shape of a crater a few hundred metres across; it does not
  resolve a boulder, a metre-scale step, or the rim of the small craters the
  detector finds in the imagery.
* SLDEM2015 does not cover this sector. That product spans 60 degrees south to
  60 degrees north, and the sector is near 71 degrees south, so the merged
  LOLA/SELENE DEM is not an option here regardless of which tile is fetched.

The DEM is sampled through the sector's own longitude/latitude, taken from the
OHRC NAV grid, so the elevation grid and the imagery share one frame by
construction rather than by a coincidence of extents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

# The DEM is defined on a sphere of this radius, as its own label states.
MOON_SPHERE_RADIUS_M = 1737400.0

# Margin, in DEM pixels, read around the sector so bilinear sampling at the
# edges has neighbours to interpolate between.
WINDOW_MARGIN_PX = 4


def _label_float(label: str, key: str, default: float) -> float:
    """Pull a numeric keyword out of a PDS label, or fall back."""
    import re

    match = re.search(rf"{key}\s*=\s*(-?[\d.]+)", label)
    return float(match.group(1)) if match else default


@dataclass
class SectorElevation:
    """Elevation sampled onto a sector grid, with its provenance."""

    elevation_m: np.ndarray
    gsd_m: float
    source_gsd_m: float
    product: str
    valid_fraction: float
    notes: list[str]

    def describe(self) -> dict:
        finite = self.elevation_m[np.isfinite(self.elevation_m)]
        return {
            "product": self.product,
            "is_synthetic": False,
            "width_px": int(self.elevation_m.shape[1]),
            "height_px": int(self.elevation_m.shape[0]),
            "gsd_m": round(self.gsd_m, 4),
            "source_gsd_m": round(self.source_gsd_m, 2),
            "resample_factor": round(self.source_gsd_m / self.gsd_m, 2),
            "valid_fraction": round(self.valid_fraction, 4),
            "min_elevation_m": round(float(finite.min()), 2) if finite.size else None,
            "max_elevation_m": round(float(finite.max()), 2) if finite.size else None,
            "relief_m": round(float(finite.max() - finite.min()), 2) if finite.size else None,
            "notes": list(self.notes),
        }


class LolaPolarDEM:
    """
    A LOLA polar-stereographic DEM read through its PDS label.

    GDAL parses the label's projection, so the polar stereographic conventions
    - which axis runs which way, where the pole sits in the array - come from
    the product rather than from assumptions made here.
    """

    def __init__(self, label_path: Path):
        self.label_path = Path(label_path)
        if not self.label_path.exists():
            raise FileNotFoundError(f"DEM label not found: {self.label_path}")

        with rasterio.open(self.label_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.transform = src.transform
            self.nodata = src.nodata
            # The label carries the DN scaling; taking it from the product
            # avoids hard-coding a factor that differs between LOLA releases.
            self.scale = float(src.scales[0]) if src.scales else 1.0
            self.pixel_size_m = float(abs(src.transform[0]))

        self._geographic = CRS.from_proj4(f"+proj=longlat +R={MOON_SPHERE_RADIUS_M} +no_defs")

        # Latitude band the product advertises, read from its label.
        label = self.label_path.read_text("utf-8", "replace")
        self.min_latitude = _label_float(label, "MINIMUM_LATITUDE", -90.0)
        self.max_latitude = _label_float(label, "MAXIMUM_LATITUDE", 90.0)

    @property
    def product_id(self) -> str:
        return self.label_path.stem

    def covers(self, longitude: float, latitude: float) -> bool:
        """
        Whether a point falls inside this polar tile.

        Tested by projecting the point and checking it lands on the array,
        rather than by comparing latitudes: a polar stereographic tile is a
        disc, so its corners are outside the latitude band it advertises.
        """
        try:
            cols, rows = self.lonlat_to_pixel(
                np.array([longitude], float), np.array([latitude], float)
            )
        except Exception:
            return False
        return bool(
            0 <= cols[0] < self.width
            and 0 <= rows[0] < self.height
            # The advertised latitude band still bounds it; the projection
            # happily places the whole sphere somewhere on the plane.
            and self.min_latitude <= latitude <= self.max_latitude
        )

    def lonlat_to_pixel(
        self, longitude: np.ndarray, latitude: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map longitude/latitude arrays to fractional DEM column/row."""
        xs, ys = warp_transform(
            self._geographic,
            self.crs,
            np.asarray(longitude, float).ravel().tolist(),
            np.asarray(latitude, float).ravel().tolist(),
        )
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        inverse = ~self.transform
        cols = inverse.a * xs + inverse.b * ys + inverse.c
        rows = inverse.d * xs + inverse.e * ys + inverse.f
        return cols, rows

    def sample_sector(self, basemap, output_px: Optional[int] = None) -> SectorElevation:
        """
        Sample the DEM onto a sector basemap's grid.

        Every output cell is placed by its own longitude and latitude, taken
        from the OHRC NAV grid, so the elevation lines up with the imagery
        rather than merely covering the same ground.
        """
        if basemap.grid is None:
            raise ValueError(
                "The sector basemap is not georeferenced, so the DEM cannot be "
                "aligned to it. A NAV geometry grid is required."
            )

        notes: list[str] = []
        size = int(output_px or basemap.width_px)
        gsd = basemap.width_m / size

        # Sector cell centres -> full-resolution OHRC pixel/scan -> lon/lat.
        centres = (np.arange(size) + 0.5) * (basemap.width_px / size)
        grid_u, grid_v = np.meshgrid(centres, centres)
        pixel = basemap.origin_pixel + grid_u * basemap.decimation
        scan = basemap.origin_scan + grid_v * basemap.decimation
        longitude, latitude = basemap.grid.lonlat_at_many(pixel.ravel(), scan.ravel())

        cols, rows = self.lonlat_to_pixel(longitude, latitude)

        col_min = int(np.floor(cols.min())) - WINDOW_MARGIN_PX
        col_max = int(np.ceil(cols.max())) + WINDOW_MARGIN_PX
        row_min = int(np.floor(rows.min())) - WINDOW_MARGIN_PX
        row_max = int(np.ceil(rows.max())) + WINDOW_MARGIN_PX

        if col_max < 0 or row_max < 0 or col_min >= self.width or row_min >= self.height:
            raise ValueError(
                f"The sector falls outside {self.product_id}. This DEM covers a "
                "different region of the Moon."
            )

        col_min = max(0, col_min)
        row_min = max(0, row_min)
        col_max = min(self.width, col_max)
        row_max = min(self.height, row_max)

        # One small window, not the whole 1.9 GB product.
        with rasterio.open(self.label_path) as src:
            window = rasterio.windows.Window(
                col_min, row_min, col_max - col_min, row_max - row_min
            )
            patch = src.read(1, window=window).astype(np.float64)

        if self.nodata is not None:
            patch[patch == self.nodata] = np.nan
        patch *= self.scale

        local_cols = cols - col_min
        local_rows = rows - row_min
        elevation = _bilinear(patch, local_cols, local_rows).reshape(size, size)

        valid = np.isfinite(elevation)
        valid_fraction = float(valid.mean())
        if valid_fraction < 1.0:
            notes.append(
                f"{(1 - valid_fraction) * 100:.1f}% of the sector has no DEM coverage; "
                "those cells are reported as gaps rather than filled."
            )

        if self.pixel_size_m > gsd * 1.5:
            notes.append(
                f"Elevation is posted at {self.pixel_size_m:.0f} m and resampled to "
                f"{gsd:.2f} m to share the imagery's grid. The detail is not real at "
                "the finer spacing; slope and roughness describe the "
                f"{self.pixel_size_m:.0f} m scale."
            )

        return SectorElevation(
            elevation_m=elevation.astype(np.float32),
            gsd_m=gsd,
            source_gsd_m=self.pixel_size_m,
            product=self.product_id,
            valid_fraction=valid_fraction,
            notes=notes,
        )


def _bilinear(array: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Bilinear sample, returning NaN outside the array or next to a gap."""
    height, width = array.shape

    col0 = np.floor(cols).astype(int)
    row0 = np.floor(rows).astype(int)
    inside = (col0 >= 0) & (row0 >= 0) & (col0 < width - 1) & (row0 < height - 1)

    out = np.full(cols.shape, np.nan, dtype=np.float64)
    if not inside.any():
        return out

    c0 = col0[inside]
    r0 = row0[inside]
    tx = cols[inside] - c0
    ty = rows[inside] - r0

    out[inside] = (
        array[r0, c0] * (1 - tx) * (1 - ty)
        + array[r0, c0 + 1] * tx * (1 - ty)
        + array[r0 + 1, c0] * (1 - tx) * ty
        + array[r0 + 1, c0 + 1] * tx * ty
    )
    return out


class SectorDEMProvider:
    """DEMProvider over a real DEM sampled into the sector frame."""

    def __init__(self, elevation: SectorElevation):
        self._elevation = elevation

    def get_dem(self) -> np.ndarray:
        dem = self._elevation.elevation_m
        if not np.isfinite(dem).all():
            # Slope and roughness use finite differences, which propagate a
            # single NaN across the neighbourhood. Fill gaps with the mean of
            # what is present so a small hole cannot blank the sector.
            finite = dem[np.isfinite(dem)]
            fill = float(finite.mean()) if finite.size else 0.0
            dem = np.where(np.isfinite(dem), dem, fill)
        return dem.astype(np.float32)

    def get_gsd_m(self) -> float:
        return float(self._elevation.gsd_m)
