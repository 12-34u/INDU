"""
Sector basemap built from real Chandrayaan-2 OHRC imagery.

This is the ground the simulation runs on. A tile of the OHRC browse raster is
anchored at the sector's centre coordinate using the bundle's own NAV geometry
grid, so a position in the sector frame is a position on the Moon and can be
reported as longitude/latitude.

What is real here, and what is not:

* the imagery is the mission's own browse product, read in place from
  data/raw and never modified;
* the metre scale is measured from the NAV grid rather than taken from the
  label's nominal pixel_resolution, because the two disagree;
* the sun direction is derived from the label's sun azimuth rotated into image
  coordinates through the grid's local north, not assumed;
* the anchor is accurate to about half a grid node (~50 full-res pixels), as
  the grid is sampled rather than a camera model.

Sector frame convention, matching simulation.camera: x to the right, y
downward, metres from the tile's top-left corner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..io.ohrc_browse import (
    GeometryGrid,
    MOON_RADIUS_M,
    browse_decimation,
    load_browse_image,
    load_geometry_grid,
)

# Fallback only, measured from the strip by correlating brightness along
# candidate directions. Used when the label carries no sun azimuth.
FALLBACK_SUN_DIRECTION_DEG = 340.0

DEFAULT_SECTOR_SIZE_M = 2400.0


@dataclass
class SectorBasemap:
    """A georeferenced tile of real OHRC imagery in the sector frame."""

    image: np.ndarray
    gsd_m: float
    origin_pixel: float
    origin_scan: float
    decimation: float
    sun_direction_deg: float
    product_id: str
    source_ref: str
    grid: Optional[GeometryGrid] = None
    anchor_error_m: Optional[float] = None
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    @property
    def height_px(self) -> int:
        return int(self.image.shape[0])

    @property
    def width_px(self) -> int:
        return int(self.image.shape[1])

    @property
    def width_m(self) -> float:
        return self.width_px * self.gsd_m

    @property
    def height_m(self) -> float:
        return self.height_px * self.gsd_m

    def sector_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        return x_m / self.gsd_m, y_m / self.gsd_m

    def pixel_to_sector(self, u: float, v: float) -> tuple[float, float]:
        return u * self.gsd_m, v * self.gsd_m

    def contains_sector_point(self, x_m: float, y_m: float, margin_m: float = 0.0) -> bool:
        return (
            margin_m <= x_m <= self.width_m - margin_m
            and margin_m <= y_m <= self.height_m - margin_m
        )

    def lonlat_at(self, x_m: float, y_m: float) -> Optional[tuple[float, float]]:
        """
        Selenographic coordinates of a sector point, or None without a grid.

        Interpolated from the NAV lattice, so it inherits that lattice's
        accuracy and is not a camera-model solution.
        """
        if self.grid is None:
            return None
        u, v = self.sector_to_pixel(x_m, y_m)
        pixel = self.origin_pixel + u * self.decimation
        scan = self.origin_scan + v * self.decimation
        return self.grid.lonlat_at(pixel, scan)

    def describe(self) -> dict:
        """Provenance for the API and UI, stated rather than implied."""
        centre = self.lonlat_at(self.width_m / 2.0, self.height_m / 2.0)
        return {
            "product_id": self.product_id,
            "source_ref": self.source_ref,
            "is_synthetic": False,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width_m": round(self.width_m, 1),
            "height_m": round(self.height_m, 1),
            "gsd_m": round(self.gsd_m, 4),
            "browse_decimation": round(self.decimation, 3),
            "origin_pixel": round(self.origin_pixel, 1),
            "origin_scan": round(self.origin_scan, 1),
            "sun_direction_deg": round(self.sun_direction_deg, 2),
            "georeferenced": self.grid is not None,
            "centre_longitude": round(centre[0], 6) if centre else None,
            "centre_latitude": round(centre[1], 6) if centre else None,
            "anchor_error_m": round(self.anchor_error_m, 1)
            if self.anchor_error_m is not None
            else None,
            "notes": list(self.notes),
        }


def sun_direction_in_image(
    grid: GeometryGrid, pixel: float, scan: float, sun_azimuth_deg: float, step: float = 100.0
) -> float:
    """
    Rotate a map-frame sun azimuth into image coordinates.

    sun_azimuth_deg is clockwise from north, as PDS4 states it. The grid gives
    how ground east and north run across the image near a point; inverting that
    little Jacobian gives the image direction that points toward the sun. The
    result is the direction from a crater's shadow toward its lit rim.
    """
    lon0, lat0 = grid.lonlat_at(pixel, scan)
    lon_p, lat_p = grid.lonlat_at(pixel + step, scan)
    lon_s, lat_s = grid.lonlat_at(pixel, scan + step)

    metres_per_deg_lat = MOON_RADIUS_M * np.pi / 180.0
    metres_per_deg_lon = metres_per_deg_lat * np.cos(np.radians(lat0))

    east_per_u = (lon_p - lon0) * metres_per_deg_lon / step
    north_per_u = (lat_p - lat0) * metres_per_deg_lat / step
    east_per_v = (lon_s - lon0) * metres_per_deg_lon / step
    north_per_v = (lat_s - lat0) * metres_per_deg_lat / step

    jacobian = np.array([[east_per_u, east_per_v], [north_per_u, north_per_v]])
    if abs(float(np.linalg.det(jacobian))) < 1e-12:
        return FALLBACK_SUN_DIRECTION_DEG

    azimuth = np.radians(sun_azimuth_deg)
    du, dv = np.linalg.solve(jacobian, np.array([np.sin(azimuth), np.cos(azimuth)]))
    return float(np.degrees(np.arctan2(dv, du)) % 360.0)


def build_sector_basemap_from_source(
    source: dict,
    centre_longitude: Optional[float] = None,
    centre_latitude: Optional[float] = None,
    size_m: float = DEFAULT_SECTOR_SIZE_M,
) -> SectorBasemap:
    """
    Build a basemap from a DataManager simulation source descriptor.

    The descriptor says whether the imagery is a member of a bundle or a loose
    file; this keeps that distinction from leaking into every caller.
    """
    archive = source.get("archive") or source.get("browse_path")
    if archive is None:
        raise ValueError("Simulation source carries no browse imagery")

    geometry = source.get("geometry_member")
    geometry_archive = source.get("archive")
    if geometry is None and source.get("geometry_path"):
        geometry_archive = source["geometry_path"]

    return build_sector_basemap(
        archive=archive,
        browse_member=source.get("browse_member"),
        geometry_member=geometry,
        geometry_archive=geometry_archive,
        label=source.get("label"),
        centre_longitude=centre_longitude,
        centre_latitude=centre_latitude,
        size_m=size_m,
        product_id=source.get("product_id", ""),
    )


def build_sector_basemap(
    archive: Path,
    browse_member: Optional[str] = None,
    geometry_member: Optional[str] = None,
    label: Optional[dict] = None,
    geometry_archive: Optional[Path] = None,
    centre_longitude: Optional[float] = None,
    centre_latitude: Optional[float] = None,
    size_m: float = DEFAULT_SECTOR_SIZE_M,
    product_id: str = "",
) -> SectorBasemap:
    """
    Cut a sector-sized tile of real OHRC imagery, anchored on the Moon.

    The tile is centred on the requested coordinate when the NAV grid can place
    it and the coordinate falls inside the strip; otherwise it is centred on
    the strip and reported as not anchored. Requesting a tile larger than the
    strip yields the largest tile the strip can supply, which is stated in the
    returned notes rather than silently padded.
    """
    archive = Path(archive)
    notes: list[str] = []

    image = load_browse_image(archive, browse_member)
    decimation = browse_decimation(image.shape, label)

    grid: Optional[GeometryGrid] = None
    geometry_source = geometry_archive if geometry_archive is not None else archive
    if geometry_member or (geometry_archive is not None and geometry_archive != archive):
        try:
            grid = load_geometry_grid(Path(geometry_source), geometry_member)
        except Exception as exc:  # a malformed grid must not take the sim down
            notes.append(f"NAV geometry unavailable ({exc}); sector is not georeferenced.")

    # Metre scale: measured from the grid where possible, nominal otherwise.
    centre_pixel_full = (image.shape[1] / 2.0) * decimation
    centre_scan_full = (image.shape[0] / 2.0) * decimation
    if grid is not None:
        across, along = grid.ground_sampling_m(centre_pixel_full, centre_scan_full)
        full_gsd = float((across + along) / 2.0)
        nominal = label.get("pixel_resolution") if label else None
        if nominal and abs(full_gsd - float(nominal)) / float(nominal) > 0.05:
            notes.append(
                f"Metre scale measured from the NAV grid ({full_gsd:.3f} m/full px) differs "
                f"from the label's nominal pixel_resolution ({float(nominal):.3f}); the "
                "measured value is used."
            )
    else:
        full_gsd = float(label.get("pixel_resolution", 0.24)) if label else 0.24
        notes.append("No NAV grid; metre scale taken from the label's nominal resolution.")

    gsd_m = full_gsd * decimation

    tile_px = int(round(size_m / gsd_m))
    max_tile = min(image.shape[0], image.shape[1])
    if tile_px > max_tile:
        notes.append(
            f"Requested {size_m:.0f} m sector needs {tile_px} px but the strip is "
            f"{max_tile} px across; using the largest tile the strip supports."
        )
        tile_px = max_tile
    tile_px = max(16, tile_px)

    # Anchor the tile.
    anchor_error_m: Optional[float] = None
    if grid is not None and centre_longitude is not None and centre_latitude is not None:
        if grid.contains(centre_longitude, centre_latitude):
            pixel_c, scan_c = grid.nearest_image_point(centre_longitude, centre_latitude)
            lon_a, lat_a = grid.lonlat_at(pixel_c, scan_c)
            from ..io.ohrc_browse import great_circle_m

            anchor_error_m = float(
                great_circle_m(lat_a, lon_a, centre_latitude, centre_longitude)
            )
            u_c = pixel_c / decimation
            v_c = scan_c / decimation
        else:
            notes.append(
                f"Sector centre ({centre_latitude:.4f}, {centre_longitude:.4f}) falls outside "
                "this product's footprint; the tile is centred on the strip instead."
            )
            u_c, v_c = image.shape[1] / 2.0, image.shape[0] / 2.0
    else:
        u_c, v_c = image.shape[1] / 2.0, image.shape[0] / 2.0

    half = tile_px / 2.0
    u0 = int(round(min(max(u_c - half, 0), image.shape[1] - tile_px)))
    v0 = int(round(min(max(v_c - half, 0), image.shape[0] - tile_px)))
    tile = np.ascontiguousarray(image[v0 : v0 + tile_px, u0 : u0 + tile_px])

    origin_pixel = u0 * decimation
    origin_scan = v0 * decimation

    if grid is not None and label and label.get("sun_azimuth") is not None:
        sun_direction = sun_direction_in_image(
            grid,
            origin_pixel + (tile_px / 2.0) * decimation,
            origin_scan + (tile_px / 2.0) * decimation,
            float(label["sun_azimuth"]),
        )
    else:
        sun_direction = FALLBACK_SUN_DIRECTION_DEG
        notes.append(
            "Sun direction could not be derived from the label; using the value measured "
            "from the strip."
        )

    return SectorBasemap(
        image=tile,
        gsd_m=gsd_m,
        origin_pixel=origin_pixel,
        origin_scan=origin_scan,
        decimation=decimation,
        sun_direction_deg=sun_direction,
        product_id=product_id or archive.stem,
        source_ref=(
            f"/vsizip/{archive}/{browse_member}" if browse_member else str(archive)
        ),
        grid=grid,
        anchor_error_m=anchor_error_m,
        notes=notes,
    )
