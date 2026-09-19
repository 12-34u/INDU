"""
Real elevation for the sector, from the LOLA polar DEM.

Skips when no DEM is present under data/raw. The tests are mostly about the
DEM landing on the right piece of the Moon and being honest about its own
resolution, because a DEM that covers the wrong region still reads cleanly and
still produces a plausible-looking terrain surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunar_platform.core.data_manager import DataManager
from lunar_platform.observation.basemap import build_sector_basemap_from_source
from lunar_platform.terrain.lola_dem import LolaPolarDEM, SectorDEMProvider
from lunar_platform.terrain.slope import calculate_slope

SECTOR_SIZE_M = 2400.0


@pytest.fixture(scope="module")
def manager():
    return DataManager()


@pytest.fixture(scope="module")
def dem(manager):
    label = manager.get_raw_dem_label()
    if label is None:
        pytest.skip("No DEM with a PDS label under data/raw/dem")
    return LolaPolarDEM(label)


@pytest.fixture(scope="module")
def basemap(manager):
    source = manager.get_simulation_source()
    if source is None:
        pytest.skip("No Chandrayaan-2 bundle under data/raw")
    centre = manager.get_sector_centre_lonlat()
    return build_sector_basemap_from_source(
        source,
        centre_longitude=centre[0] if centre else None,
        centre_latitude=centre[1] if centre else None,
        size_m=SECTOR_SIZE_M,
    )


@pytest.fixture(scope="module")
def elevation(dem, basemap):
    return dem.sample_sector(basemap)


def test_dem_projection_comes_from_the_label(dem):
    """The projection is read, not assumed: a wrong sign flips the hemisphere."""
    assert dem.crs is not None
    assert dem.crs.is_projected
    assert dem.pixel_size_m > 0
    # The label states the DN scaling; hard-coding it would break on a
    # different LOLA release.
    assert dem.scale > 0


def test_dem_actually_covers_the_sector(elevation):
    """
    The failure this guards against is a DEM of the wrong region.

    SLDEM2015 tiles, for instance, span 60 south to 60 north and cannot cover
    this sector at about 71 south, but they read perfectly well and yield a
    full array of plausible elevations.
    """
    assert elevation.valid_fraction > 0.99
    assert np.isfinite(elevation.elevation_m).mean() > 0.99


def test_elevation_is_physically_plausible(elevation):
    finite = elevation.elevation_m[np.isfinite(elevation.elevation_m)]
    # Lunar topography relative to the 1737.4 km datum spans roughly -9 to +11 km.
    assert -9000.0 < finite.min() < 11000.0
    assert -9000.0 < finite.max() < 11000.0
    # A 2.4 km patch of highlands has relief, but not kilometres of it.
    relief = float(finite.max() - finite.min())
    assert 1.0 < relief < 2000.0


def test_elevation_shares_the_imagery_grid(elevation, basemap):
    """Frames must agree exactly, or the rover, route and image disagree."""
    assert elevation.elevation_m.shape == (basemap.height_px, basemap.width_px)
    assert elevation.gsd_m == pytest.approx(basemap.gsd_m, rel=1e-9)


def test_resolution_is_reported_honestly(elevation):
    """The DEM is far coarser than the imagery and must say so."""
    described = elevation.describe()
    assert described["is_synthetic"] is False
    assert described["source_gsd_m"] > described["gsd_m"]
    assert described["resample_factor"] > 1.0
    assert any("posted at" in note for note in described["notes"])


def test_slope_is_computed_from_real_relief(elevation):
    slope = calculate_slope(elevation.elevation_m, elevation.gsd_m)
    finite = slope[np.isfinite(slope)]
    assert finite.size
    # Real terrain is neither a plane nor a cliff everywhere.
    assert finite.mean() > 0.01
    assert finite.max() < 80.0


def test_provider_fills_gaps_rather_than_propagating_them(elevation):
    """
    A single gap would otherwise spread across the sector.

    Slope and roughness use finite differences, so one NaN contaminates its
    whole neighbourhood; the provider fills instead.
    """
    holed = np.array(elevation.elevation_m, dtype=np.float32)
    holed[10, 10] = np.nan
    from lunar_platform.terrain.lola_dem import SectorElevation

    provider = SectorDEMProvider(
        SectorElevation(
            elevation_m=holed,
            gsd_m=elevation.gsd_m,
            source_gsd_m=elevation.source_gsd_m,
            product=elevation.product,
            valid_fraction=0.99,
            notes=[],
        )
    )
    dem = provider.get_dem()
    assert np.isfinite(dem).all()
    assert np.isfinite(calculate_slope(dem, provider.get_gsd_m())).all()


def test_a_sector_off_the_dem_is_refused(dem, basemap):
    """
    Asking for ground the DEM does not cover must fail, not return zeros.

    This is exactly the SLDEM2015 situation, and silently returning a filled
    array would have hidden it.
    """
    import copy

    elsewhere = copy.copy(basemap)
    # Drive the sampling far into the northern hemisphere, which a south polar
    # DEM cannot cover.
    elsewhere.origin_scan = basemap.origin_scan
    elsewhere.grid = _ShiftedGrid(basemap.grid, latitude_offset=+150.0)

    with pytest.raises(ValueError):
        dem.sample_sector(elsewhere)


class _ShiftedGrid:
    """A NAV grid moved to another part of the Moon, for the negative test."""

    def __init__(self, grid, latitude_offset: float):
        self._grid = grid
        self._offset = latitude_offset

    def lonlat_at_many(self, pixel, scan):
        longitude, latitude = self._grid.lonlat_at_many(pixel, scan)
        return longitude, latitude + self._offset
