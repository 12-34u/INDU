"""
The SLDEM2015 tile, and choosing a DEM by coverage rather than by name.

SLDEM2015 cannot cover sector_001 - that product family spans 60S to 60N and
the sector is near 71S - so these tests do two separate things: prove the tile
reads correctly over ground it DOES cover, and prove the platform refuses to
use it over ground it does not.

The second is the one that matters. A DEM of the wrong region is not an
obvious failure: it reads cleanly, fills every cell, and yields terrain that
looks entirely reasonable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lunar_platform.core.data_manager import DataManager
from lunar_platform.terrain.dem_registry import discover_dems, select_dem

SECTOR_CENTRE = (22.784, -70.881)  # longitude, latitude
INSIDE_SLDEM = (300.0, 30.0)


@pytest.fixture(scope="module")
def candidates():
    found = discover_dems(DataManager().raw_dir / "dem")
    if not found:
        pytest.skip("No DEMs under data/raw/dem")
    return found


@pytest.fixture(scope="module")
def sldem(candidates):
    for candidate in candidates:
        if candidate.kind == "equirectangular" and "SLDEM" in candidate.product_id.upper():
            return candidate.dem
    pytest.skip("No SLDEM2015 tile present")


# -- the tile itself ---------------------------------------------------------


def test_sidecar_matches_the_raster(sldem):
    """Geometry is recorded, not guessed, and is checked against the file."""
    assert sldem.spec.expected_bytes == sldem.raster_path.stat().st_size
    assert sldem.spec.map_resolution_ppd > 0
    # 256 px/degree is about 118 m at the equator.
    assert 100 < sldem.pixel_size_m < 140


def test_values_are_metres_after_scaling(sldem):
    """
    The tile stores kilometres; the reader must return metres.

    Getting this wrong is silent: relief comes out a thousand times too small
    and the terrain reads as a billiard table rather than as an error.
    """
    cols, rows = sldem.lonlat_to_pixel(
        np.array([INSIDE_SLDEM[0]]), np.array([INSIDE_SLDEM[1]])
    )
    patch = sldem.read_window(int(cols[0]), int(rows[0]), 64, 64)

    # Lunar topography relative to the 1737.4 km datum, in metres.
    assert -9000.0 < patch.min() < 11000.0
    assert -9000.0 < patch.max() < 11000.0
    # A patch several km across has real relief - tens of metres at least.
    assert (patch.max() - patch.min()) > 5.0


def test_coverage_is_honest_about_its_own_bounds(sldem):
    assert sldem.covers(*INSIDE_SLDEM)
    # The sector this project works on is in the wrong hemisphere entirely.
    assert not sldem.covers(*SECTOR_CENTRE)


def test_sampling_outside_the_tile_is_refused(sldem):
    """Rather than returning the wrong ground, which would look fine."""

    class FakeGrid:
        def lonlat_at_many(self, pixel, scan):
            n = len(np.asarray(pixel))
            return (
                np.full(n, SECTOR_CENTRE[0]),
                np.full(n, SECTOR_CENTRE[1]),
            )

    class FakeBasemap:
        grid = FakeGrid()
        width_px = height_px = 32
        width_m = height_m = 2400.0
        origin_pixel = origin_scan = 0.0
        decimation = 10.0

    with pytest.raises(ValueError, match="falls outside"):
        sldem.sample_sector(FakeBasemap())


def test_missing_sidecar_is_refused_not_guessed(tmp_path):
    """A bare raster's georeferencing has to be stated somewhere."""
    from lunar_platform.terrain.equirect_dem import EquirectangularDEM

    raster = tmp_path / "SOMETHING_256_0N_60N_240_360_FLOAT.IMG"
    raster.write_bytes(b"\x00" * 64)
    with pytest.raises(FileNotFoundError, match="sidecar"):
        EquirectangularDEM(raster)


def test_sidecar_disagreeing_with_the_raster_is_refused(tmp_path):
    from lunar_platform.terrain.equirect_dem import EquirectangularDEM

    raster = tmp_path / "tile.IMG"
    raster.write_bytes(b"\x00" * 64)
    (tmp_path / "tile.json").write_text(
        json.dumps(
            {
                "lines": 100,
                "line_samples": 100,
                "minimum_latitude": 0,
                "maximum_latitude": 10,
                "westernmost_longitude": 0,
                "easternmost_longitude": 10,
                "map_resolution_ppd": 10,
                "sample_bits": 32,
            }
        )
    )
    with pytest.raises(ValueError, match="bytes but its sidecar describes"):
        EquirectangularDEM(raster)


# -- selection ---------------------------------------------------------------


def test_selection_is_by_coverage_not_by_name(candidates):
    chosen = select_dem(candidates, *SECTOR_CENTRE)
    if chosen is None:
        pytest.skip("No DEM covers the sector")
    assert chosen.covers(*SECTOR_CENTRE)
    assert "SLDEM" not in chosen.product_id.upper()


def test_sldem_is_chosen_where_it_does_cover(candidates):
    chosen = select_dem(candidates, *INSIDE_SLDEM)
    if chosen is None:
        pytest.skip("No DEM covers that point")
    assert "SLDEM" in chosen.product_id.upper()


def test_no_dem_anywhere_returns_none(candidates):
    """The far side at a latitude neither product reaches."""
    assert select_dem(candidates, 150.0, 40.0) is None
