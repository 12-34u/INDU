"""
The geometric prior and the radiometry it depends on.

These cover the three faults that kept the cross-instrument case at zero
candidate matches: a pair never brought to a common ground scale, a reference
never restricted to the overlap, and a nodata sentinel that destroyed the
reference's dynamic range while passing every validity check.

The nodata test is the important one. That defect was silent - the image had
the right shape, the right dtype and a healthy-looking valid-data fraction, and
was a blank rectangle by the time it reached the matcher.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lunar_platform.preprocessing.normalization import to_uint8, valid_data_fraction
from lunar_platform.registration.ground_scale import (
    GroundScale,
    plan_common_scale,
    resolve_ground_scale,
)
from lunar_platform.registration.overlap import compute_overlap_window
from lunar_platform.registration.pipeline import load_image


def write_raster(path, data, nodata=None, pixel_size=None):
    """A small GeoTIFF, optionally georeferenced and with a nodata value."""
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype.name,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    if pixel_size is not None:
        profile["crs"] = "+proj=stere +lat_0=-90 +R=1737400 +units=m +no_defs"
        profile["transform"] = from_origin(0, 0, pixel_size, pixel_size)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return str(path)


# -- the radiometry defect ---------------------------------------------------


def test_declared_nodata_does_not_destroy_the_stretch(tmp_path):
    """
    The LROC NAC failure, reproduced in miniature.

    Real surface values occupy a narrow range; a small proportion of the frame
    carries a large negative nodata sentinel. Left in, that sentinel is below
    the 1st percentile, the percentile stretch spans it, and everything real
    maps to the top of the 8-bit range.
    """
    data = np.random.default_rng(0).integers(20, 900, size=(200, 200)).astype(np.int16)
    data[:20, :] = -32768  # 10% nodata, as NAC carries about 1%

    path = write_raster(tmp_path / "nodata.tif", data, nodata=-32768)
    loaded, _ = load_image(path, max_dimension=256)

    # Marked as missing, not read as a very dark surface.
    assert np.isnan(loaded).any()
    assert np.isfinite(loaded).mean() == pytest.approx(0.9, abs=0.02)

    stretched = to_uint8(loaded)
    real = stretched[np.isfinite(loaded)]
    # The real surface must span the range, not pile up against white.
    assert real.mean() < 200
    assert real.std() > 25


def test_nodata_left_in_would_have_been_silent(tmp_path):
    """Shows why the defect survived review: every cheap check still passes."""
    data = np.random.default_rng(1).integers(20, 900, size=(200, 200)).astype(np.int16)
    data[:20, :] = -32768

    # The array as it was read before the fix: sentinel still present.
    naive = to_uint8(data.astype(np.float32))
    assert naive.shape == data.shape  # right shape
    assert naive.dtype == np.uint8  # right dtype
    # ...and almost entirely white, which is the actual failure.
    assert naive.mean() > 200


def test_valid_fraction_counts_nodata_as_missing():
    data = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    assert valid_data_fraction(data) == pytest.approx(0.75)


# -- ground scale ------------------------------------------------------------


def test_scale_comes_from_georeferencing_when_present(tmp_path):
    path = write_raster(
        tmp_path / "geo.tif", np.ones((64, 64), np.uint8), pixel_size=7.5
    )
    scale = resolve_ground_scale(path)
    assert scale.known
    assert scale.gsd_m == pytest.approx(7.5)
    assert scale.method == "georeference"


def test_explicit_scale_overrides_everything(tmp_path):
    path = write_raster(tmp_path / "geo.tif", np.ones((64, 64), np.uint8), pixel_size=7.5)
    scale = resolve_ground_scale(path, explicit_gsd_m=2.0)
    assert scale.gsd_m == pytest.approx(2.0)
    assert scale.method == "explicit"


def test_unknown_scale_is_reported_not_guessed(tmp_path):
    path = write_raster(tmp_path / "plain.tif", np.ones((64, 64), np.uint8))
    scale = resolve_ground_scale(path)
    assert not scale.known
    assert scale.method == "unknown"


def test_common_scale_equalises_a_mismatched_pair():
    """The whole point: after planning, both images are at one scale."""
    source = GroundScale(2.85, "nav_grid")
    reference = GroundScale(1.30, "spice")
    plan = plan_common_scale(
        "s", "r", (1000, 1000), (2000, 2000), 4000, source, reference
    )

    assert plan.equalised
    # The coarser of the two: the finer image is reduced, never magnified.
    assert plan.working_gsd_m == pytest.approx(2.85)
    # Both cover the same ground per output pixel.
    source_ground = 1000 * source.gsd_m / plan.source_out_shape[0]
    reference_ground = 2000 * reference.gsd_m / plan.reference_out_shape[0]
    assert source_ground == pytest.approx(reference_ground, rel=0.01)


def test_common_scale_respects_the_pixel_budget():
    """Budget pressure coarsens the shared scale; it must not split the pair."""
    plan = plan_common_scale(
        "s", "r", (40000, 4000), (60000, 4000), 512,
        GroundScale(1.0, "explicit"), GroundScale(1.0, "explicit"),
    )
    assert plan.equalised
    assert plan.source_out_shape[0] * plan.source_out_shape[1] <= 512**2
    assert plan.reference_out_shape[0] * plan.reference_out_shape[1] <= 512**2


def test_unknown_scale_falls_back_rather_than_inventing_one():
    plan = plan_common_scale(
        "s", "r", (100, 100), (100, 100), 1000,
        GroundScale(None, "unknown"), GroundScale(1.0, "explicit"),
    )
    assert not plan.equalised
    assert plan.working_gsd_m is None
    assert plan.source_out_shape is None
    assert "does not bring the pair to a common scale" in plan.note


# -- overlap -----------------------------------------------------------------


def test_overlap_needs_geometry_on_both_sides(tmp_path):
    """Without mission geometry there is no prior, and that is reported."""
    source = write_raster(tmp_path / "a.tif", np.ones((64, 64), np.uint8))
    reference = write_raster(tmp_path / "b.tif", np.ones((64, 64), np.uint8))
    assert compute_overlap_window(source, reference, (64, 64)) is None
