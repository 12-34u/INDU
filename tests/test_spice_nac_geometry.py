"""
SPICE geolocation of the LROC NAC reference image.

The NAC label carries spacecraft clock counts and an orbit number and no
geographic metadata, so everything asserted here is computed from the mission
kernels. Tests skip when the kernels or spiceypy are absent.

The round-trip test is the load-bearing one. A pushbroom model can be wrong in
ways that still produce a plausible-looking footprint - swapping the along-track
and cross-track axes lays the swath along the orbit instead of across it and
still yields a region of roughly the right size in roughly the right place - and
only forcing forward and inverse to agree catches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunar_platform.core.data_manager import DataManager

spiceypy = pytest.importorskip("spiceypy", reason="spiceypy is not installed")

from lunar_platform.planetary.spice_adapter import (  # noqa: E402
    LUNAR_FRAME,
    NacCameraModel,
    SpiceKernelSet,
    parse_nac_label,
    sector_coverage,
)


@pytest.fixture(scope="module")
def manager():
    return DataManager()


@pytest.fixture(scope="module")
def kernels(manager):
    discovered = manager.get_spice_kernels()
    if discovered is None or not discovered.is_complete:
        pytest.skip("No complete SPICE kernel set under data/raw/spice")
    return discovered


@pytest.fixture(scope="module")
def nac(manager):
    reference = manager.get_nac_reference()
    if reference is None:
        pytest.skip("No LROC NAC product with a label under data/raw")
    return reference


@pytest.fixture(scope="module")
def camera(nac, kernels):
    observation = parse_nac_label(nac["label_path"].read_text("utf-8", "replace"), nac["product_id"])
    return NacCameraModel(observation, kernels)


# -- kernel discovery --------------------------------------------------------


def test_kernel_set_reports_what_it_is_missing(tmp_path):
    """An incomplete set must name the gap rather than look merely empty."""
    (tmp_path / "naif0012.tls").write_text("KPL/LSK")
    partial = SpiceKernelSet.discover(tmp_path)

    assert partial.paths
    assert not partial.is_complete
    assert any("ephemeris" in item for item in partial.missing())


def test_kernels_load_in_dependency_order(kernels):
    """Text kernels that reference others must come after them."""
    suffixes = [path.suffix.lower() for path in kernels.paths]
    assert suffixes.index(".tls") < suffixes.index(".bc")
    assert kernels.is_complete
    assert kernels.missing() == []


# -- label parsing -----------------------------------------------------------


def test_label_yields_the_pushbroom_timing(nac, kernels):
    kernels.load()  # str2et below needs the leapseconds kernel furnished
    observation = parse_nac_label(
        nac["label_path"].read_text("utf-8", "replace"), nac["product_id"]
    )
    assert observation.lines > 0 and observation.samples > 0
    assert observation.line_exposure_s > 0
    # A summed image has fewer samples than the detector has pixels; missing
    # this halves the field of view.
    assert observation.crosstrack_summing >= 1
    assert observation.instrument_frame in ("LRO_LROCNACL", "LRO_LROCNACR")

    # The label's own start and stop times must agree with lines x exposure.
    import spiceypy as sp

    stated = sp.str2et(observation.stop_time) - sp.str2et(observation.start_time)
    assert observation.duration_s == pytest.approx(stated, rel=0.02)


# -- geometry ----------------------------------------------------------------


def test_every_corner_of_the_image_lands_on_the_moon(camera):
    footprint = camera.footprint()
    assert all(corner is not None for corner in footprint["corners"].values())
    assert footprint["frame"] == LUNAR_FRAME
    # This product images the southern highlands.
    assert -90.0 < footprint["min_latitude"] < footprint["max_latitude"] < 0.0


def test_forward_and_inverse_agree(camera):
    """The decisive check: a pixel's ground point must map back to that pixel."""
    observation = camera.observation
    errors = []
    for line in (10, observation.lines // 4, observation.lines // 2, observation.lines - 100):
        for sample in (0, observation.samples // 2, observation.samples - 1):
            ground = camera.ground_point(line, sample)
            assert ground is not None
            pixel = camera.image_point(*ground)
            assert pixel is not None, f"({line}, {sample}) did not invert"
            errors.append(np.hypot(pixel[0] - line, pixel[1] - sample))

    assert max(errors) < 0.5


def test_swath_width_matches_the_detector(camera):
    """
    Geometry cross-check independent of the round trip.

    The distance across the image on the ground must equal the sample count
    times the cross-track sampling. Swapping the detector axes breaks this even
    though the footprint still looks reasonable.
    """
    from lunar_platform.io.ohrc_browse import great_circle_m

    footprint = camera.footprint()
    left = footprint["corners"]["upper_left"]
    right = footprint["corners"]["upper_right"]
    measured = great_circle_m(
        left["latitude"], left["longitude"], right["latitude"], right["longitude"]
    )
    expected = camera.observation.samples * camera.ground_sampling_m()
    assert measured == pytest.approx(expected, rel=0.02)


def test_along_track_is_perpendicular_to_the_detector_row(camera):
    """
    Advancing a line must move the footprint across the swath, not along it.

    This is the axis confusion stated in the module docstring, asserted
    directly: the along-track step and the cross-track step have to be roughly
    perpendicular on the ground.
    """
    middle_line = camera.observation.lines // 2
    middle_sample = camera.observation.samples // 2

    origin = np.array(camera.ground_point(middle_line, middle_sample))
    along = np.array(camera.ground_point(middle_line + 500, middle_sample)) - origin
    across = np.array(camera.ground_point(middle_line, middle_sample + 500)) - origin

    # Work in a local metric frame so longitude convergence does not skew the angle.
    scale = np.array([np.cos(np.radians(origin[1])), 1.0])
    along = along * scale
    across = across * scale

    cosine = float(
        along @ across / (np.linalg.norm(along) * np.linalg.norm(across))
    )
    assert abs(cosine) < 0.2, f"axes are not perpendicular (cos={cosine:.3f})"


def test_a_point_off_the_swath_has_no_pixel(camera):
    """The swath is kilometres wide; the far side of the Moon is not in it."""
    footprint = camera.footprint()
    centre = footprint["centre"]
    # Half a degree of longitude here is far outside a ~3 km swath.
    assert camera.image_point(centre["longitude"] + 0.5, centre["latitude"]) is None
    assert camera.image_point(centre["longitude"], centre["latitude"] + 5.0) is None


def test_ground_sampling_is_the_published_scale(camera):
    """NAC is a 0.5 m/px camera; a 2x summed image is about 1 m at nadir."""
    gsd = camera.ground_sampling_m()
    assert 0.4 * camera.observation.crosstrack_summing < gsd < 1.2 * camera.observation.crosstrack_summing


# -- the question the README could not answer --------------------------------


def test_overlap_with_the_ohrc_sector_is_decided(camera):
    """
    Whether OHRC and NAC share ground is now a computed answer.

    data/raw/README.md recorded that this could not be established without
    SPICE. It can now, and the test asserts the computation runs and commits to
    an answer rather than asserting which answer it is.
    """
    from lunar_platform.observation.basemap import build_sector_basemap_from_source

    manager = DataManager()
    source = manager.get_simulation_source()
    if source is None:
        pytest.skip("No OHRC bundle under data/raw")

    centre = manager.get_sector_centre_lonlat()
    basemap = build_sector_basemap_from_source(
        source, centre[0] if centre else None, centre[1] if centre else None, size_m=2400.0
    )

    coverage = sector_coverage(camera, basemap, grid=5)
    assert coverage["points_tested"] > 0
    assert isinstance(coverage["overlaps"], bool)
    assert 0.0 <= coverage["coverage_fraction"] <= 1.0
    if coverage["overlaps"]:
        line_low, line_high = coverage["nac_line_range"]
        assert 0 <= line_low <= line_high <= camera.observation.lines
