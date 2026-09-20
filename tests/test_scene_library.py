"""
The scene library: what the system knows about a view without reading it.

The library exists so that whether two images can be matched is answerable
before any feature is extracted. The properties it records - footprint, ground
scale, illumination - are exactly the ones that decided the outcome of the
cross-instrument registration this project could not make work.

Tests that need real products skip without them; the model and persistence
tests do not.
"""

from __future__ import annotations

import pytest

from lunar_platform.core.data_manager import DataManager
from lunar_platform.scene.models import (
    Footprint,
    Illumination,
    SceneLibrary,
    SceneView,
)


# -- illumination reasoning --------------------------------------------------


def test_shadow_ratio_is_what_makes_a_pair_hard():
    """
    The measured case: 2.2 degrees against 7.5 is a 3.4x difference in shadow
    length, which is why those two images do not look alike. The elevation
    difference alone would not say that - the same 5 degree gap higher up
    barely matters.
    """
    low = Illumination(sun_elevation_deg=2.2)
    high = Illumination(sun_elevation_deg=7.5)
    assert 3.3 < low.shadow_length_ratio(high) < 3.6

    steep_a = Illumination(sun_elevation_deg=40.0)
    steep_b = Illumination(sun_elevation_deg=45.0)
    assert steep_a.shadow_length_ratio(steep_b) < 1.3


def test_shadow_ratio_is_symmetric_and_needs_both_sides():
    a = Illumination(sun_elevation_deg=2.2)
    b = Illumination(sun_elevation_deg=7.5)
    assert a.shadow_length_ratio(b) == pytest.approx(b.shadow_length_ratio(a))
    assert a.shadow_length_ratio(Illumination()) is None


def test_azimuth_difference_wraps():
    a = Illumination(sun_azimuth_deg=350.0)
    b = Illumination(sun_azimuth_deg=10.0)
    assert a.azimuth_difference_deg(b) == pytest.approx(20.0)


# -- footprint reasoning -----------------------------------------------------


def test_bounds_overlap_is_zero_for_disjoint_ground():
    a = Footprint(corners=[(0, 0), (1, 0), (1, 1), (0, 1)], centre_longitude=0.5, centre_latitude=0.5)
    b = Footprint(corners=[(5, 5), (6, 5), (6, 6), (5, 6)], centre_longitude=5.5, centre_latitude=5.5)
    assert a.bounds_overlap(b) == 0.0


def test_separation_is_a_real_distance():
    a = Footprint(centre_longitude=0.0, centre_latitude=0.0)
    b = Footprint(centre_longitude=1.0, centre_latitude=0.0)
    # One degree at the lunar equator is about 30 km.
    assert 29000 < a.separation_m(b) < 31000


# -- persistence -------------------------------------------------------------


def test_library_round_trips_through_disk(tmp_path):
    view = SceneView(
        view_id="test:one",
        sector_id="sector_x",
        sensor="TEST",
        product_id="one",
        source_ref="/somewhere/one.tif",
        gsd_m=1.25,
        acquired_utc="2019-09-05T18:10:20Z",
        footprint=Footprint(
            corners=[(10.0, -70.0), (11.0, -70.0), (11.0, -71.0), (10.0, -71.0)],
            centre_longitude=10.5,
            centre_latitude=-70.5,
        ),
        illumination=Illumination(sun_elevation_deg=3.3, sun_azimuth_deg=80.0, source="spice"),
    )
    library = SceneLibrary(sector_id="sector_x", views=[view])
    library.save(tmp_path)

    back = SceneLibrary.load(tmp_path, "sector_x")
    assert len(back) == 1
    restored = back.views[0]
    assert restored.view_id == view.view_id
    assert restored.gsd_m == view.gsd_m
    assert restored.footprint.corners == view.footprint.corners
    assert restored.illumination.sun_elevation_deg == 3.3


def test_loading_a_missing_library_is_empty_not_an_error(tmp_path):
    assert len(SceneLibrary.load(tmp_path, "nothing_here")) == 0


def test_adding_the_same_view_replaces_rather_than_duplicates():
    library = SceneLibrary(sector_id="s")
    for gsd in (1.0, 2.0):
        library.add(
            SceneView(
                view_id="same", sector_id="s", sensor="T", product_id="p",
                source_ref="r", gsd_m=gsd,
            )
        )
    assert len(library) == 1
    assert library.by_id("same").gsd_m == 2.0


# -- built from the real archive ---------------------------------------------


@pytest.fixture(scope="module")
def built():
    from lunar_platform.scene.builder import build_library

    manager = DataManager()
    if manager.get_simulation_source() is None:
        pytest.skip("No products under data/raw")
    return build_library(manager)


def test_both_instruments_are_catalogued(built):
    assert len(built) >= 2
    assert {"CH2_OHRC", "LRO_NAC"} <= set(built.describe()["sensors"])


def test_every_view_is_placed_on_the_moon(built):
    for view in built.views:
        assert view.footprint.known, f"{view.view_id} has no footprint"
        assert -90 <= view.footprint.centre_latitude <= 90


def test_illumination_does_not_depend_on_build_order(built):
    """
    Both views must have illumination.

    The NAC camera model furnishes the SPICE kernels as a side effect, so an
    earlier version of the builder gave illumination only to whichever view
    happened to be built after it. The kernels are now loaded up front.
    """
    described = built.describe()
    assert described["n_with_illumination"] == described["n_views"]


def test_the_library_reproduces_the_measured_illumination_gap(built):
    """The pair's 3.4x shadow difference should fall out of the records."""
    ohrc = built.by_sensor("CH2_OHRC")[0]
    nac = built.by_sensor("LRO_NAC")[0]

    assert 6.5 < ohrc.illumination.sun_elevation_deg < 8.5
    assert 1.5 < nac.illumination.sun_elevation_deg < 3.0
    assert 3.0 < ohrc.illumination.shadow_length_ratio(nac.illumination) < 4.0


def test_ground_scales_are_measured_not_nominal(built):
    """OHRC's label nominal is wrong by 19%; the recorded value is measured."""
    ohrc = built.by_sensor("CH2_OHRC")[0]
    nac = built.by_sensor("LRO_NAC")[0]
    assert 2.5 < ohrc.gsd_m < 3.2
    assert 1.0 < nac.gsd_m < 1.6


def test_missing_ephemeris_is_recorded_not_invented(built):
    """
    Emission and phase need the camera's position. There is no Chandrayaan-2
    SPK here, so OHRC must report them as absent rather than plausible.
    """
    ohrc = built.by_sensor("CH2_OHRC")[0]
    assert ohrc.illumination.emission_deg is None
    assert ohrc.illumination.sun_elevation_deg is not None
