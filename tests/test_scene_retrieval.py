"""
Choosing which views are worth matching against.

The point of retrieval here is not speed, it is avoiding futile work. This
project has one cross-instrument pair and it cannot be registered, because the
two images were taken at 2.2 and 7.5 degrees of sun elevation - a 3.4x
difference in shadow length. Retrieval exists so that, given a choice, that
pairing is not the one attempted.
"""

from __future__ import annotations

import pytest

from lunar_platform.core.data_manager import DataManager
from lunar_platform.scene.models import Footprint, Illumination, SceneLibrary, SceneView
from lunar_platform.scene.retrieval import (
    illumination_similarity,
    query_from_view,
    retrieve,
    score_view,
    RetrievalQuery,
)


def square(centre_lon: float, centre_lat: float, half: float = 0.5) -> Footprint:
    return Footprint(
        corners=[
            (centre_lon - half, centre_lat - half),
            (centre_lon + half, centre_lat - half),
            (centre_lon + half, centre_lat + half),
            (centre_lon - half, centre_lat + half),
        ],
        centre_longitude=centre_lon,
        centre_latitude=centre_lat,
    )


def view(view_id: str, footprint: Footprint, elevation: float, azimuth: float = 80.0, gsd: float = 1.0):
    return SceneView(
        view_id=view_id,
        sector_id="s",
        sensor="TEST",
        product_id=view_id,
        source_ref=f"/{view_id}.tif",
        gsd_m=gsd,
        footprint=footprint,
        illumination=Illumination(
            sun_elevation_deg=elevation, sun_azimuth_deg=azimuth, source="test"
        ),
    )


# -- the property the whole stage exists for --------------------------------


def test_prefers_a_like_lit_view_over_a_better_overlapping_one():
    """
    The acceptance criterion.

    One candidate shares more ground but was lit completely differently; the
    other shares less but is lit like the query. The second must win, because
    overlap you cannot match is worth nothing.
    """
    query_footprint = square(0.0, 0.0, 0.5)
    query = RetrievalQuery(
        footprint=query_footprint,
        illumination=Illumination(sun_elevation_deg=7.5, sun_azimuth_deg=80.0),
        gsd_m=1.0,
    )

    # Fully overlapping, but grazing sun against the query's higher sun.
    badly_lit = view("overlaps-but-dark", square(0.0, 0.0, 0.5), elevation=2.0)
    # Only partly overlapping, but lit almost identically.
    well_lit = view("partial-but-matched", square(0.4, 0.0, 0.5), elevation=7.2)

    library = SceneLibrary(sector_id="s", views=[badly_lit, well_lit])
    ranked = retrieve(library, query, k=2)

    assert [c.view.view_id for c in ranked][0] == "partial-but-matched"
    assert ranked[0].overlap < ranked[1].overlap  # it really did share less ground
    # and it says why the other was passed over
    assert any("Shadows differ" in r for r in ranked[1].reasons)


def test_ranking_explains_itself():
    query = RetrievalQuery(
        footprint=square(0.0, 0.0),
        illumination=Illumination(sun_elevation_deg=7.5, sun_azimuth_deg=80.0),
        gsd_m=1.0,
    )
    candidate = score_view(query, view("v", square(0.0, 0.0), elevation=2.0, gsd=4.0))
    joined = " ".join(candidate.reasons)
    assert "Shadows differ" in joined
    assert "Ground sampling differs" in joined


# -- overlap -----------------------------------------------------------------


def test_views_sharing_no_ground_are_dropped_not_ranked_low():
    query = RetrievalQuery(footprint=square(0.0, 0.0), illumination=Illumination(sun_elevation_deg=7.0))
    far_away = view("elsewhere", square(40.0, 40.0), elevation=7.0)
    library = SceneLibrary(sector_id="s", views=[far_away])
    assert retrieve(library, query, k=5) == []


def test_polygon_overlap_is_not_the_bounding_box():
    """
    Diagonal ribbons fill the same bounding box while sharing almost no ground.

    A box test cannot see this: each strip's box is the whole square, so by
    boxes they agree completely, and by area they cross only in the middle.
    Perpendicular strips, by contrast, have a SMALL box intersection, so they
    are not the case that misleads - the misleading case is a footprint whose
    box is much larger than the footprint.
    """
    rising = Footprint(
        corners=[(-2.0, -2.0), (-1.8, -2.0), (2.0, 1.8), (2.0, 2.0)],
        centre_longitude=0.0, centre_latitude=0.0,
    )
    falling = Footprint(
        corners=[(-2.0, 2.0), (-1.8, 2.0), (2.0, -1.8), (2.0, -2.0)],
        centre_longitude=0.0, centre_latitude=0.0,
    )

    # Both boxes are essentially the full square, so boxes claim near-total
    # agreement...
    assert rising.bounds_overlap(falling) > 0.9
    # ...while the ground genuinely shared is a sliver where they cross.
    assert rising.polygon_overlap(falling) < 0.15


def test_polygon_overlap_matches_an_analytic_case():
    """A cross of two thin strips: shared 0.2x0.2 of a 4x0.2 strip is 5%."""
    horizontal = Footprint(
        corners=[(-2, -0.1), (2, -0.1), (2, 0.1), (-2, 0.1)],
        centre_longitude=0.0, centre_latitude=0.0,
    )
    vertical = Footprint(
        corners=[(-0.1, -2), (0.1, -2), (0.1, 2), (-0.1, 2)],
        centre_longitude=0.0, centre_latitude=0.0,
    )
    assert horizontal.polygon_overlap(vertical) == pytest.approx(0.05, abs=0.01)


# -- illumination scoring ----------------------------------------------------


def test_identical_lighting_scores_best():
    same = Illumination(sun_elevation_deg=7.5, sun_azimuth_deg=80.0)
    score, ratio, azimuth = illumination_similarity(same, same)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert ratio == pytest.approx(1.0)
    assert azimuth == pytest.approx(0.0)


def test_the_real_pair_scores_poorly():
    ohrc = Illumination(sun_elevation_deg=7.5, sun_azimuth_deg=62.9)
    nac = Illumination(sun_elevation_deg=2.2, sun_azimuth_deg=79.7)
    score, ratio, _ = illumination_similarity(ohrc, nac)
    assert 3.0 < ratio < 4.0
    assert score < 0.35


def test_opposed_lighting_is_worse_than_merely_different():
    query = Illumination(sun_elevation_deg=10.0, sun_azimuth_deg=0.0)
    same_side = Illumination(sun_elevation_deg=10.0, sun_azimuth_deg=20.0)
    opposed = Illumination(sun_elevation_deg=10.0, sun_azimuth_deg=180.0)
    assert illumination_similarity(query, same_side)[0] > illumination_similarity(query, opposed)[0]


def test_unknown_illumination_is_neutral_not_disqualifying():
    known = Illumination(sun_elevation_deg=7.5, sun_azimuth_deg=80.0)
    score, ratio, azimuth = illumination_similarity(known, Illumination())
    assert score == pytest.approx(0.5)
    assert ratio is None and azimuth is None


def test_elevation_difference_alone_would_mislead():
    """Five degrees means very different things at 2 and at 40."""
    low_gap = illumination_similarity(
        Illumination(sun_elevation_deg=2.2, sun_azimuth_deg=0.0),
        Illumination(sun_elevation_deg=7.2, sun_azimuth_deg=0.0),
    )[1]
    high_gap = illumination_similarity(
        Illumination(sun_elevation_deg=40.0, sun_azimuth_deg=0.0),
        Illumination(sun_elevation_deg=45.0, sun_azimuth_deg=0.0),
    )[1]
    assert low_gap > 3.0
    assert high_gap < 1.3


# -- against the real archive ------------------------------------------------


@pytest.fixture(scope="module")
def built():
    from lunar_platform.scene.builder import build_library

    manager = DataManager()
    if manager.get_simulation_source() is None:
        pytest.skip("No products under data/raw")
    return build_library(manager)


def test_polygon_overlap_agrees_with_the_projected_measurement(built):
    """
    Cross-check against a completely different method.

    Pushing OHRC grid points through the SPICE camera model put 71% of them
    inside the NAC frame. The polygon intersection knows nothing about SPICE
    and should land near the same number.
    """
    ohrc = built.by_sensor("CH2_OHRC")[0]
    nac = built.by_sensor("LRO_NAC")[0]
    overlap = ohrc.footprint.polygon_overlap(nac.footprint)
    assert 0.6 < overlap < 0.85


def test_retrieval_flags_the_one_real_pair_as_poorly_lit(built):
    ohrc = built.by_sensor("CH2_OHRC")[0]
    ranked = retrieve(built, query_from_view(ohrc), k=5, exclude_view_ids={ohrc.view_id})

    assert ranked, "the NAC frame shares ground and should be returned"
    best = ranked[0]
    assert best.view.sensor == "LRO_NAC"
    # Returned, but honestly described as a hard pairing.
    assert best.illumination_score < 0.4
    assert any("Shadows differ" in r for r in best.reasons)
