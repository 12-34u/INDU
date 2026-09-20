"""
Placing an image by agreement between views.

The consensus tests are the important ones and need no data: they are about
what the system is entitled to claim. A single registration's residuals say
how well a transform explains its own matches, not whether those matches were
right, so a lone view must not be reported with the confidence of several.

The end-to-end test uses a crop of a real product whose position is known
exactly from the bundle's own NAV grid, which makes the recovered coordinate
checkable rather than merely plausible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_platform.localization.multi_view import (
    ConsensusFix,
    ViewFix,
    consensus_position,
    great_circle_m,
    _mean_lonlat,
)
from lunar_platform.registration.pipeline import FrameMapping


def fix(name: str, lon: float, lat: float, inliers: int = 50) -> ViewFix:
    return ViewFix(
        view_id=name, sensor="TEST", succeeded=True,
        longitude=lon, latitude=lat, n_inliers=inliers, inlier_ratio=0.8,
    )


# -- what the system is entitled to claim ------------------------------------


def test_one_view_gives_a_position_but_no_corroboration():
    result = consensus_position([fix("a", 22.78, -70.88)])
    assert result.succeeded
    assert result.cross_checked is False
    assert result.n_agreeing == 1
    assert result.confidence <= 0.3
    assert "nothing to check it against" in result.note


def test_agreeing_views_raise_confidence_above_a_lone_one():
    lone = consensus_position([fix("a", 22.78, -70.88)])
    several = consensus_position(
        [fix("a", 22.78, -70.88), fix("b", 22.7801, -70.8801), fix("c", 22.7799, -70.8799)]
    )
    assert several.cross_checked
    assert several.confidence > lone.confidence
    assert several.n_agreeing == 3


def test_an_outlier_is_excluded_from_the_answer_but_reported():
    """A wrong registration must not drag the position toward itself."""
    result = consensus_position(
        [
            fix("a", 22.78, -70.88),
            fix("b", 22.7801, -70.8801),
            fix("c", 22.7799, -70.8799),
            fix("wrong", 23.9, -70.10),
        ]
    )
    assert result.n_registered == 4
    assert result.n_agreeing == 3
    # The answer sits with the cluster, not between it and the outlier.
    assert great_circle_m(result.longitude, result.latitude, 22.78, -70.88) < 100
    # ...and the dissenter is still in the record.
    dissent = [c for c in result.contributions if c.view_id == "wrong"][0]
    assert dissent.agreed is False
    assert dissent.offset_from_consensus_m > 1000
    assert "disagreed" in result.note


def test_no_position_when_nothing_registered():
    result = consensus_position(
        [ViewFix("a", "T", False, reason="too few inliers"),
         ViewFix("b", "T", False, reason="no overlap")]
    )
    assert not result.succeeded
    assert result.confidence == 0.0
    assert "too few inliers" in result.note


def test_tighter_agreement_scores_higher_than_loose():
    tight = consensus_position([fix("a", 22.78, -70.88), fix("b", 22.78005, -70.88005)])
    loose = consensus_position([fix("a", 22.78, -70.88), fix("b", 22.7840, -70.8820)])
    assert tight.spread_m < loose.spread_m
    assert tight.confidence > loose.confidence


def test_a_split_vote_keeps_the_larger_group():
    result = consensus_position(
        [fix("a", 22.78, -70.88), fix("b", 22.7801, -70.88), fix("c", 30.0, -60.0)]
    )
    assert result.n_agreeing == 2
    assert great_circle_m(result.longitude, result.latitude, 22.78, -70.88) < 100


# -- averaging positions -----------------------------------------------------


def test_positions_are_averaged_as_directions_not_numbers():
    """
    Arithmetic mean of longitudes is wrong across the meridian.

    Averaging 359 and 1 degrees numerically gives 180 - the opposite side of
    the Moon. Averaging them as directions gives 0.
    """
    longitude, latitude = _mean_lonlat([(359.0, 0.0), (1.0, 0.0)])
    assert abs(((longitude + 180) % 360) - 180) < 0.01
    assert abs(latitude) < 0.01


def test_averaging_near_the_pole_is_stable():
    """The sector this project works on is at 71 degrees south."""
    points = [(22.78, -70.88), (22.79, -70.881), (22.77, -70.879)]
    longitude, latitude = _mean_lonlat(points)
    for lon, lat in points:
        assert great_circle_m(longitude, latitude, lon, lat) < 1000


# -- working pixels back to the product's own pixels -------------------------


def test_frame_mapping_undoes_window_and_decimation():
    frame = FrameMapping(window=(100, 2000, 500, 800), decimation=0.5)
    assert frame.to_native(10, 20) == (120.0, 2040.0)


def test_frame_mapping_is_identity_without_either():
    assert FrameMapping().to_native(10, 20) == (10.0, 20.0)


# -- end to end, against a position known from the mission's own geometry -----


@pytest.fixture(scope="module")
def archive():
    from lunar_platform.core.data_manager import DataManager

    manager = DataManager()
    if manager.get_simulation_source() is None:
        pytest.skip("No products under data/raw")
    return manager


def test_a_crop_of_known_position_is_recovered(archive, tmp_path):
    """
    The whole chain, with truth available.

    A window of the OHRC browse raster is cut out and handed back to the system
    as though it were a new image. Its true centre comes from the bundle's NAV
    grid, so the recovered coordinate can be checked rather than merely
    inspected. This exercises retrieval, registration, the working-to-native
    coordinate recovery, per-view geolocation and consensus together.
    """
    import cv2
    from pathlib import Path

    from lunar_platform.io.ohrc_browse import (
        browse_decimation,
        load_browse_image,
        load_geometry_grid,
    )
    from lunar_platform.localization.multi_view import localize_against_library
    from lunar_platform.scene.builder import build_library
    from lunar_platform.scene.models import Footprint
    from lunar_platform.utils.config import get_config

    library = build_library(archive)
    source = archive.get_simulation_source()
    image = load_browse_image(source["archive"], source["browse_member"])
    decimation = browse_decimation(image.shape, source["label"])
    grid = load_geometry_grid(Path(source["archive"]), source["geometry_member"])

    col, row, width, height = 300, 4000, 600, 600
    crop_path = tmp_path / "query.png"
    cv2.imwrite(str(crop_path), image[row : row + height, col : col + width])

    truth = grid.lonlat_at((col + width / 2) * decimation, (row + height / 2) * decimation)
    corners = [
        grid.lonlat_at(c * decimation, r * decimation)
        for c, r in (
            (col, row), (col + width, row), (col + width, row + height), (col, row + height)
        )
    ]
    footprint = Footprint(
        corners=corners, centre_longitude=truth[0], centre_latitude=truth[1]
    )

    config = get_config("registration") or {}
    config["max_dimension_px"] = 1400

    result = localize_against_library(
        str(crop_path),
        footprint,
        library.by_sensor("CH2_OHRC")[0].illumination,
        2.8535,
        library,
        archive,
        k=3,
        config=config,
    )

    assert result.succeeded, result.note
    error = great_circle_m(truth[0], truth[1], result.longitude, result.latitude)
    # Roughly a couple of pixels at this product's 2.85 m sampling.
    assert error < 30.0, f"recovered position is {error:.1f} m from truth"

    contributing = [c for c in result.contributions if c.succeeded]
    assert contributing, "no view contributed a position"
    assert any("ohrc" in c.view_id for c in contributing)
