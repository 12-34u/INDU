"""
The simulation running on real Chandrayaan-2 data.

These tests use the actual bundle under data/raw and skip when it is absent,
so the suite still runs on a checkout without the mission archive. What they
assert is that the simulation is driven by the data rather than by a
rendering: that the observation is cut from the real raster, that the position
is recovered from it, and that a fix which cannot be justified is refused
rather than reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunar_platform.core.data_manager import DataManager
from lunar_platform.crater_detection.base import detect_all
from lunar_platform.crater_detection.shadow_pairs import ShadowPairDetector
from lunar_platform.localization.estimator import run_real_pipeline
from lunar_platform.localization.image_match import BasemapLocalizer, heading_from_transform
from lunar_platform.map.reference_map import build_reference_map_from_basemap
from lunar_platform.observation.basemap import build_sector_basemap_from_source
from lunar_platform.observation.quadrants import split_quadrants
from lunar_platform.observation.real_view import (
    basemap_to_image_transform,
    native_camera,
    render_real_observation,
)
from lunar_platform.simulation.camera import NadirCamera, RoverPose

SECTOR_SIZE_M = 2400.0


@pytest.fixture(scope="module")
def source():
    descriptor = DataManager().get_simulation_source()
    if descriptor is None:
        pytest.skip("No Chandrayaan-2 OHRC bundle under data/raw")
    return descriptor


@pytest.fixture(scope="module")
def basemap(source):
    centre = DataManager().get_sector_centre_lonlat()
    return build_sector_basemap_from_source(
        source,
        centre_longitude=centre[0] if centre else None,
        centre_latitude=centre[1] if centre else None,
        size_m=SECTOR_SIZE_M,
    )


@pytest.fixture(scope="module")
def localizer():
    return BasemapLocalizer()


@pytest.fixture(scope="module")
def reference_craters(basemap):
    return build_reference_map_from_basemap(basemap, ShadowPairDetector())


def poses(basemap, camera, count, seed=17):
    """Poses whose whole footprint stays inside the imagery."""
    rng = np.random.default_rng(seed)
    margin = max(camera.footprint_m) / 2.0
    return [
        RoverPose(
            float(rng.uniform(margin, basemap.width_m - margin)),
            float(rng.uniform(margin, basemap.height_m - margin)),
            float(rng.uniform(0.0, 360.0)),
        )
        for _ in range(count)
    ]


# -- the basemap is real, and placed on the Moon ----------------------------


def test_basemap_is_real_imagery_not_a_rendering(basemap):
    assert basemap.describe()["is_synthetic"] is False
    assert basemap.image.dtype == np.uint8
    # A rendering of regolith is smooth; a real strip is not.
    assert basemap.image.std() > 10.0
    assert len(np.unique(basemap.image)) > 100


def test_basemap_is_georeferenced_from_the_bundles_own_grid(basemap):
    assert basemap.grid is not None
    centre = basemap.lonlat_at(basemap.width_m / 2.0, basemap.height_m / 2.0)
    assert centre is not None
    longitude, latitude = centre
    # The sector this project describes sits in the southern highlands.
    assert -72.0 < latitude < -70.0
    assert 22.0 < longitude < 23.5


def test_basemap_anchors_on_the_requested_sector_centre(basemap):
    manifest_centre = DataManager().get_sector_centre_lonlat()
    if manifest_centre is None or basemap.anchor_error_m is None:
        pytest.skip("Sector manifest carries no centre coordinate")
    # The NAV lattice is sampled every ~100 full-resolution pixels, so the
    # anchor is good to about half a node and no better.
    assert basemap.anchor_error_m < 50.0


def test_metre_scale_is_measured_rather_than_nominal(basemap, source):
    nominal = float(source["label"].get("pixel_resolution", 0.24))
    # The browse product decimates by 10x, so its scale is 10 nominal pixels.
    assert basemap.gsd_m == pytest.approx(nominal * basemap.decimation, rel=0.35)
    assert basemap.gsd_m > 0


# -- the observation is cut from that imagery -------------------------------


def test_observation_transform_matches_the_camera_model(basemap):
    """The cut and the camera must agree, or every offset downstream is wrong."""
    camera = native_camera(basemap, 128)
    pose = RoverPose(900.0, 1100.0, 53.0)
    transform = basemap_to_image_transform(pose, camera, basemap)

    rng = np.random.default_rng(0)
    for _ in range(50):
        x = float(rng.uniform(0, basemap.width_m))
        y = float(rng.uniform(0, basemap.height_m))
        u_base, v_base = basemap.sector_to_pixel(x, y)
        mapped = transform @ np.array([u_base, v_base, 1.0])
        expected = camera.world_to_image(pose, x, y)
        assert mapped[0] == pytest.approx(expected[0], abs=1e-6)
        assert mapped[1] == pytest.approx(expected[1], abs=1e-6)


def test_observation_pixels_come_from_the_basemap(basemap):
    camera = native_camera(basemap, 192)
    # Sit the rover exactly on a basemap pixel. At native resolution and a
    # heading that is a whole quarter turn, the cut then maps pixel centres to
    # pixel centres, so the observation carries the raster's own values rather
    # than an interpolation of them.
    pixel_u, pixel_v = 400, 400
    pose = RoverPose(*basemap.pixel_to_sector(pixel_u, pixel_v), heading_deg=0.0)
    observation = render_real_observation(pose, camera, basemap)

    assert observation.valid_fraction == pytest.approx(1.0)
    expected = int(basemap.image[pixel_v, pixel_u])
    centre = int(observation.image[camera.height_px // 2, camera.width_px // 2])
    assert abs(centre - expected) <= 1

    # The frame is a real raster, not a smooth rendering.
    assert observation.image.std() > 10.0


def test_observation_off_the_edge_is_reported_not_padded(basemap):
    camera = native_camera(basemap, 256)
    observation = render_real_observation(RoverPose(0.0, 0.0, 0.0), camera, basemap)
    # A quarter of a centred frame overlaps the corner at most.
    assert observation.valid_fraction < 0.5
    assert not observation.valid_mask.all()


def test_sun_bearing_turns_with_the_rover(basemap):
    camera = native_camera(basemap, 64)
    straight = render_real_observation(RoverPose(1000.0, 1000.0, 0.0), camera, basemap)
    turned = render_real_observation(RoverPose(1000.0, 1000.0, 90.0), camera, basemap)
    difference = (straight.sun_direction_deg - turned.sun_direction_deg) % 360.0
    assert difference == pytest.approx(90.0, abs=1e-6)


# -- craters are detected in the real imagery -------------------------------


def test_reference_map_is_detected_not_generated(reference_craters):
    assert len(reference_craters) > 30
    assert all(c.provenance == "detected" for c in reference_craters)
    assert all(c.diameter_m >= 20.0 for c in reference_craters)


def test_detector_reports_points_in_every_quadrant(basemap):
    camera = native_camera(basemap, 256)
    observation = render_real_observation(RoverPose(1200.0, 1200.0, 20.0), camera, basemap)
    detector = ShadowPairDetector(sun_direction_deg=basemap.sun_direction_deg)
    points = detect_all(detector, split_quadrants(observation.image))

    assert points
    assert {p.quadrant for p in points} == {"Q1", "Q2", "Q3", "Q4"}


def test_detector_finds_nothing_in_featureless_terrain():
    """A flat frame has no shadows to pair, so it must yield no craters."""
    detector = ShadowPairDetector()
    assert detector.detect_craters(np.full((256, 256), 120, np.uint8)) == []


# -- localisation ------------------------------------------------------------


def test_heading_is_recovered_from_a_synthesised_transform(basemap):
    """The heading inversion must match the forward model it inverts."""
    camera = native_camera(basemap, 128)
    for heading in (0.0, 37.0, 123.5, 270.0):
        forward = basemap_to_image_transform(RoverPose(500.0, 500.0, heading), camera, basemap)
        square = np.vstack([forward, [0.0, 0.0, 1.0]])
        inverse = np.linalg.inv(square)[:2]
        assert heading_from_transform(inverse) == pytest.approx(heading, abs=1e-6)


def test_position_is_recovered_from_real_imagery(basemap, reference_craters, localizer):
    camera = native_camera(basemap, 256)
    detector = ShadowPairDetector(sun_direction_deg=basemap.sun_direction_deg)

    errors = []
    for pose in poses(basemap, camera, 6):
        result = run_real_pipeline(
            pose, camera, basemap, reference_craters, detector, localizer
        )
        assert result.localization.position_error_m is not None, result.localization.note
        errors.append(result.localization.position_error_m)

    # Sub-pixel against a basemap sampled at roughly 2.9 m/px.
    assert max(errors) < basemap.gsd_m
    assert float(np.median(errors)) < 3.0


def test_ground_truth_is_the_pose_the_observation_was_cut_from(
    basemap, reference_craters, localizer
):
    camera = native_camera(basemap, 256)
    pose = RoverPose(1100.0, 1300.0, 64.0)
    result = run_real_pipeline(
        pose,
        camera,
        basemap,
        reference_craters,
        ShadowPairDetector(sun_direction_deg=basemap.sun_direction_deg),
        localizer,
    )
    localization = result.localization

    assert localization.is_synthetic is False
    assert localization.ground_truth_x_m == pose.x_m
    assert localization.ground_truth_y_m == pose.y_m
    # The reported error must be the distance it claims to be.
    assert localization.position_error_m == pytest.approx(
        float(np.hypot(localization.estimated_x_m - pose.x_m,
                       localization.estimated_y_m - pose.y_m)),
        abs=1e-6,
    )


def test_position_is_reported_as_real_coordinates(basemap, reference_craters, localizer):
    camera = native_camera(basemap, 256)
    result = run_real_pipeline(
        RoverPose(1000.0, 1000.0, 15.0),
        camera,
        basemap,
        reference_craters,
        ShadowPairDetector(sun_direction_deg=basemap.sun_direction_deg),
        localizer,
    )
    geodetic = result.geodetic
    assert "ground_truth" in geodetic and "estimated" in geodetic
    assert -72.0 < geodetic["estimated"]["latitude"] < -70.0
    # The two coordinates describe positions metres apart, so they must agree
    # closely without being identical.
    assert geodetic["estimated"]["latitude"] == pytest.approx(
        geodetic["ground_truth"]["latitude"], abs=1e-3
    )


def test_unrelated_terrain_is_refused_rather_than_fixed(basemap, source, localizer):
    """
    An observation from elsewhere in the strip has no honest answer here.

    Registration will always return its best transform; what must not happen is
    that best transform being reported as a position.
    """
    elsewhere = build_sector_basemap_from_source(
        source, centre_longitude=22.55, centre_latitude=-70.60, size_m=SECTOR_SIZE_M
    )
    if abs(elsewhere.origin_scan - basemap.origin_scan) < 5000:
        pytest.skip("Could not obtain a disjoint region of the strip")

    camera = native_camera(basemap, 256)
    fixes = 0
    for pose in poses(elsewhere, camera, 5, seed=3):
        observation = render_real_observation(pose, camera, elsewhere)
        fix = localizer.locate(
            observation.image, camera, basemap, pose.heading_deg, observation.valid_mask
        )
        fixes += int(fix.succeeded)
    assert fixes == 0


def test_a_geometrically_inconsistent_match_is_rejected(basemap, localizer):
    """A fix must agree with the heading the rover already knows."""
    camera = native_camera(basemap, 256)
    pose = RoverPose(1200.0, 1200.0, 30.0)
    observation = render_real_observation(pose, camera, basemap)

    honest = localizer.locate(observation.image, camera, basemap, pose.heading_deg)
    assert honest.succeeded

    # Same imagery, but the rover claims a heading it is not holding.
    lying = localizer.locate(observation.image, camera, basemap, pose.heading_deg + 40.0)
    assert not lying.succeeded
    assert "heading" in (lying.rejected_reason or "").lower()


def test_blank_observation_yields_no_fix(basemap, localizer):
    camera = native_camera(basemap, 256)
    blank = np.full((camera.height_px, camera.width_px), 90, np.uint8)
    fix = localizer.locate(blank, camera, basemap, 0.0)
    assert not fix.succeeded
    assert fix.rejected_reason
