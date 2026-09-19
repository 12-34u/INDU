"""
API surface for the simulation running on real data/raw imagery.

Skips when the mission bundle is absent so the suite still runs without it.
The demo path is asserted alongside, because switching a mode must not change
what the other mode reports.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def raw_available() -> bool:
    return bool(client.get("/data/status").json().get("real_raw_available"))


requires_raw = pytest.mark.skipif(
    not raw_available(), reason="No Chandrayaan-2 bundle under data/raw"
)


@pytest.fixture(autouse=True)
def restore_demo_mode():
    """Leave the server in DEMO however a test ends; the mode is global."""
    yield
    client.post("/data/mode", json={"mode": "DEMO"})


def test_status_reports_whether_raw_data_can_drive_the_simulation():
    body = client.get("/data/status").json()
    assert "real_raw_available" in body
    assert isinstance(body["real_raw_available"], bool)


@requires_raw
def test_real_raw_mode_can_be_selected():
    response = client.post("/data/mode", json={"mode": "REAL_RAW"})
    assert response.status_code == 200
    assert response.json()["current_mode"] == "REAL_RAW"


@requires_raw
def test_basemap_describes_its_provenance():
    client.post("/data/mode", json={"mode": "REAL_RAW"})
    body = client.get("/simulation/basemap").json()

    assert body["is_synthetic"] is False
    assert body["georeferenced"] is True
    assert body["gsd_m"] > 0
    assert body["n_reference_craters"] > 0
    # The reference must name the product it was read from, in place.
    assert "ch2_ohr" in body["source_ref"].lower()


@requires_raw
def test_basemap_image_is_served_as_a_display_png():
    client.post("/data/mode", json={"mode": "REAL_RAW"})
    response = client.get("/simulation/basemap.png?max_size=256")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 1000


@requires_raw
def test_reference_map_is_detected_craters_in_real_mode():
    client.post("/data/mode", json={"mode": "REAL_RAW"})
    body = client.get("/reference_map").json()

    assert body["is_synthetic"] is False
    assert body["craters"]
    assert all(c["provenance"] == "detected" for c in body["craters"])


def test_reference_map_stays_synthetic_in_demo_mode():
    client.post("/data/mode", json={"mode": "DEMO"})
    body = client.get("/reference_map").json()
    assert body["is_synthetic"] is True


@requires_raw
def test_perception_on_real_imagery_recovers_the_pose():
    client.post("/data/mode", json={"mode": "REAL_RAW"})
    basemap = client.get("/simulation/basemap").json()

    response = client.post(
        "/perception",
        json={
            "x_m": basemap["width_m"] / 2.0,
            "y_m": basemap["height_m"] / 2.0,
            "heading_deg": 25.0,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["is_real_imagery"] is True
    assert body["localization"]["is_synthetic"] is False
    assert body["localization"]["method"] == "sift-basemap-registration"
    assert body["localization"]["position_error_m"] < basemap["gsd_m"]
    assert body["geodetic"]["estimated"]["latitude"] < 0
    # The camera must default to the imagery's own resolution rather than
    # magnifying it.
    assert body["camera"]["gsd_m"] == pytest.approx(basemap["gsd_m"], rel=1e-3)
    assert all(stage["status"] == "ok" for stage in body["stages"])


@requires_raw
def test_real_imagery_can_be_requested_without_switching_mode():
    client.post("/data/mode", json={"mode": "DEMO"})
    body = client.post(
        "/perception",
        json={"x_m": 1200.0, "y_m": 1200.0, "heading_deg": 0.0, "use_real_data": True},
    ).json()
    assert body["is_real_imagery"] is True
    assert body["basemap"]["is_synthetic"] is False


def test_demo_perception_is_unchanged_and_still_synthetic():
    client.post("/data/mode", json={"mode": "DEMO"})
    body = client.post(
        "/perception", json={"x_m": 500.0, "y_m": 500.0, "heading_deg": 45.0}
    ).json()

    assert body["is_real_imagery"] is False
    assert body["localization"]["is_synthetic"] is True
    assert body["localization"]["method"] == "constellation-translation-vote"
    # The demo camera defaults are the ones the frontend has always relied on.
    assert body["camera"]["width_px"] == 512
    assert body["camera"]["gsd_m"] == 0.5


# -- SPICE geolocation and real elevation ------------------------------------


def spice_available() -> bool:
    body = client.get("/data/capabilities").json()
    return bool(body.get("spice"))


requires_spice = pytest.mark.skipif(
    not spice_available(), reason="No SPICE kernels under data/raw"
)


@requires_spice
def test_nac_image_is_placed_on_the_moon():
    """The NAC label has no geographic metadata; this comes from the kernels."""
    body = client.get("/reference/nac").json()

    assert body["kernels"]["complete"] is True
    footprint = body["footprint"]
    assert footprint["centre"] is not None
    assert -90.0 < footprint["min_latitude"] < footprint["max_latitude"] < 0.0
    assert "SPICE" in footprint["method"]
    # NAC is a 0.5 m/px camera; a 2x summed product is about a metre.
    assert 0.5 < body["cross_track_gsd_m"] < 2.5


@requires_spice
def test_overlap_with_the_sector_is_reported():
    client.post("/data/mode", json={"mode": "REAL_RAW"})
    coverage = client.get("/reference/nac?coverage_grid=5").json()["sector_coverage"]

    assert coverage is not None
    assert coverage["points_tested"] > 0
    assert isinstance(coverage["overlaps"], bool)
    # Stated as a computation over sector points, not a bounding-box guess.
    assert "bounding boxes" in coverage["note"]


@requires_raw
def test_real_mode_terrain_shares_the_basemap_frame():
    client.post("/data/mode", json={"mode": "REAL_RAW"})
    terrain = client.get("/terrain?max_grid_size=40").json()
    basemap = client.get("/simulation/basemap").json()

    assert terrain["dem_width"] == basemap["width_px"]
    assert terrain["gsd_m"] == pytest.approx(basemap["gsd_m"], rel=1e-3)
    assert terrain["width_m"] == pytest.approx(basemap["width_m"], rel=1e-3)
    # Hazards come from craters detected in the imagery, not a demo file.
    assert terrain["craters_are_detected"] is True
    assert terrain["craters"]


@requires_raw
def test_real_mode_elevation_is_measured_when_a_dem_is_present():
    """With a real DEM the surface must be lunar topography, not a generated slope."""
    status = client.get("/data/capabilities").json()
    client.post("/data/mode", json={"mode": "REAL_RAW"})
    terrain = client.get("/terrain?max_grid_size=40").json()

    relief = terrain["max_elevation"] - terrain["min_elevation"]
    assert relief > 0
    if status.get("dem"):
        # Real highlands relative to the 1737.4 km datum, not a ramp from zero.
        assert -9000.0 < terrain["min_elevation"] < 11000.0
