"""
Tests for raw-data discovery and the registration workflow.

None of these need the real mission archives: fixtures are built in a tmp_path,
so the suite runs offline and stays fast. Two tests do touch data/raw if it
happens to be present, and both only assert that it was left alone.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lunar_platform.core.data_manager import DataManager, DataMode
from lunar_platform.io.raw_inventory import discover_raw, parse_pds4_label
from lunar_platform.preprocessing.illumination import apply_representation
from lunar_platform.registration.distribution import (
    distribution_metrics,
    select_uniform_matches,
)
from lunar_platform.registration.pipeline import run_registration
from lunar_platform.registration.products import write_registration_products
from lunar_platform.registration.validation import (
    apply_illumination_change,
    apply_known_transform,
    transform_error,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _textured_image(width: int = 420, height: int = 420, seed: int = 3) -> np.ndarray:
    """A repeatable textured image with enough structure for SIFT."""
    rng = np.random.default_rng(seed)
    image = rng.integers(60, 190, size=(height, width), dtype=np.uint8)
    for _ in range(90):
        cx, cy = rng.integers(20, width - 20), rng.integers(20, height - 20)
        radius = int(rng.integers(6, 22))
        cv2.circle(image, (int(cx), int(cy)), radius, int(rng.integers(200, 255)), 2)
        cv2.circle(image, (int(cx), int(cy)), max(1, radius // 2), int(rng.integers(20, 70)), -1)
    return cv2.GaussianBlur(image, (3, 3), 0)


@pytest.fixture
def fake_raw(tmp_path: Path) -> Path:
    """A miniature data/raw with an archived source and a loose reference."""
    raw = tmp_path / "raw"
    ohrc_dir = raw / "chandrayaan2" / "ohrc"
    ref_dir = raw / "reference" / "lro" / "nac"
    ohrc_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)

    image = _textured_image()
    browse_path = tmp_path / "browse.png"
    cv2.imwrite(str(browse_path), image)

    label = """<?xml version="1.0"?><Product_Observational>
      <logical_identifier>urn:test:ohrc</logical_identifier>
      <name>Chandrayaan-2</name>
      <start_date_time>2019-09-07T04:38:12Z</start_date_time>
      <data_type>UnsignedByte</data_type>
      <pixel_resolution>0.24</pixel_resolution>
      <Axis_Array><axis_name>Line</axis_name><elements>420</elements></Axis_Array>
      <Axis_Array><axis_name>Sample</axis_name><elements>420</elements></Axis_Array>
    </Product_Observational>"""

    archive = ohrc_dir / "ch2_ohr_test_d_img_g26.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data/calibrated/x/ch2_ohr_test_d_img_g26.xml", label)
        zf.writestr("data/calibrated/x/ch2_ohr_test_d_img_g26.img", b"\x00" * 5000)
        zf.write(browse_path, "browse/calibrated/x/ch2_ohr_test_b_brw_g26.png")
        zf.writestr("geometry/calibrated/x/ch2_ohr_test_g_grd_g26.csv",
                    "Longitude,Latitude,Pixel,Scan\n22.5,-70.5,0,0\n22.6,-70.6,100,100\n")
        zf.writestr("miscellaneous/calibrated/x/ch2_ohr_test_d_img_g26.oath", "ORBTATTD-HDR")

    cv2.imwrite(str(ref_dir / "M_test.png"), image)
    return raw


# --- 1. raw data discovery ---------------------------------------------------

def test_discovery_finds_source_and_reference(fake_raw: Path):
    inventory = discover_raw(fake_raw)
    assert inventory["exists"]
    assert len(inventory["source_products"]) == 1
    assert len(inventory["reference_products"]) == 1

    source = inventory["source_products"][0]
    assert source.is_archive
    assert source.instrument == "OHRC"
    assert source.label["pixel_resolution"] == 0.24
    assert source.label["lines"] == 420


def test_discovery_classifies_members_and_nav(fake_raw: Path):
    inventory = discover_raw(fake_raw)
    source = inventory["source_products"][0]
    kinds = {m.kind for m in source.members}
    assert {"image", "browse", "geometry", "label", "nav"} <= kinds

    # The browse PNG must win over the browse label as the usable product.
    assert source.member_by_kind("browse").name.endswith(".png")
    assert len(inventory["nav_files"]) == 1


def test_parse_pds4_label_reports_absent_fields_as_missing():
    label = parse_pds4_label("<Product><name>Chandrayaan-2</name></Product>")
    assert label["mission_name"] == "Chandrayaan-2"
    assert "pixel_resolution" not in label  # not invented


# --- 2. DataManager source/reference discovery -------------------------------

def test_data_manager_exposes_source_and_reference(fake_raw: Path):
    dm = DataManager(base_dir=str(fake_raw.parent))
    capabilities = dm.get_capabilities()
    assert capabilities["source_image"] is True
    assert capabilities["reference_image"] is True
    assert capabilities["navigation"] is True
    assert capabilities["registration"] is True
    assert capabilities["source_ref"].startswith("/vsizip/")


# --- 3 & 4. missing DEM / SPICE must not break registration ------------------

def test_missing_dem_and_spice_do_not_block_registration(fake_raw: Path):
    dm = DataManager(base_dir=str(fake_raw.parent))
    capabilities = dm.get_capabilities()

    assert capabilities["dem"] is False
    assert capabilities["spice"] is False
    # The whole point: registration stays available anyway.
    assert capabilities["registration"] is True
    assert any("DEM not available" in m for m in capabilities["messages"])
    assert any("SPICE not available" in m for m in capabilities["messages"])


# --- 5. spatially distributed match points -----------------------------------

def test_uniform_selection_caps_points_per_cell():
    # 200 points crammed into one corner.
    clustered = np.column_stack([
        np.random.default_rng(0).uniform(0, 40, 200),
        np.random.default_rng(1).uniform(0, 40, 200),
    ])
    scores = np.linspace(0, 1, 200)

    keep = select_uniform_matches(
        clustered, clustered, scores, 400, 400, grid_cols=8, grid_rows=8, max_per_cell=3
    )
    assert len(keep) <= 3 * 8 * 8
    counts = distribution_metrics(clustered[keep], 400, 400, 8, 8)
    assert counts["max_points_in_cell"] <= 3


def test_uniform_selection_improves_coverage():
    rng = np.random.default_rng(5)
    spread = np.column_stack([rng.uniform(0, 400, 600), rng.uniform(0, 400, 600)])
    scores = rng.uniform(0, 1, 600)

    before = distribution_metrics(spread, 400, 400)
    keep = select_uniform_matches(spread, spread, scores, 400, 400, max_per_cell=2)
    after = distribution_metrics(spread[keep], 400, 400)

    # Far fewer points, but they still span the frame, and more evenly.
    assert len(keep) < len(spread)
    assert after["gini"] <= before["gini"] + 1e-6


def test_uniform_selection_rejects_invalid_grid():
    pts = np.zeros((4, 2))
    with pytest.raises(ValueError):
        select_uniform_matches(pts, pts, np.zeros(4), 100, 100, grid_cols=0)


# --- 6, 7 & 8. RANSAC, metrics and output products ---------------------------

@pytest.fixture
def registration_pair(tmp_path: Path):
    """A real known transform applied to a textured image, written as files."""
    base = _textured_image(480, 480, seed=11)
    warped, truth = apply_known_transform(base, rotation_deg=5.0, scale=0.9,
                                          translation=(8.0, -6.0))
    warped = apply_illumination_change(warped)

    fixed = tmp_path / "fixed.png"
    moving = tmp_path / "moving.png"
    cv2.imwrite(str(fixed), base)
    cv2.imwrite(str(moving), warped)
    return str(moving), str(fixed), truth, base


def test_registration_verifies_inliers_and_recovers_transform(registration_pair):
    moving, fixed, truth, base = registration_pair
    outcome = run_registration(moving, fixed, "unit_pair", {
        "max_dimension_px": 640,
        "geometry": {"default_model": "similarity", "ransac_threshold": 3.0},
        "min_matches": 20, "min_inlier_ratio": 0.3,
        "max_rmse_px": 5.0, "min_spatial_coverage": 0.15,
    })

    metrics = outcome.metrics
    assert metrics["n_candidate_matches"] > 0
    # Candidates and verified inliers are distinct quantities.
    assert metrics["n_verified_inliers"] <= metrics["n_uniform_selected"]
    assert metrics["n_uniform_selected"] <= metrics["n_candidate_matches"]
    assert outcome.succeeded, metrics["quality_reasons"]

    error = transform_error(outcome.transform, np.linalg.inv(truth),
                            base.shape[1], base.shape[0])
    assert error["mean_px"] < 3.0


def test_registration_reports_failure_rather_than_guessing(tmp_path: Path):
    """Unrelated images must not produce a passing registration."""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    cv2.imwrite(str(a), _textured_image(300, 300, seed=1))
    cv2.imwrite(str(b), np.full((300, 300), 128, dtype=np.uint8))

    outcome = run_registration(str(a), str(b), "unrelated", {"max_dimension_px": 320})
    assert outcome.succeeded is False
    assert outcome.metrics["quality_passed"] is False
    assert outcome.metrics["quality_reasons"]


def test_registration_writes_all_products(registration_pair, tmp_path: Path):
    moving, fixed, _, _ = registration_pair
    outcome = run_registration(moving, fixed, "product_scene", {
        "max_dimension_px": 640, "min_matches": 20, "min_inlier_ratio": 0.3,
        "max_rmse_px": 5.0, "min_spatial_coverage": 0.15,
    })
    written = write_registration_products(outcome, tmp_path / "outputs", moving, fixed)

    scene = Path(written["scene_dir"])
    for name in ("match_points.json", "matches.csv", "transform.json", "metrics.json"):
        assert (scene / name).exists(), name

    matches = json.loads((scene / "match_points.json").read_text())
    assert matches["matches"], "match-point product must not be empty"
    first = matches["matches"][0]
    for field in ("source_x", "source_y", "reference_x", "reference_y", "error_px", "inlier"):
        assert field in first

    manifest = written["manifest"]
    assert manifest["uses_dem"] is False
    assert manifest["uses_spice"] is False


def test_illumination_representations_available():
    image = _textured_image(160, 160)
    for name in ("raw", "clahe", "local_contrast", "gradient"):
        out = apply_representation(image, name)
        assert out.shape == image.shape
    with pytest.raises(ValueError):
        apply_representation(image, "not_a_representation")


# --- 9. data/raw is never modified -------------------------------------------

@pytest.mark.skipif(not (REPO_ROOT / "data" / "raw").exists(), reason="no data/raw present")
def test_discovery_does_not_modify_raw():
    raw = REPO_ROOT / "data" / "raw"
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns)
              for p in sorted(raw.rglob("*")) if p.is_file()}

    discover_raw(raw)
    DataManager(base_dir=str(REPO_ROOT / "data")).get_capabilities()

    after = {p: (p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(raw.rglob("*")) if p.is_file()}
    assert before == after, "data/raw must be treated as immutable"


@pytest.mark.skipif(not (REPO_ROOT / "data" / "raw").exists(), reason="no data/raw present")
def test_huge_archive_member_is_never_selected():
    """The source reference must resolve to the browse product, not the 1.1 GB image."""
    dm = DataManager(base_dir=str(REPO_ROOT / "data"))
    source_ref = dm.get_source_image_ref()
    if source_ref is None:
        pytest.skip("no source product discovered")
    assert not source_ref.endswith(".img"), "must not point at the full-resolution image"


# --- 10. DEMO pipeline still works offline -----------------------------------

def test_demo_mode_still_resolves_without_raw(tmp_path: Path):
    dm = DataManager(base_dir=str(tmp_path))
    assert dm.active_mode == DataMode.DEMO
    capabilities = dm.get_capabilities()
    assert capabilities["mode"] == "DEMO"
    # No raw data at all must not raise.
    assert capabilities["registration"] is False
