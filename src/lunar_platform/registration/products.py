"""
Registration output products.

Writes outputs/registration/<scene_id>/ with the registered image, the
match-point product in both JSON and CSV, the transform, and the metrics.

Display derivatives are written alongside the full arrays so the frontend never
has to load a science-sized raster into a browser.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .pipeline import RegistrationOutcome

MATCH_CSV_COLUMNS = [
    "index",
    "source_x",
    "source_y",
    "reference_x",
    "reference_y",
    "error_px",
    "inlier",
    "match_score",
    "refine_correlation",
]

DISPLAY_MAX_DIMENSION = 900


def _write_display(image: np.ndarray, path: Path, max_dimension: int = DISPLAY_MAX_DIMENSION) -> dict:
    scale = min(1.0, max_dimension / max(image.shape[0], image.shape[1]))
    out = (
        cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else image
    )
    cv2.imwrite(str(path), out)
    return {
        "file": path.name,
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
        "scale_from_working": round(float(scale), 5),
    }


def write_registration_products(
    outcome: RegistrationOutcome,
    output_root: Path,
    source_ref: str,
    reference_ref: str,
) -> dict:
    """Write every product for one registration run and return a manifest."""
    scene_dir = Path(output_root) / outcome.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    written: dict = {"scene_dir": str(scene_dir), "files": []}

    if outcome.registered_image is not None:
        registered_path = scene_dir / "registered_image.tif"
        cv2.imwrite(str(registered_path), outcome.registered_image)
        written["files"].append(registered_path.name)

    with open(scene_dir / "match_points.json", "w") as f:
        json.dump(
            {
                "scene_id": outcome.scene_id,
                "n_matches": len(outcome.matches),
                "n_inliers": sum(1 for m in outcome.matches if m["inlier"]),
                "matches": outcome.matches,
            },
            f,
            indent=2,
        )
    written["files"].append("match_points.json")

    with open(scene_dir / "matches.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_CSV_COLUMNS)
        writer.writeheader()
        for match in outcome.matches:
            writer.writerow({k: match.get(k) for k in MATCH_CSV_COLUMNS})
    written["files"].append("matches.csv")

    with open(scene_dir / "transform.json", "w") as f:
        json.dump(
            {
                "scene_id": outcome.scene_id,
                "model_type": outcome.model_type,
                "succeeded": outcome.succeeded,
                "matrix": outcome.transform.tolist() if outcome.transform is not None else None,
                "frame": "image-to-image; not georeferenced (no camera model in use)",
            },
            f,
            indent=2,
        )
    written["files"].append("transform.json")

    display: dict = {}
    if outcome.source_image is not None:
        display["source"] = _write_display(outcome.source_image, scene_dir / "display_source.png")
    if outcome.reference_image is not None:
        display["reference"] = _write_display(
            outcome.reference_image, scene_dir / "display_reference.png"
        )
    if outcome.registered_image is not None:
        display["registered"] = _write_display(
            outcome.registered_image, scene_dir / "display_registered.png"
        )
    written["files"].extend(v["file"] for v in display.values())

    metrics = {
        "scene_id": outcome.scene_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "succeeded": outcome.succeeded,
        "source_ref": source_ref,
        "reference_ref": reference_ref,
        "uses_dem": False,
        "uses_spice": False,
        "caveat": (
            "Image-space registration only. Without a camera model or DEM these "
            "residuals describe agreement between two image planes, not ground accuracy."
        ),
        "metrics": outcome.metrics,
        "stages": [
            {
                "key": s.key,
                "label": s.label,
                "status": s.status,
                "detail": s.detail,
                "duration_ms": s.duration_ms,
            }
            for s in outcome.stages
        ],
        "notes": outcome.notes,
        "display": display,
    }
    with open(scene_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    written["files"].append("metrics.json")

    written["manifest"] = metrics
    return written
