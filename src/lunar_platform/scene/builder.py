"""
Building the scene library from whatever is under data/raw.

Each product is asked the same four questions - where does it look, how finely
does it sample, when was it taken, how was it lit - and answers them through
its own geometry: the OHRC bundle through its NAV grid, the LROC NAC frame
through its SPICE camera model. A product that cannot answer is still admitted,
with the gaps recorded, because a view with no footprint is still a view
somebody may want to look at; it simply cannot be retrieved by ground.

Nothing is read but labels and geometry. Building a library over hundreds of
products must not mean decoding hundreds of rasters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..registration.ground_scale import resolve_ground_scale
from .illumination import illumination_at
from .models import Footprint, SceneLibrary, SceneView

FOOTPRINT_EDGE_SAMPLES = 5


def _ohrc_view(manager, sector_id: str) -> Optional[SceneView]:
    """The Chandrayaan-2 OHRC strip, placed by its own NAV geometry grid."""
    source = manager.get_simulation_source()
    if source is None:
        return None

    from ..io.ohrc_browse import browse_decimation, load_browse_image, load_geometry_grid

    notes: list[str] = []
    label = source.get("label") or {}
    source_ref = (
        f"/vsizip/{source['archive']}/{source['browse_member']}"
        if source.get("archive")
        else str(source.get("browse_path"))
    )

    footprint = Footprint()
    gsd = None
    width = height = None

    try:
        archive = source.get("archive") or source.get("browse_path")
        image = load_browse_image(archive, source.get("browse_member"))
        height, width = int(image.shape[0]), int(image.shape[1])
        decimation = browse_decimation(image.shape, label)

        geometry = source.get("geometry_member")
        grid = load_geometry_grid(
            Path(source.get("archive") or source.get("geometry_path")), geometry
        )

        corners = []
        for v in (0, height - 1):
            for u in (0, width - 1):
                corners.append(grid.lonlat_at(u * decimation, v * decimation))
        # Order the four corners so the polygon does not self-intersect.
        corners = [corners[0], corners[1], corners[3], corners[2]]
        centre = grid.lonlat_at((width / 2) * decimation, (height / 2) * decimation)
        footprint = Footprint(
            corners=[(round(c[0], 6), round(c[1], 6)) for c in corners],
            centre_longitude=round(centre[0], 6),
            centre_latitude=round(centre[1], 6),
        )
    except Exception as exc:
        notes.append(f"Footprint unavailable: {exc}")

    scale = resolve_ground_scale(source_ref, label=label, data_manager=manager)
    if scale.known:
        gsd = round(scale.gsd_m, 4)
        notes.append(f"Ground scale {scale.method}: {scale.detail}")

    acquired = label.get("start_date_time")
    illumination = (
        illumination_at(acquired, footprint.centre_longitude, footprint.centre_latitude)
        if acquired and footprint.known
        else None
    )
    if illumination is None:
        notes.append("Illumination unavailable: no acquisition time or no footprint.")
    elif not illumination.known:
        notes.append(
            "Illumination could not be computed; SPICE kernels are required even "
            "for sun-only geometry."
        )
    elif illumination.emission_deg is None:
        notes.append(
            "Emission and phase are unavailable: this project carries no "
            "Chandrayaan-2 ephemeris, so the camera cannot be located."
        )

    return SceneView(
        view_id=f"ohrc:{source['product_id']}",
        sector_id=sector_id,
        sensor="CH2_OHRC",
        product_id=source["product_id"],
        source_ref=source_ref,
        gsd_m=gsd,
        width_px=width,
        height_px=height,
        acquired_utc=acquired,
        footprint=footprint,
        illumination=illumination or illumination_at("", 0, 0),
        notes=notes,
    )


def _nac_view(manager, sector_id: str) -> Optional[SceneView]:
    """The LROC NAC frame, placed by its SPICE camera model."""
    reference = manager.get_nac_reference()
    if reference is None:
        return None

    from ..planetary.spice_adapter import NacCameraModel, parse_nac_label

    notes: list[str] = []
    footprint = Footprint()
    gsd = None
    observation = None

    kernels = manager.get_spice_kernels()
    if kernels is None or not kernels.is_complete:
        notes.append(
            "SPICE kernels incomplete, so this frame cannot be placed. Its label "
            "carries no geographic metadata of its own."
        )
    else:
        try:
            observation = parse_nac_label(
                reference["label_path"].read_text("utf-8", "replace"),
                reference["product_id"],
            )
            camera = NacCameraModel(observation, kernels)
            described = camera.footprint()
            corners = [
                described["corners"][k]
                for k in ("upper_left", "upper_right", "lower_right", "lower_left")
            ]
            if all(corners) and described["centre"]:
                footprint = Footprint(
                    corners=[(c["longitude"], c["latitude"]) for c in corners],
                    centre_longitude=described["centre"]["longitude"],
                    centre_latitude=described["centre"]["latitude"],
                )
            measured = camera.ground_sampling_m()
            if np.isfinite(measured):
                gsd = round(float(measured), 4)
        except Exception as exc:
            notes.append(f"SPICE geometry failed: {exc}")

    acquired = observation.start_time if observation else None
    illumination = (
        illumination_at(
            acquired, footprint.centre_longitude, footprint.centre_latitude, observer="LRO"
        )
        if acquired and footprint.known
        else illumination_at("", 0, 0)
    )

    return SceneView(
        view_id=f"nac:{reference['product_id']}",
        sector_id=sector_id,
        sensor="LRO_NAC",
        product_id=reference["product_id"],
        source_ref=str(reference["image_path"]),
        gsd_m=gsd,
        width_px=observation.samples if observation else None,
        height_px=observation.lines if observation else None,
        acquired_utc=acquired,
        footprint=footprint,
        illumination=illumination,
        notes=notes,
    )


def build_library(manager, sector_id: Optional[str] = None) -> SceneLibrary:
    """
    Catalogue every product under data/raw as a view.

    A builder that fails on one product would make the whole library
    unavailable, so each is attempted independently and its failure recorded on
    the view rather than raised.
    """
    sector_id = sector_id or manager.active_sector
    library = SceneLibrary(sector_id=sector_id)

    # Furnish the kernels once, up front. Otherwise whether a view gets its
    # illumination depends on build order: the NAC camera model loads them as a
    # side effect, so anything built before it silently had no SPICE and
    # anything after it did.
    kernels = manager.get_spice_kernels()
    if kernels is not None and kernels.is_complete:
        try:
            kernels.load()
        except Exception:
            pass

    for build in (_ohrc_view, _nac_view):
        try:
            view = build(manager, sector_id)
        except Exception:
            view = None
        if view is not None:
            library.add(view)

    return library
