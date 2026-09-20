"""
Turning a pixel in a library view into a point on the Moon.

Registration gives a transform between two image planes. That is not a
position until one of the planes is tied to the ground, so every view needs to
answer "what longitude and latitude is this pixel looking at". Each sensor
answers through its own geometry: the OHRC bundle through the NAV grid shipped
with it, the LROC NAC frame through its SPICE camera model.

A view that cannot answer is not a failure to hide. It simply cannot
contribute a position, and saying so is better than producing one from a
default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .models import SceneView

# A geolocator maps (column, row) in the view's own raster to (lon, lat).
Geolocator = Callable[[float, float], Optional[tuple[float, float]]]


def _ohrc_geolocator(view: SceneView, manager) -> Optional[Geolocator]:
    """Browse-raster pixel to lon/lat, through the bundle's NAV grid."""
    source = manager.get_simulation_source()
    if source is None:
        return None

    try:
        from ..io.ohrc_browse import browse_decimation, load_browse_image, load_geometry_grid

        archive = source.get("archive") or source.get("browse_path")
        image = load_browse_image(archive, source.get("browse_member"))
        decimation = browse_decimation(image.shape, source.get("label") or {})
        grid = load_geometry_grid(
            Path(source.get("archive") or source.get("geometry_path")),
            source.get("geometry_member"),
        )
    except Exception:
        return None

    def locate(column: float, row: float) -> Optional[tuple[float, float]]:
        try:
            return grid.lonlat_at(column * decimation, row * decimation)
        except Exception:
            return None

    return locate


def _nac_geolocator(view: SceneView, manager) -> Optional[Geolocator]:
    """NAC pixel to lon/lat, through the SPICE pushbroom camera model."""
    reference = manager.get_nac_reference()
    if reference is None:
        return None

    kernels = manager.get_spice_kernels()
    if kernels is None or not kernels.is_complete:
        return None

    try:
        from ..planetary.spice_adapter import NacCameraModel, parse_nac_label

        observation = parse_nac_label(
            reference["label_path"].read_text("utf-8", "replace"), reference["product_id"]
        )
        camera = NacCameraModel(observation, kernels)
    except Exception:
        return None

    def locate(column: float, row: float) -> Optional[tuple[float, float]]:
        # The camera model is addressed (line, sample); a raster is (row, col).
        return camera.ground_point(row, column)

    return locate


def geolocator_for(view: SceneView, manager) -> Optional[Geolocator]:
    """
    A pixel-to-ground function for this view, or None if it has no geometry.

    Dispatch is on the sensor because that is what determines which geometry
    exists, not on anything recorded in the view itself.
    """
    sensor = (view.sensor or "").upper()
    if "OHRC" in sensor:
        return _ohrc_geolocator(view, manager)
    if "NAC" in sensor:
        return _nac_geolocator(view, manager)
    return None
