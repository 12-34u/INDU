"""
Resolving what one pixel of a product covers on the ground.

Registration across missions fails at the first step if the two images are not
at the same ground scale, and the scale cannot be taken from a label: the OHRC
bundle states a nominal pixel_resolution of 0.24 m while its own NAV grid
measures about 0.285 m, a 19% error. Applied to a 2 km strip that is hundreds
of metres of accumulated mismatch.

So each product is asked how big its pixels really are, in this order:

1. the raster's own georeferencing, when it has any - generic, and the right
   answer for any properly georeferenced product;
2. the product's mission geometry - the OHRC NAV grid, the LROC NAC camera
   model through SPICE - which is where these two particular products keep it;
3. the label's nominal resolution, which is a stated intention rather than a
   measurement;
4. nothing, reported as unknown rather than guessed.

A caller can always override with an explicit value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio


@dataclass
class GroundScale:
    """Metres per pixel for one product, and where the number came from."""

    gsd_m: Optional[float]
    method: str  # "georeference" | "nav_grid" | "spice" | "label" | "explicit" | "unknown"
    detail: str = ""

    @property
    def known(self) -> bool:
        return self.gsd_m is not None and self.gsd_m > 0

    def to_dict(self) -> dict:
        return {
            "gsd_m": round(self.gsd_m, 5) if self.known else None,
            "method": self.method,
            "detail": self.detail,
        }


@dataclass
class ScalePlan:
    """How to bring a pair to one working scale."""

    working_gsd_m: Optional[float]
    source: GroundScale
    reference: GroundScale
    source_out_shape: Optional[tuple[int, int]]  # (height, width)
    reference_out_shape: Optional[tuple[int, int]]
    equalised: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "working_gsd_m": round(self.working_gsd_m, 5) if self.working_gsd_m else None,
            "equalised": self.equalised,
            "source_scale": self.source.to_dict(),
            "reference_scale": self.reference.to_dict(),
            "note": self.note,
        }


def _archive_and_member(ref: str) -> tuple[Optional[Path], Optional[str]]:
    """Split a /vsizip/<archive>/<member> reference, if that is what this is."""
    if not ref.startswith("/vsizip/"):
        return None, None
    rest = ref[len("/vsizip/") :]
    marker = ".zip/"
    index = rest.lower().find(marker)
    if index == -1:
        return None, None
    return Path(rest[: index + 4]), rest[index + 5 :]


def _from_georeference(ref: str) -> Optional[GroundScale]:
    """Pixel size straight off the raster, when it carries a projected CRS."""
    try:
        with rasterio.open(ref) as src:
            crs = src.crs
            if crs is None or not crs.is_projected:
                return None
            size = float(abs(src.transform[0]))
            if size <= 0:
                return None
            # Projected units are metres for every lunar product we handle, but
            # a degree-valued transform would be off by a factor of 30000.
            if size > 1e5:
                return None
            return GroundScale(size, "georeference", f"from the raster's own transform ({crs.to_string()[:40]})")
    except Exception:
        return None


def _from_ohrc_nav_grid(ref: str) -> Optional[GroundScale]:
    """
    Measured scale for an OHRC browse product, from the bundle's NAV grid.

    The browse raster is a decimation of the full image, so the grid's
    full-resolution sampling is scaled by that decimation factor.
    """
    archive, member = _archive_and_member(ref)
    if archive is None or member is None or "brw" not in member.lower():
        return None

    try:
        from ..io.ohrc_browse import browse_decimation, load_browse_image, load_geometry_grid
        from ..io.raw_inventory import inspect_archive

        product = inspect_archive(archive, "source")
        geometry = product.member_by_kind("geometry")
        if geometry is None:
            return None

        grid = load_geometry_grid(archive, geometry.name)
        image = load_browse_image(archive, member)
        decimation = browse_decimation(image.shape, product.label)

        across, along = grid.ground_sampling_m(
            (image.shape[1] / 2.0) * decimation, (image.shape[0] / 2.0) * decimation
        )
        full_gsd = float((across + along) / 2.0)
        return GroundScale(
            full_gsd * decimation,
            "nav_grid",
            f"measured from the NAV grid ({full_gsd:.4f} m/full px, {decimation:.1f}x decimated browse)",
        )
    except Exception:
        return None


def _from_spice_nac(ref: str, data_manager=None) -> Optional[GroundScale]:
    """Measured scale for an LROC NAC product, from its SPICE camera model."""
    path = Path(ref)
    if path.suffix.lower() not in (".img", ".tif", ".tiff") or "nac" not in str(path).lower():
        return None

    try:
        from ..core.data_manager import DataManager
        from ..planetary.spice_adapter import NacCameraModel, parse_nac_label

        manager = data_manager or DataManager()
        kernels = manager.get_spice_kernels()
        if kernels is None or not kernels.is_complete:
            return None

        reference = manager.get_nac_reference()
        if reference is None or Path(reference["image_path"]).name != path.name:
            return None

        observation = parse_nac_label(
            reference["label_path"].read_text("utf-8", "replace"), reference["product_id"]
        )
        camera = NacCameraModel(observation, kernels)
        gsd = camera.ground_sampling_m()
        if not np.isfinite(gsd) or gsd <= 0:
            return None
        return GroundScale(
            float(gsd),
            "spice",
            f"cross-track sampling from the SPICE camera model "
            f"({observation.crosstrack_summing}x summed)",
        )
    except Exception:
        return None


def _from_label(ref: str, label: Optional[dict]) -> Optional[GroundScale]:
    if not label:
        return None
    nominal = label.get("pixel_resolution")
    if nominal is None:
        return None
    try:
        value = float(nominal)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return GroundScale(value, "label", "nominal pixel_resolution stated in the label; not measured")


def resolve_ground_scale(
    ref: str,
    label: Optional[dict] = None,
    explicit_gsd_m: Optional[float] = None,
    data_manager=None,
) -> GroundScale:
    """Best available metres-per-pixel for a product reference."""
    if explicit_gsd_m and explicit_gsd_m > 0:
        return GroundScale(float(explicit_gsd_m), "explicit", "supplied by the caller")

    for resolver in (
        lambda: _from_georeference(ref),
        lambda: _from_ohrc_nav_grid(ref),
        lambda: _from_spice_nac(ref, data_manager),
        lambda: _from_label(ref, label),
    ):
        scale = resolver()
        if scale is not None and scale.known:
            return scale

    return GroundScale(None, "unknown", "no georeferencing, mission geometry or label resolution")


def plan_common_scale(
    source_ref: str,
    reference_ref: str,
    source_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    max_dimension: int,
    source_scale: GroundScale,
    reference_scale: GroundScale,
) -> ScalePlan:
    """
    Choose one working scale for the pair and the output shape for each image.

    The working scale is the COARSER of the two products. Resampling the finer
    image down discards detail it has; resampling the coarser one up would
    invent detail it does not, and a matcher keying on that invented texture
    finds correspondences that are not there.

    Shapes are (height, width), matching the arrays they describe.
    """
    if not (source_scale.known and reference_scale.known):
        # Fall back to the old per-image pixel budget, and say so.
        return ScalePlan(
            working_gsd_m=None,
            source=source_scale,
            reference=reference_scale,
            source_out_shape=None,
            reference_out_shape=None,
            equalised=False,
            note=(
                "Ground scale unknown for "
                + ("source" if not source_scale.known else "reference")
                + "; falling back to an independent pixel budget per image, which does "
                "not bring the pair to a common scale."
            ),
        )

    working = max(float(source_scale.gsd_m), float(reference_scale.gsd_m))

    def shape_at(native_shape, native_gsd, gsd):
        factor = float(native_gsd) / gsd
        return (
            max(8, int(round(native_shape[0] * factor))),
            max(8, int(round(native_shape[1] * factor))),
        )

    # Respect the memory budget by coarsening the shared scale, never by
    # letting the two images drift apart again.
    budget = float(max_dimension) ** 2
    for _ in range(24):
        source_out = shape_at(source_shape, source_scale.gsd_m, working)
        reference_out = shape_at(reference_shape, reference_scale.gsd_m, working)
        if (
            source_out[0] * source_out[1] <= budget
            and reference_out[0] * reference_out[1] <= budget
        ):
            break
        working *= 1.25

    ratio = max(source_scale.gsd_m, reference_scale.gsd_m) / min(
        source_scale.gsd_m, reference_scale.gsd_m
    )
    return ScalePlan(
        working_gsd_m=working,
        source=source_scale,
        reference=reference_scale,
        source_out_shape=source_out,
        reference_out_shape=reference_out,
        equalised=True,
        note=(
            f"Both images resampled to {working:.3f} m/px. Native scales differ by "
            f"{ratio:.2f}x (source {source_scale.gsd_m:.3f} via {source_scale.method}, "
            f"reference {reference_scale.gsd_m:.3f} via {reference_scale.method})."
        ),
    )
