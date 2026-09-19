"""
Finding where two products actually share ground, before matching them.

A feature matcher given two images that mostly do not overlap does not report
"these barely overlap". It reports whatever its descriptors happen to agree on,
which is nothing, and the run fails at the matching stage with no explanation.
That is exactly how `ch2_ohrc_vs_lronac` failed: a 22 km source strip was
matched against a 75 km reference strip whose shared ground is a diagonal band
across part of each.

With mission geometry on both sides the shared band can be computed rather than
searched for. The source's own geometry gives longitude and latitude for its
pixels; the reference's camera model turns those back into reference pixels.
The bounding box of the result is the only part of the reference worth reading.

This is a prior, not a solution: it says where to look, and the matcher still
has to find correspondences there. It is deliberately generous - a margin is
added - because a prior that crops away real overlap is worse than one that
includes some extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class OverlapWindow:
    """A read window into the reference, in its native pixels."""

    col_off: int
    row_off: int
    width: int
    height: int
    method: str
    n_projected: int
    n_tested: int
    note: str
    source_window: Optional[tuple[int, int, int, int]] = None

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.col_off, self.row_off, self.width, self.height)

    def to_dict(self) -> dict:
        return {
            "col_off": self.col_off,
            "row_off": self.row_off,
            "width": self.width,
            "height": self.height,
            "method": self.method,
            "points_projected": self.n_projected,
            "points_tested": self.n_tested,
            "source_window": list(self.source_window) if self.source_window else None,
            "note": self.note,
        }


def _archive_and_member(ref: str) -> tuple[Optional[Path], Optional[str]]:
    if not ref.startswith("/vsizip/"):
        return None, None
    rest = ref[len("/vsizip/") :]
    index = rest.lower().find(".zip/")
    if index == -1:
        return None, None
    return Path(rest[: index + 4]), rest[index + 5 :]


def ohrc_lonlat_sampler(ref: str) -> Optional[tuple[Callable, tuple[int, int]]]:
    """
    A (column, row) -> (longitude, latitude) function for an OHRC browse product.

    Returns the sampler and the browse raster's (height, width), or None when
    this reference is not an OHRC browse product with a NAV grid.
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

        def sample(column: float, row: float):
            return grid.lonlat_at(column * decimation, row * decimation)

        return sample, image.shape[:2]
    except Exception:
        return None


def nac_reference_projector(ref: str, data_manager=None) -> Optional[Callable]:
    """A (longitude, latitude) -> (row, column) function for an LROC NAC product."""
    path = Path(ref)
    if "nac" not in str(path).lower():
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

        def project(longitude: float, latitude: float):
            return camera.image_point(longitude, latitude)

        return project
    except Exception:
        return None


def compute_overlap_window(
    source_ref: str,
    reference_ref: str,
    reference_shape: tuple[int, int],
    data_manager=None,
    grid: int = 11,
    margin_fraction: float = 0.06,
    min_points: int = 4,
) -> Optional[OverlapWindow]:
    """
    Where the source lands in the reference, as a read window.

    A grid of source pixels is pushed through both geometries. Points that miss
    the reference are expected and are simply not counted: two strips crossing
    at an angle share a band, not a rectangle.

    Returns None when geometry is unavailable on either side or too little of
    the source lands in the reference to trust the result.
    """
    sampler = ohrc_lonlat_sampler(source_ref)
    projector = nac_reference_projector(reference_ref, data_manager)
    if sampler is None or projector is None:
        return None

    sample, (source_height, source_width) = sampler

    rows: list[float] = []
    cols: list[float] = []
    source_rows: list[float] = []
    source_cols: list[float] = []
    tested = 0

    for v in np.linspace(0, source_height - 1, grid):
        for u in np.linspace(0, source_width - 1, grid):
            tested += 1
            try:
                longitude, latitude = sample(float(u), float(v))
                pixel = projector(longitude, latitude)
            except Exception:
                continue
            if pixel is None:
                continue
            rows.append(pixel[0])
            cols.append(pixel[1])
            source_rows.append(float(v))
            source_cols.append(float(u))

    if len(rows) < min_points:
        return None

    height, width = reference_shape
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)

    row_margin = (row_max - row_min) * margin_fraction
    col_margin = (col_max - col_min) * margin_fraction

    row_off = int(max(0, np.floor(row_min - row_margin)))
    col_off = int(max(0, np.floor(col_min - col_margin)))
    row_end = int(min(height, np.ceil(row_max + row_margin)))
    col_end = int(min(width, np.ceil(col_max + col_margin)))

    if row_end - row_off < 16 or col_end - col_off < 16:
        return None

    # The part of the source that actually landed, so the moving image can be
    # trimmed to the shared band too rather than carrying dead area into the
    # matcher.
    source_window = (
        int(max(0, np.floor(min(source_cols)))),
        int(max(0, np.floor(min(source_rows)))),
        int(min(source_width, np.ceil(max(source_cols))) - max(0, np.floor(min(source_cols)))),
        int(min(source_height, np.ceil(max(source_rows))) - max(0, np.floor(min(source_rows)))),
    )

    fraction = len(rows) / max(tested, 1)
    return OverlapWindow(
        col_off=col_off,
        row_off=row_off,
        width=col_end - col_off,
        height=row_end - row_off,
        method="spice-projected source footprint",
        n_projected=len(rows),
        n_tested=tested,
        source_window=source_window,
        note=(
            f"{len(rows)} of {tested} source grid points land in the reference "
            f"({fraction * 100:.0f}%). Reference read restricted to "
            f"{col_end - col_off} x {row_end - row_off} px of {width} x {height}, "
            f"a {(width * height) / max((col_end - col_off) * (row_end - row_off), 1):.1f}x reduction."
        ),
    )
