"""Read-only access to source products under data/raw."""

from __future__ import annotations

import zipfile
from pathlib import Path

import rasterio

IMAGE_SUFFIXES = {".img", ".tif", ".tiff", ".cub", ".jp2", ".xml", ".lbl"}

# Labels are only openable when the detached data file sits beside them, so a
# real image member is always preferred when an archive contains both.
LABEL_SUFFIXES = {".xml", ".lbl"}


def resolve_raster_uri(path: str | Path) -> str:
    """
    Return a GDAL-readable URI for a source product.

    A .zip is addressed through /vsizip/ so the archive is read in place. The
    1.1 GB OHRC product is never expanded onto disk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source product not found: {path}")

    if path.suffix.lower() != ".zip":
        return str(path)

    with zipfile.ZipFile(path) as archive:
        members = [
            m
            for m in archive.infolist()
            if not m.is_dir() and Path(m.filename).suffix.lower() in IMAGE_SUFFIXES
        ]

    if not members:
        raise ValueError(
            f"No readable image member found inside {path}. "
            f"Expected one of: {', '.join(sorted(IMAGE_SUFFIXES))}"
        )

    images = [m for m in members if Path(m.filename).suffix.lower() not in LABEL_SUFFIXES]
    chosen = max(images or members, key=lambda m: m.file_size)
    return f"/vsizip/{path}/{chosen.filename}"


def open_source_raster(path: str | Path):
    """Open a source product read-only. The caller is responsible for closing it."""
    return rasterio.open(resolve_raster_uri(path))
