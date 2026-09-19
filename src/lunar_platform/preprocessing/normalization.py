"""
Radiometric normalisation shared by both sides of a registration pair.

Source and reference arrive with different dtypes and different dynamic ranges
(OHRC calibrated bytes against NAC calibrated int16). Feature detectors expect
8-bit input, so both are brought to a common representation before anything
photometric is attempted.

Percentile stretching rather than min/max: a single saturated or dead pixel
would otherwise compress the entire useful range.
"""

from __future__ import annotations

import numpy as np


def to_uint8(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    ignore_value: float | None = None,
) -> np.ndarray:
    """Stretch an array to 8-bit using a robust percentile range."""
    data = image.astype(np.float32)

    valid = np.isfinite(data)
    if ignore_value is not None:
        valid &= data != ignore_value
    if not valid.any():
        return np.zeros(data.shape, dtype=np.uint8)

    lo, hi = np.percentile(data[valid], [low_percentile, high_percentile])
    if hi <= lo:
        lo, hi = float(data[valid].min()), float(data[valid].max())
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)

    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255.0
    scaled[~valid] = 0
    return scaled.astype(np.uint8)


def valid_data_fraction(image: np.ndarray, ignore_value: float = 0.0) -> float:
    """
    Fraction of pixels carrying data. A mostly empty tile cannot register.

    NaN counts as missing: load_image marks a raster's declared nodata that
    way, and those pixels are absent data rather than dark ground.
    """
    if image.size == 0:
        return 0.0
    data = np.asarray(image)
    present = np.isfinite(data) if np.issubdtype(data.dtype, np.floating) else np.ones(data.shape, bool)
    present &= data != ignore_value
    return float(np.count_nonzero(present) / data.size)
