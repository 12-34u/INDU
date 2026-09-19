"""
Controlled validation of the registration pipeline on real lunar imagery.

The two real products in this project cannot be checked against each other:
the NAC label carries no geographic metadata, so without SPICE there is no way
to know whether the two strips even overlap, and a residual computed from an
unverifiable pairing means nothing.

This provides the check that *is* available. A real OHRC tile is transformed by
a known similarity transform and given a different illumination response, then
registered back against the original. Because the applied transform is known
exactly, the recovered transform can be scored against it - which measures the
pipeline itself on real lunar texture.

This validates the implementation. It does not validate cross-instrument
registration, which needs geometry this project does not yet have.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def apply_known_transform(
    image: np.ndarray,
    rotation_deg: float = 7.0,
    scale: float = 0.82,
    translation: tuple[float, float] = (14.0, -9.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a known similarity transform. Returns (warped, 3x3 matrix).

    Defaults exercise all three variations the problem statement names:
    rotation and translation for viewpoint, scale for resolution difference.
    """
    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)

    matrix_2x3 = cv2.getRotationMatrix2D(centre, rotation_deg, scale)
    matrix_2x3[0, 2] += translation[0]
    matrix_2x3[1, 2] += translation[1]

    warped = cv2.warpAffine(
        image, matrix_2x3, (width, height), flags=cv2.INTER_LINEAR, borderValue=0
    )

    matrix = np.eye(3)
    matrix[:2, :] = matrix_2x3
    return warped, matrix


def apply_illumination_change(
    image: np.ndarray, gamma: float = 1.7, gain: float = 0.78, bias: float = 26.0
) -> np.ndarray:
    """
    Simulate a different solar illumination response.

    A gamma and linear response change stands in for a different sun angle. It
    reproduces the radiometric difference between two acquisitions but not the
    geometric one: real shadows move and change shape, and nothing here does
    that. Passing this check is necessary, not sufficient.
    """
    normalized = image.astype(np.float32) / 255.0
    adjusted = np.power(normalized, gamma) * 255.0 * gain + bias
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def transform_error(
    recovered: np.ndarray,
    truth: np.ndarray,
    width: int,
    height: int,
    n_samples: int = 25,
) -> dict:
    """
    Compare two transforms by how differently they move a grid of points.

    Comparing matrix entries directly would weight translation and rotation
    arbitrarily; displacement in pixels is what actually matters.
    """
    xs = np.linspace(0, width - 1, int(math.sqrt(n_samples)))
    ys = np.linspace(0, height - 1, int(math.sqrt(n_samples)))
    grid = np.array([[x, y, 1.0] for y in ys for x in xs])

    def project(matrix: np.ndarray) -> np.ndarray:
        out = (matrix @ grid.T).T
        return out[:, :2] / out[:, 2:]

    errors = np.linalg.norm(project(recovered) - project(truth), axis=1)
    return {
        "mean_px": round(float(errors.mean()), 4),
        "median_px": round(float(np.median(errors)), 4),
        "max_px": round(float(errors.max()), 4),
        "rmse_px": round(float(np.sqrt(np.mean(errors**2))), 4),
        "n_samples": int(len(errors)),
    }
