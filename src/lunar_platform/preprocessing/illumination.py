"""
Illumination-robust representations for cross-date lunar imagery.

The two products in this project were acquired two days apart with different
sun geometry, so the same crater can appear as a bright arc in one image and a
shadow in the other. Matching raw intensity across that is unreliable, which is
why the pipeline offers explicit representations rather than assuming pixel
values are comparable.

These mitigate illumination difference. They do not remove it: a slope lit from
opposite sides genuinely carries different information, and no normalisation
recovers what was never observed.
"""

from __future__ import annotations

import cv2
import numpy as np


def clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation."""
    operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return operator.apply(image)


def local_contrast_normalize(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Divide out the low-frequency illumination gradient.

    Large-scale brightness follows the sun; local texture follows the terrain.
    Removing the former keeps the part that should be common to both images.
    """
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
    normalized = image.astype(np.float32) - blurred
    lo, hi = np.percentile(normalized, [1, 99])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    return (np.clip((normalized - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def gradient_magnitude(image: np.ndarray, blur_sigma: float = 1.2) -> np.ndarray:
    """
    Sobel gradient magnitude.

    Edge strength survives an illumination sign flip better than intensity,
    though edge *direction* does not, which is why magnitude is used alone.
    """
    smoothed = cv2.GaussianBlur(image.astype(np.float32), (0, 0), blur_sigma)
    dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(dx, dy)

    hi = np.percentile(magnitude, 99)
    if hi <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    return (np.clip(magnitude / hi, 0, 1) * 255).astype(np.uint8)


def _log_gabor_bank(
    rows: int, columns: int, n_scales: int, min_wavelength: float, multiplier: float, sigma_ratio: float
) -> list[np.ndarray]:
    """Radial log-Gabor filters in the frequency domain, DC removed."""
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float32)
    y = (y - rows / 2.0) / rows
    x = (x - columns / 2.0) / columns
    radius = np.fft.ifftshift(np.hypot(x, y))
    radius[0, 0] = 1.0  # avoid log(0); the DC term is zeroed below anyway

    filters = []
    for scale in range(n_scales):
        wavelength = min_wavelength * (multiplier ** scale)
        f0 = 1.0 / wavelength
        bank = np.exp(-((np.log(radius / f0)) ** 2) / (2 * np.log(sigma_ratio) ** 2))
        bank[0, 0] = 0.0
        filters.append(bank.astype(np.float32))
    return filters


def phase_congruency(
    image: np.ndarray,
    n_scales: int = 4,
    min_wavelength: float = 3.0,
    multiplier: float = 2.1,
    sigma_ratio: float = 0.55,
) -> np.ndarray:
    """
    Monogenic phase congruency: structure without brightness.

    Phase congruency measures where the local Fourier components line up in
    phase, which happens at edges and ridges regardless of how strongly they
    are lit. It is dimensionless and normalised by local energy, so it does not
    care that one image is a grazing-sun scene of deep shadows and the other is
    the same ground lit five degrees higher.

    That is the case this project actually has. The two products here were
    acquired at sun elevations of 2.2 and 7.5 degrees, so the same crater casts
    shadows differing by about 3.4x in length. Every intensity-based
    representation keys on those shadow edges, and shadow edges move. Phase
    congruency keys on the rim structure that does not.

    Implemented over the monogenic signal (log-Gabor radial filters plus the
    Riesz transform) rather than a bank of oriented filters: it is isotropic,
    which matters when the two scenes are lit from different bearings.
    """
    data = image.astype(np.float32)
    rows, columns = data.shape[:2]
    spectrum = np.fft.fft2(data)

    # Riesz transform kernels, the odd-symmetric pair of the monogenic signal.
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float32)
    y = (y - rows / 2.0) / rows
    x = (x - columns / 2.0) / columns
    radius = np.fft.ifftshift(np.hypot(x, y))
    radius[0, 0] = 1.0
    riesz_x = np.fft.ifftshift(x) / radius
    riesz_y = np.fft.ifftshift(y) / radius

    sum_even = np.zeros((rows, columns), np.float32)
    sum_odd_x = np.zeros((rows, columns), np.float32)
    sum_odd_y = np.zeros((rows, columns), np.float32)
    sum_amplitude = np.zeros((rows, columns), np.float32)

    for bank in _log_gabor_bank(rows, columns, n_scales, min_wavelength, multiplier, sigma_ratio):
        filtered = spectrum * bank
        even = np.real(np.fft.ifft2(filtered))
        odd_x = np.real(np.fft.ifft2(filtered * riesz_x * 1j))
        odd_y = np.real(np.fft.ifft2(filtered * riesz_y * 1j))

        sum_even += even
        sum_odd_x += odd_x
        sum_odd_y += odd_y
        sum_amplitude += np.sqrt(even**2 + odd_x**2 + odd_y**2)

    energy = np.sqrt(sum_even**2 + sum_odd_x**2 + sum_odd_y**2)
    # Normalising by summed amplitude is what makes this a phase measure rather
    # than an energy measure, and therefore contrast invariant.
    congruency = energy / (sum_amplitude + 1e-6)

    return (np.clip(congruency, 0.0, 1.0) * 255).astype(np.uint8)


def shadow_mask(image: np.ndarray, percentile: float = 12.0) -> np.ndarray:
    """
    Pixels carrying surface information, as a uint8 mask.

    At grazing illumination a large fraction of the frame is cast shadow, which
    is not dark ground but an absence of observation. Features found there
    describe the shadow's outline, which moves with the sun, so they cannot
    correspond between two dates.
    """
    cutoff = np.percentile(image, percentile)
    return ((image > cutoff).astype(np.uint8)) * 255


def phase_congruency_masked(image: np.ndarray) -> np.ndarray:
    """Phase congruency with cast shadow suppressed."""
    congruency = phase_congruency(image)
    return cv2.bitwise_and(congruency, congruency, mask=shadow_mask(image))


# Registered by name so the pipeline can be pointed at a different
# representation from configuration without code changes.
REPRESENTATIONS = {
    "raw": lambda img: img,
    "clahe": clahe,
    "local_contrast": local_contrast_normalize,
    "gradient": gradient_magnitude,
    "phase_congruency": phase_congruency,
    "phase_congruency_masked": phase_congruency_masked,
}


def apply_representation(image: np.ndarray, name: str) -> np.ndarray:
    if name not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown illumination representation '{name}'. "
            f"Available: {', '.join(sorted(REPRESENTATIONS))}"
        )
    return REPRESENTATIONS[name](image)
