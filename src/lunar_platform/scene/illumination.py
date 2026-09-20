"""
Lighting geometry at a point on the Moon, computed rather than read.

Neither product in this project states its illumination usefully. The LROC NAC
label carries no angles at all, and the OHRC label gives a sun azimuth and
nothing else. Since illumination is the property that decides whether two views
can be matched, it has to come from somewhere reliable, and that is the
ephemeris.

Sun elevation and incidence need only the Sun's position and the surface
normal, so they are available for any product whose acquisition time and
ground point are known - including one whose spacecraft has no SPK loaded.
Emission and phase depend on where the camera was, so they are reported only
when an observer can actually be located, and left as None otherwise rather
than filled with a plausible number.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..planetary.spice_adapter import LUNAR_FRAME, SpiceUnavailable, _spiceypy
from .models import Illumination

MOON_RADIUS_KM = 1737.4


def illumination_at(
    utc: str,
    longitude: float,
    latitude: float,
    observer: Optional[str] = None,
) -> Illumination:
    """
    Sun geometry at a surface point, in degrees.

    `observer` names a body with a loaded SPK - "LRO" for the NAC frames. Left
    out, emission and phase are reported as None, which is the honest answer
    for a spacecraft this project carries no ephemeris for.
    """
    try:
        spice = _spiceypy()
    except SpiceUnavailable:
        return Illumination(source="unavailable")

    try:
        et = spice.str2et(utc.replace("Z", ""))
        point = np.asarray(
            spice.latrec(MOON_RADIUS_KM, math.radians(longitude), math.radians(latitude))
        )

        # Sun direction in the body-fixed frame, then resolved into the local
        # horizontal frame at the point so it becomes an azimuth and elevation.
        sun, _ = spice.spkpos("SUN", et, LUNAR_FRAME, "LT+S", "MOON")
        to_sun = np.asarray(sun) - point
        norm = np.linalg.norm(to_sun)
        if norm <= 0:
            return Illumination(source="unavailable")
        to_sun = to_sun / norm

        up = point / np.linalg.norm(point)
        # Any vector not parallel to up gives a consistent east; the pole is
        # the degenerate case and is handled by falling back to the x axis.
        polar = np.array([0.0, 0.0, 1.0])
        east = np.cross(polar, up)
        if np.linalg.norm(east) < 1e-9:
            east = np.cross(np.array([1.0, 0.0, 0.0]), up)
        east = east / np.linalg.norm(east)
        north = np.cross(up, east)

        elevation = math.degrees(math.asin(float(np.clip(np.dot(to_sun, up), -1.0, 1.0))))
        azimuth = math.degrees(
            math.atan2(float(np.dot(to_sun, east)), float(np.dot(to_sun, north)))
        ) % 360.0

        emission = phase = None
        incidence = 90.0 - elevation
        if observer:
            try:
                _, _, phase_rad, incidence_rad, emission_rad = spice.ilumin(
                    "ELLIPSOID", "MOON", et, LUNAR_FRAME, "CN+S", observer, point
                )
                incidence = math.degrees(incidence_rad)
                emission = round(math.degrees(emission_rad), 3)
                phase = round(math.degrees(phase_rad), 3)
            except Exception:
                # The observer has no ephemeris here; sun-only geometry stands.
                pass

        return Illumination(
            sun_azimuth_deg=round(azimuth, 3),
            sun_elevation_deg=round(elevation, 3),
            incidence_deg=round(incidence, 3),
            emission_deg=emission,
            phase_deg=phase,
            source="spice",
        )
    except Exception:
        return Illumination(source="unavailable")
