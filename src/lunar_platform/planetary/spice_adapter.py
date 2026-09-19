"""
SPICE geometry for LROC NAC.

This is what the project previously could not do. The NAC label carries
spacecraft clock counts and an orbit number and no geographic metadata at all,
so placing that image on the Moon needs the mission's own ephemeris, attitude
and instrument geometry. With the kernels under data/raw/spice present, every
pixel of the NAC image can be turned into a longitude and latitude, and the
question of whether the OHRC and NAC products overlap on the ground - which the
data/raw README records as unanswerable - can be answered.

NAC is a linescan (pushbroom) camera: the detector is a single row of 5064
pixels and the image is built up as the spacecraft moves, so every line has its
own time and its own pose. There is no single camera position for the image,
which is why a frame-camera model cannot be substituted here.

What is modelled and what is not, stated rather than assumed:

* the pushbroom timing, the spacecraft trajectory and attitude, the instrument
  alignment and the light-time and stellar aberration corrections are all taken
  from the kernels;
* the surface is the reference ellipsoid. Intersecting against a DEM instead
  would move a point by roughly the local relief times the tangent of the
  emission angle, which near nadir is small but not zero;
* optical distortion is applied as the radial term the instrument kernel
  carries. It reaches about 0.6% of field radius at the edge of the detector.

Kernels are mission data and live under data/raw, which is read-only. Nothing
here writes to them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# The reference frame lunar cartographic products use. IAU_MOON is a different
# realisation and differs from it by of order 100 m on the ground, so mixing
# the two would quietly introduce that as a systematic error.
LUNAR_FRAME = "MOON_ME"

# Order matters: text kernels that reference others must come after them.
KERNEL_SUFFIX_ORDER = {
    ".tls": 0,   # leapseconds
    ".tpc": 1,   # planetary constants
    ".tsc": 2,   # spacecraft clock
    ".tf": 3,    # frames
    ".ti": 4,    # instrument
    ".bsp": 5,   # ephemeris
    ".bpc": 6,   # binary orientation
    ".bc": 7,    # attitude
}


class SpiceUnavailable(RuntimeError):
    """Raised when SPICE geometry is asked for but cannot be provided."""


def _spiceypy():
    try:
        import spiceypy
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SpiceUnavailable(
            "spiceypy is not installed, so SPICE geometry is unavailable. "
            "Install it with: pip install spiceypy"
        ) from exc
    return spiceypy


@dataclass
class SpiceKernelSet:
    """
    The kernels discovered under a directory, in load order.

    Discovery is by suffix rather than by filename, so a bundle laid out any
    way at all is picked up as long as the kernels are there.
    """

    root: Path
    paths: list[Path] = field(default_factory=list)
    loaded: bool = False

    @classmethod
    def discover(cls, root: Path) -> "SpiceKernelSet":
        root = Path(root)
        if not root.exists():
            return cls(root=root, paths=[])

        found = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in KERNEL_SUFFIX_ORDER
        ]
        found.sort(key=lambda p: (KERNEL_SUFFIX_ORDER[p.suffix.lower()], p.name))
        return cls(root=root, paths=found)

    def kinds(self) -> dict:
        counts: dict[str, int] = {}
        for path in self.paths:
            counts[path.suffix.lower()] = counts.get(path.suffix.lower(), 0) + 1
        return counts

    @property
    def is_complete(self) -> bool:
        """
        Whether the set can support a NAC geolocation.

        Every one of these is load-bearing: without the leapseconds and clock
        kernels a time cannot be converted, without the ephemeris and attitude
        the spacecraft cannot be placed or pointed, and without the frames and
        instrument kernels the camera cannot be related to the spacecraft.
        """
        present = self.kinds()
        return all(
            present.get(suffix)
            for suffix in (".tls", ".tsc", ".tf", ".ti", ".bsp", ".bc")
        )

    def missing(self) -> list[str]:
        names = {
            ".tls": "leapseconds (lsk)",
            ".tsc": "spacecraft clock (sclk)",
            ".tf": "frames (fk)",
            ".ti": "instrument (ik)",
            ".bsp": "ephemeris (spk)",
            ".bc": "attitude (ck)",
        }
        present = self.kinds()
        return [label for suffix, label in names.items() if not present.get(suffix)]

    def load(self) -> None:
        if self.loaded:
            return
        if not self.paths:
            raise SpiceUnavailable(f"No SPICE kernels found under {self.root}")
        spice = _spiceypy()
        for path in self.paths:
            spice.furnsh(str(path))
        self.loaded = True

    def unload(self) -> None:
        if not self.loaded:
            return
        spice = _spiceypy()
        for path in self.paths:
            try:
                spice.unload(str(path))
            except Exception:
                pass
        self.loaded = False

    def describe(self) -> dict:
        return {
            "root": str(self.root),
            "n_kernels": len(self.paths),
            "kinds": self.kinds(),
            "complete": self.is_complete,
            "missing": self.missing(),
            "kernels": [p.name for p in self.paths],
        }


@dataclass
class NacObservation:
    """The timing and shape of one NAC image, read from its PDS4 label."""

    product_id: str
    start_time: str
    stop_time: str
    lines: int
    samples: int
    line_exposure_s: float
    crosstrack_summing: int
    instrument_frame: str

    @property
    def duration_s(self) -> float:
        return self.lines * self.line_exposure_s


def parse_nac_label(xml: str, product_id: str = "") -> NacObservation:
    """
    Pull the pushbroom timing and geometry out of a NAC PDS4 label.

    The summing factor matters as much as the dimensions: a 2x summed image has
    half as many samples as the detector has pixels, and treating an image
    sample as a detector pixel would halve the field of view.
    """

    def first(tag: str) -> Optional[str]:
        match = re.search(rf"<(?:\w+:)?{tag}[^>]*>([^<]+)</", xml)
        return match.group(1).strip() if match else None

    axes = re.findall(r"<axis_name>([^<]+)</axis_name>\s*<elements>(\d+)</elements>", xml)
    dimensions = {name.strip().lower(): int(count) for name, count in axes}

    exposure_ms = first("line_exposure_duration")
    summing = first("crosstrack_summing")
    detector = (first("instrument_id") or "").upper()

    # NAC has a left and a right camera with separate frames and alignments.
    if "NACR" in detector or product_id.upper().endswith("RC"):
        frame = "LRO_LROCNACR"
    else:
        frame = "LRO_LROCNACL"

    return NacObservation(
        product_id=product_id or (first("logical_identifier") or "").split(":")[-1],
        start_time=first("start_date_time") or "",
        stop_time=first("stop_date_time") or "",
        lines=int(dimensions.get("line", 0)),
        samples=int(dimensions.get("sample", 0)),
        line_exposure_s=float(exposure_ms) / 1000.0 if exposure_ms else 0.0,
        crosstrack_summing=int(summing) if summing else 1,
        instrument_frame=frame,
    )


class NacCameraModel:
    """
    Pushbroom camera model for one NAC image.

    Forward: an image pixel becomes a point on the Moon. Inverse: a point on
    the Moon becomes an image pixel, found by solving for the line whose
    detector row sees it, since a linescan camera images a given point at
    exactly one time.
    """

    def __init__(self, observation: NacObservation, kernels: SpiceKernelSet):
        self.observation = observation
        self.kernels = kernels
        kernels.load()

        spice = _spiceypy()
        self._spice = spice
        self._et0 = spice.str2et(observation.start_time)

        frame_id = spice.bodn2c(observation.instrument_frame)
        self.focal_length_mm = float(spice.gdpool(f"INS{frame_id}_FOCAL_LENGTH", 0, 1)[0])
        self.pixel_pitch_mm = float(spice.gdpool(f"INS{frame_id}_PIXEL_PITCH", 0, 1)[0])
        self.boresight_sample = float(
            spice.gdpool(f"INS{frame_id}_BORESIGHT_SAMPLE", 0, 1)[0]
        )
        try:
            self.distortion_k = float(spice.gdpool(f"INS{frame_id}_OD_K", 0, 1)[0])
        except Exception:
            self.distortion_k = 0.0

    # -- timing --------------------------------------------------------------

    def line_et(self, line: float) -> float:
        """Ephemeris time at which a given image line was read out."""
        return self._et0 + line * self.observation.line_exposure_s

    # -- optics --------------------------------------------------------------

    def focal_plane_y(self, sample: float) -> float:
        """
        Focal-plane cross-track coordinate, in millimetres, for an image sample.

        The summing factor is undone first, so the value corresponds to the
        centre of the group of detector pixels the image sample was built from.
        """
        summing = max(1, self.observation.crosstrack_summing)
        detector_sample = summing * sample + (summing - 1) / 2.0
        return (detector_sample - self.boresight_sample) * self.pixel_pitch_mm

    def look_vector(self, sample: float) -> list[float]:
        """
        Look direction for an image sample, in the instrument frame.

        The detector row runs along the frame's y axis and the spacecraft
        travels along x, which is the opposite of what the instrument kernel's
        TRANSX/TRANSY names suggest. The kernel's own inverse transforms settle
        it: ITRANSS makes sample a function of y and ITRANSL makes line a
        function of x. Getting this backwards still yields a plausible-looking
        footprint of about the right size, because it lays the swath along the
        track instead of across it.

        The kernel's radial term is then applied; it is small near the
        boresight and reaches a few tens of microradians at the detector edge.
        """
        y = self.focal_plane_y(sample)

        if self.distortion_k:
            y = y * (1.0 + self.distortion_k * y * y)

        return [0.0, y, self.focal_length_mm]

    # -- forward -------------------------------------------------------------

    def ground_point(self, line: float, sample: float) -> Optional[tuple[float, float]]:
        """
        Longitude and latitude a pixel sees, or None where it misses the Moon.

        A miss is a real outcome at the limb and is reported rather than
        clamped to the nearest point on the surface.
        """
        spice = self._spice
        try:
            point, _, _ = spice.sincpt(
                "ELLIPSOID",
                "MOON",
                self.line_et(line),
                LUNAR_FRAME,
                "CN+S",
                "LRO",
                self.observation.instrument_frame,
                self.look_vector(sample),
            )
        except Exception:
            return None

        _, longitude, latitude = spice.reclat(point)
        return float(np.degrees(longitude)), float(np.degrees(latitude))

    def apparent_direction(self, line: float, point_bodyfixed: np.ndarray) -> np.ndarray:
        """
        Apparent direction from the spacecraft to a fixed ground point.

        This goes through spkcpt rather than differencing positions by hand so
        that it carries exactly the light-time and stellar aberration
        convention sincpt uses on the way out. Doing the subtraction manually
        omits stellar aberration, which is about 1e-4 radians here - a constant
        seven-line bias in the recovered position, small enough to look like
        rounding and far too large to be one.
        """
        spice = self._spice
        state, _ = spice.spkcpt(
            trgpos=point_bodyfixed,
            trgctr="MOON",
            trgref=LUNAR_FRAME,
            et=self.line_et(line),
            outref=self.observation.instrument_frame,
            refloc="OBSERVER",
            abcorr="CN+S",
            obsrvr="LRO",
        )
        return np.asarray(state[:3])

    def _along_track_residual(self, line: float, point_bodyfixed: np.ndarray) -> float:
        """
        How far off the detector row a ground point falls, at a given line.

        The detector is one row, so a point is imaged at the instant this
        crosses zero. x is the along-track axis - the component that sweeps as
        the spacecraft moves. Expressed as an angle so the scale does not
        depend on altitude.
        """
        direction = self.apparent_direction(line, point_bodyfixed)
        return float(direction[0] / direction[2])

    # -- inverse -------------------------------------------------------------

    def image_point(
        self,
        longitude: float,
        latitude: float,
        tolerance_px: float = 0.01,
        edge_margin_px: float = 0.5,
    ) -> Optional[tuple[float, float]]:
        """
        The image pixel that sees a point on the Moon, or None if none does.

        Solves for the line whose detector row contains the point by bisection
        on the along-track residual, then reads the sample straight off the
        geometry. Returns None when the point is outside the image, which is
        the common case: the swath is a few kilometres wide.
        """
        spice = self._spice
        radii = spice.bodvrd("MOON", "RADII", 3)[1]
        point = np.asarray(spice.latrec(radii[0], np.radians(longitude), np.radians(latitude)))
        # A point on a sphere of the equatorial radius is close enough to seed
        # the search; the intercept below is what fixes the surface point.
        last_line = self.observation.lines - 1

        try:
            low_value = self._along_track_residual(0.0, point)
            high_value = self._along_track_residual(last_line, point)
        except Exception:
            return None

        if low_value * high_value > 0:
            return None  # the detector row never sweeps across this point

        low, high = 0.0, float(last_line)
        for _ in range(60):
            middle = 0.5 * (low + high)
            value = self._along_track_residual(middle, point)
            if low_value * value <= 0:
                high = middle
            else:
                low, low_value = middle, value
            if high - low < tolerance_px:
                break

        line = 0.5 * (low + high)

        direction = self.apparent_direction(line, point)
        if direction[2] <= 0:
            return None

        measured = self.focal_length_mm * direction[1] / direction[2]
        y_mm = measured
        if self.distortion_k:
            # Newton steps invert the small radial term applied on the way out.
            for _ in range(4):
                error = y_mm * (1.0 + self.distortion_k * y_mm * y_mm) - measured
                slope = 1.0 + 3.0 * self.distortion_k * y_mm * y_mm
                y_mm -= error / slope

        detector_sample = y_mm / self.pixel_pitch_mm + self.boresight_sample
        summing = max(1, self.observation.crosstrack_summing)
        sample = (detector_sample - (summing - 1) / 2.0) / summing

        # Half a pixel of slack at the border: a point imaged by the outermost
        # detector column solves to a sample a rounding error outside it, and
        # reporting that as "not in the image" would be wrong.
        if not (
            -edge_margin_px <= line <= last_line + edge_margin_px
            and -edge_margin_px <= sample <= self.observation.samples - 1 + edge_margin_px
        ):
            return None
        return float(line), float(sample)

    # -- products ------------------------------------------------------------

    def footprint(self) -> dict:
        """Corner and centre coordinates of the image on the ground."""
        last_line = self.observation.lines - 1
        last_sample = self.observation.samples - 1

        corners = {}
        for name, (line, sample) in {
            "upper_left": (0, 0),
            "upper_right": (0, last_sample),
            "lower_left": (last_line, 0),
            "lower_right": (last_line, last_sample),
        }.items():
            ground = self.ground_point(line, sample)
            corners[name] = (
                {"longitude": round(ground[0], 6), "latitude": round(ground[1], 6)}
                if ground
                else None
            )

        centre = self.ground_point(last_line / 2.0, last_sample / 2.0)
        placed = [c for c in corners.values() if c]

        return {
            "product_id": self.observation.product_id,
            "frame": LUNAR_FRAME,
            "instrument_frame": self.observation.instrument_frame,
            "lines": self.observation.lines,
            "samples": self.observation.samples,
            "crosstrack_summing": self.observation.crosstrack_summing,
            "start_time": self.observation.start_time,
            "corners": corners,
            "centre": (
                {"longitude": round(centre[0], 6), "latitude": round(centre[1], 6)}
                if centre
                else None
            ),
            "min_latitude": round(min(c["latitude"] for c in placed), 6) if placed else None,
            "max_latitude": round(max(c["latitude"] for c in placed), 6) if placed else None,
            "min_longitude": round(min(c["longitude"] for c in placed), 6) if placed else None,
            "max_longitude": round(max(c["longitude"] for c in placed), 6) if placed else None,
            "method": "SPICE pushbroom, ellipsoid intercept, CN+S aberration",
        }

    def ground_sampling_m(self, line: Optional[float] = None) -> float:
        """Cross-track ground sampling at a line, measured from the geometry."""
        if line is None:
            line = (self.observation.lines - 1) / 2.0
        middle = self.observation.samples // 2
        first = self.ground_point(line, middle)
        second = self.ground_point(line, middle + 1)
        if not first or not second:
            return float("nan")

        from ..io.ohrc_browse import great_circle_m

        return float(great_circle_m(first[1], first[0], second[1], second[0]))


def sector_coverage(camera: NacCameraModel, basemap, grid: int = 9) -> dict:
    """
    How much of a sector the NAC image actually covers.

    The data/raw README records that without SPICE there is no way to
    establish whether the OHRC and NAC products overlap on the ground, so a
    cross-instrument registration between them cannot be geographically
    verified. With the kernels loaded that question has an answer, and this
    computes it: every point of a grid over the sector is pushed through the
    NAC camera model and either lands in the image or does not.

    A bounding-box test would not do. Both products are narrow ribbons, and
    two ribbons can have overlapping bounding boxes while sharing no ground at
    all.
    """
    covered = 0
    total = 0
    lines: list[float] = []
    samples: list[float] = []

    for row in range(grid):
        for column in range(grid):
            x = basemap.width_m * column / (grid - 1)
            y = basemap.height_m * row / (grid - 1)
            # Stay a hair inside the raster so the lookup has pixels to read.
            lonlat = basemap.lonlat_at(
                min(x, basemap.width_m - 1e-3), min(y, basemap.height_m - 1e-3)
            )
            if lonlat is None:
                continue

            total += 1
            pixel = camera.image_point(*lonlat)
            if pixel is not None:
                covered += 1
                lines.append(pixel[0])
                samples.append(pixel[1])

    fraction = covered / total if total else 0.0
    return {
        "grid": grid,
        "points_tested": total,
        "points_covered": covered,
        "coverage_fraction": round(fraction, 4),
        "overlaps": covered > 0,
        "fully_covered": total > 0 and covered == total,
        "nac_line_range": [round(min(lines), 1), round(max(lines), 1)] if lines else None,
        "nac_sample_range": [round(min(samples), 1), round(max(samples), 1)] if samples else None,
        "note": (
            "Computed by pushing sector points through the SPICE camera model, not by "
            "comparing bounding boxes. Accuracy is that of the kernels and an ellipsoid "
            "surface; relief is not accounted for."
        ),
    }
