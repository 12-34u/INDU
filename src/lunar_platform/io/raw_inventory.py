"""
Discovery of immutable mission data under data/raw.

Everything here is read-only and cheap. Archives are inspected through their
central directory and labels are parsed in place, so nothing is ever extracted,
moved or rewritten. The 1.1 GB OHRC image member is catalogued but never read.

Values are taken from the PDS4 labels as written. Where a label does not state
something, the field is reported as None rather than guessed.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

IMAGE_MEMBER_SUFFIXES = {".img", ".tif", ".tiff", ".png", ".jp2"}
LABEL_SUFFIXES = {".xml", ".lbl"}
NAV_SUFFIXES = {".oat", ".oath", ".spm", ".lbr"}
# SPICE kernels. The text kernels (.tf frames, .ti instrument, .tsc clock) are
# as load-bearing as the binary ones: without them a camera cannot be related
# to its spacecraft, so leaving them out of discovery would report a kernel set
# as present that cannot actually place an image.
SPICE_SUFFIXES = {".bsp", ".bc", ".tls", ".tpc", ".bpc", ".tf", ".ti", ".tsc", ".tm"}

# A member is treated as directly usable only if reading it is cheap. The full
# OHRC image is 1.1 GB of DEFLATE, so it is catalogued and deliberately skipped.
CHEAP_MEMBER_BYTES = 64 * 1024 * 1024


@dataclass
class ArchiveMember:
    name: str
    size_bytes: int
    compress_type: int
    kind: str  # "image" | "browse" | "label" | "geometry" | "nav" | "other"

    @property
    def is_cheap(self) -> bool:
        return self.size_bytes <= CHEAP_MEMBER_BYTES

    @property
    def vsizip_uri_suffix(self) -> str:
        return self.name


@dataclass
class RawProduct:
    """One discovered product, archived or loose."""

    product_id: str
    role: str  # "source" | "reference"
    path: Path
    is_archive: bool
    mission: Optional[str] = None
    instrument: Optional[str] = None
    size_bytes: int = 0
    members: list[ArchiveMember] = field(default_factory=list)
    label: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def member_by_kind(self, kind: str) -> Optional[ArchiveMember]:
        for m in self.members:
            if m.kind == kind:
                return m
        return None

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "role": self.role,
            "path": str(self.path),
            "is_archive": self.is_archive,
            "mission": self.mission,
            "instrument": self.instrument,
            "size_bytes": self.size_bytes,
            "label": self.label,
            "notes": self.notes,
            "members": [
                {
                    "name": m.name,
                    "size_bytes": m.size_bytes,
                    "kind": m.kind,
                    "cheap_to_read": m.is_cheap,
                }
                for m in self.members
            ],
        }


def _classify_member(name: str, size: int) -> str:
    lower = name.lower()
    suffix = Path(lower).suffix

    if lower.endswith("/"):
        return "directory"
    # Labels are claimed first: every collection carries one, and a browse or
    # geometry label must not be mistaken for the product it describes.
    if suffix in LABEL_SUFFIXES:
        return "label"
    if suffix in NAV_SUFFIXES:
        return "nav"
    if "/browse/" in lower or "_brw_" in lower:
        return "browse"
    if "/geometry/" in lower or "_grd_" in lower:
        return "geometry"
    if suffix in IMAGE_MEMBER_SUFFIXES:
        return "image"
    return "other"


def _first_tag(xml: str, tag: str) -> Optional[str]:
    match = re.search(rf"<(?:\w+:)?{tag}[^>]*>([^<]+)</", xml)
    return match.group(1).strip() if match else None


def parse_pds4_label(xml: str) -> dict:
    """Pull the fields the registration workflow needs out of a PDS4 label."""
    axes = re.findall(r"<axis_name>([^<]+)</axis_name>\s*<elements>(\d+)</elements>", xml)
    dimensions = {name.strip().lower(): int(count) for name, count in axes}

    label = {
        "logical_identifier": _first_tag(xml, "logical_identifier"),
        "mission_name": _first_tag(xml, "name"),
        "start_date_time": _first_tag(xml, "start_date_time"),
        "stop_date_time": _first_tag(xml, "stop_date_time"),
        "processing_level": _first_tag(xml, "processing_level"),
        "data_type": _first_tag(xml, "data_type"),
        "lines": dimensions.get("line"),
        "samples": dimensions.get("sample"),
    }

    for key in (
        "pixel_resolution",
        "upper_left_latitude",
        "upper_left_longitude",
        "lower_right_latitude",
        "lower_right_longitude",
        "sun_azimuth",
        "incidence_angle",
        "emission_angle",
        "phase_angle",
    ):
        raw = _first_tag(xml, key)
        if raw is not None:
            try:
                label[key] = float(raw)
            except ValueError:
                label[key] = raw

    return {k: v for k, v in label.items() if v is not None}


def inspect_archive(path: Path, role: str) -> RawProduct:
    """Catalogue an archive from its central directory. Nothing is extracted."""
    product = RawProduct(
        product_id=path.stem,
        role=role,
        path=path,
        is_archive=True,
        size_bytes=path.stat().st_size,
    )

    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            product.members.append(
                ArchiveMember(
                    name=info.filename,
                    size_bytes=info.file_size,
                    compress_type=info.compress_type,
                    kind=_classify_member(info.filename, info.file_size),
                )
            )

        # The data-collection label is small; read it in place for real
        # metadata. Browse and geometry labels describe derived products.
        for member in product.members:
            lower = member.name.lower()
            is_data_label = lower.startswith("data/") or "/data/" in lower
            if member.kind == "label" and is_data_label:
                try:
                    product.label = parse_pds4_label(
                        archive.read(member.name).decode("utf-8", "replace")
                    )
                except Exception as exc:
                    product.notes.append(f"Could not parse label {member.name}: {exc}")
                break

    nav_members = [m.name for m in product.members if m.kind == "nav"]
    if nav_members:
        product.notes.append(f"Navigation ancillary present in archive: {len(nav_members)} files.")

    image = product.member_by_kind("image")
    browse = product.member_by_kind("browse")

    if image and not image.is_cheap:
        product.notes.append(
            f"Full image member is {image.size_bytes / 1e9:.2f} GB and compressed; "
            "it is not read at runtime."
        )
    if browse:
        product.notes.append(
            f"Browse product available ({browse.size_bytes / 1e6:.1f} MB) and readable "
            "without extraction."
        )
    if product.member_by_kind("geometry"):
        product.notes.append("Geometry grid present; usable as navigation metadata.")

    product.mission = product.label.get("mission_name")
    if "ohr" in path.name.lower():
        product.instrument = "OHRC"

    return product


def inspect_loose_product(path: Path, role: str) -> RawProduct:
    """Catalogue an unarchived product and its detached label if present."""
    product = RawProduct(
        product_id=path.stem,
        role=role,
        path=path,
        is_archive=False,
        size_bytes=path.stat().st_size,
    )

    for suffix in LABEL_SUFFIXES:
        label_path = path.with_suffix(suffix)
        if label_path.exists():
            try:
                product.label = parse_pds4_label(label_path.read_text("utf-8", "replace"))
            except Exception as exc:
                product.notes.append(f"Could not parse label {label_path.name}: {exc}")
            break

    if "nac" in str(path).lower():
        product.instrument = "LROC NAC"
        product.mission = product.mission or "Lunar Reconnaissance Orbiter"

    return product


def discover_raw(raw_dir: Path) -> dict:
    """
    Walk data/raw and report what is present.

    Source products are Chandrayaan-2; reference products are everything under
    reference/. Directory layout decides the role - filenames are not assumed.
    """
    raw_dir = Path(raw_dir)
    inventory: dict = {
        "raw_dir": str(raw_dir),
        "exists": raw_dir.exists(),
        "source_products": [],
        "reference_products": [],
        "nav_files": [],
        "dem_files": [],
        "spice_files": [],
    }

    if not raw_dir.exists():
        return inventory

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        relative = str(path.relative_to(raw_dir)).lower()
        suffix = path.suffix.lower()

        if "reference" in relative:
            role = "reference"
        elif "chandrayaan2" in relative or "ohrc" in relative:
            role = "source"
        else:
            role = "unknown"

        if suffix == ".zip":
            product = inspect_archive(path, role)
        elif suffix in IMAGE_MEMBER_SUFFIXES:
            product = inspect_loose_product(path, role)
        elif suffix in NAV_SUFFIXES:
            inventory["nav_files"].append(str(path))
            continue
        elif suffix in SPICE_SUFFIXES:
            inventory["spice_files"].append(str(path))
            continue
        else:
            continue

        if role == "reference":
            inventory["reference_products"].append(product)
        elif role == "source":
            inventory["source_products"].append(product)

        # NAV ancillary usually ships inside the product archive rather than
        # as loose files, so surface those too.
        for member in product.members:
            if member.kind == "nav":
                inventory["nav_files"].append(f"{path}!{member.name}")

    return inventory


def read_geometry_grid(archive_path: Path, member_name: str, max_rows: int | None = None) -> list[dict]:
    """
    Read the OHRC geometry grid from inside the archive.

    The grid gives Longitude/Latitude at sampled (Pixel, Scan) positions. It is
    the only geolocation this project currently has for the source image, and it
    is not a camera model: it supports approximate footprint work, not the
    precise geometry SPICE would provide.
    """
    rows: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, "utf-8"))
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                try:
                    rows.append(
                        {
                            "longitude": float(row["Longitude"]),
                            "latitude": float(row["Latitude"]),
                            "pixel": int(row["Pixel"]),
                            "scan": int(row["Scan"]),
                        }
                    )
                except (KeyError, ValueError):
                    continue
    return rows


def geometry_grid_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    lons = [r["longitude"] for r in rows]
    lats = [r["latitude"] for r in rows]
    return {
        "n_points": len(rows),
        "min_longitude": min(lons),
        "max_longitude": max(lons),
        "min_latitude": min(lats),
        "max_latitude": max(lats),
        "max_pixel": max(r["pixel"] for r in rows),
        "max_scan": max(r["scan"] for r in rows),
    }
