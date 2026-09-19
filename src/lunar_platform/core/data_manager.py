import os
import json
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

from ..io.raw_inventory import discover_raw

class DataMode(Enum):
    DEMO = "DEMO"
    REAL_LOCAL = "REAL_LOCAL"
    # Simulation driven straight from the immutable bundles under data/raw.
    # Nothing is extracted or prepared first: the browse raster and the NAV
    # geometry grid are read in place, which is cheap enough to do at runtime.
    REAL_RAW = "REAL_RAW"

class DataManager:
    """
    Manages data path resolution and mode selection (DEMO vs REAL_LOCAL).
    Never touches data/raw at runtime.
    """
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.demo_dir = self.base_dir / "demo"
        self.sector_dir = self.base_dir / "sector"
        self.raw_dir = self.base_dir / "raw"
        # Derived products live here so data/raw stays immutable.
        self.working_dir = self.base_dir / "working"

        self._raw_inventory: Optional[Dict[str, Any]] = None

        # Default active mode
        self.active_mode = DataMode.DEMO
        self.active_sector = "sector_001"
        
        # Determine initial fallback if requested sector data doesn't exist
        if not self._check_real_local_available(self.active_sector):
            self.active_mode = DataMode.DEMO
            
    def _check_real_local_available(self, sector_id: str) -> bool:
        """Checks if the required small sector files exist."""
        sector_path = self.sector_dir / sector_id
        if not sector_path.exists():
            return False
            
        # Basic check for at least the imagery folder and manifest
        if not (sector_path / "manifest.yaml").exists():
            return False
        if not (sector_path / "imagery").exists():
            return False
            
        return True

    def set_mode(self, mode: DataMode, sector_id: str = "sector_001") -> None:
        """Sets the active mode. Does NOT silently fallback if files are missing."""
        if mode == DataMode.REAL_LOCAL:
            if not self._check_real_local_available(sector_id):
                raise FileNotFoundError(
                    f"Real local sector data for {sector_id} is missing. "
                    f"Expected data at {self.sector_dir / sector_id}"
                )
        if mode == DataMode.REAL_RAW and self.get_simulation_source() is None:
            raise FileNotFoundError(
                "No Chandrayaan-2 OHRC bundle with a readable browse product was found "
                f"under {self.raw_dir}. Simulation cannot run on real data without it."
            )
                
        self.active_mode = mode
        self.active_sector = sector_id

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.active_mode.value,
            "sector": self.active_sector,
            "real_local_available": self._check_real_local_available(self.active_sector),
            "real_raw_available": self.get_simulation_source() is not None,
        }

    def get_sector_manifest(self) -> Dict[str, Any]:
        """The sector's manifest, or an empty dict when it has none."""
        path = self.sector_dir / self.active_sector / "manifest.yaml"
        if not path.exists():
            return {}
        try:
            import yaml

            with open(path) as handle:
                return yaml.safe_load(handle) or {}
        except Exception:
            return {}

    def get_sector_centre_lonlat(self):
        """
        Where the sector sits on the Moon, from its manifest.

        Returns None when the manifest does not say, so callers can fall back
        to centring on the imagery rather than inventing a coordinate.
        """
        centre = self.get_sector_manifest().get("center") or {}
        lon, lat = centre.get("longitude"), centre.get("latitude")
        if lon is None or lat is None:
            return None
        return float(lon), float(lat)

    def get_simulation_source(self) -> Optional[Dict[str, Any]]:
        """
        Everything needed to build a sector basemap from real raw imagery.

        Returns the archive, the members to read in place and the product's own
        label, or None when data/raw holds no usable source bundle. The full
        image member is never referenced here: it is 1.1 GB of DEFLATE and the
        browse product carries the same scene at 10x decimation.
        """
        product = self._first_source_product()
        if product is None:
            return None

        if product.is_archive:
            browse = product.member_by_kind("browse")
            if browse is None:
                return None
            geometry = product.member_by_kind("geometry")
            return {
                "archive": product.path,
                "browse_member": browse.name,
                "geometry_member": geometry.name if geometry else None,
                "browse_path": None,
                "geometry_path": None,
                "label": dict(product.label),
                "product_id": product.product_id,
                "instrument": product.instrument,
                "mission": product.mission,
            }

        # No archive, but an extracted bundle may still be on disk. Its
        # geometry grid sits in a sibling directory of the browse image.
        loose = self._loose_browse_source()
        if loose is None:
            return None

        geometry_path = None
        bundle_root = loose.path.parent
        for _ in range(5):
            candidates = sorted(bundle_root.glob("geometry/**/*.csv"))
            if candidates:
                geometry_path = candidates[0]
                break
            if bundle_root.parent == bundle_root:
                break
            bundle_root = bundle_root.parent

        label = dict(loose.label)
        if not label.get("lines"):
            # The browse label describes the browse raster, so the full-image
            # label next to it is what states the source dimensions.
            for sibling in self.get_raw_inventory().get("source_products", []):
                if sibling is not loose and sibling.label.get("lines"):
                    label = dict(sibling.label)
                    break

        return {
            "archive": None,
            "browse_member": None,
            "geometry_member": None,
            "browse_path": loose.path,
            "geometry_path": geometry_path,
            "label": label,
            "product_id": loose.product_id,
            "instrument": loose.instrument,
            "mission": loose.mission,
        }

    def get_input_image_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "input" / "demo_input.png"
        return self.sector_dir / self.active_sector / "imagery" / "ohrc_crop.tif"
        
    def get_input_metadata_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "input" / "demo_input_metadata.json"
        # In REAL_LOCAL, metadata might be alongside or inside manifest
        return self.sector_dir / self.active_sector / "imagery" / "ohrc_crop_metadata.json"

    def get_reference_image_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "reference" / "demo_ref.tif"
        return self.sector_dir / self.active_sector / "imagery" / "lro_nac_reference_crop.tif"

    def get_dem_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "dem" / "demo_dem.tif"
        return self.sector_dir / self.active_sector / "dem" / "dem.tif"

    def get_craters_path(self) -> Path:
        if self.active_mode == DataMode.DEMO:
            return self.demo_dir / "craters" / "demo_craters.json"
        return self.sector_dir / self.active_sector / "craters" / "craters.json"

    # ------------------------------------------------------------------
    # Registration data access
    #
    # These resolve to GDAL-readable references rather than plain paths,
    # because the usable source product lives inside an archive that must not
    # be extracted. DEM and SPICE are reported as capabilities and never
    # required: their absence disables geolocation features, not registration.
    # ------------------------------------------------------------------

    def get_raw_inventory(self, refresh: bool = False) -> Dict[str, Any]:
        """Discover data/raw. Cached because it is read on every status call."""
        if self._raw_inventory is None or refresh:
            self._raw_inventory = discover_raw(self.raw_dir)
        return self._raw_inventory

    def _first_source_product(self):
        """
        The best source product discovered, not merely the first.

        data/raw may hold an archive alongside an extracted copy of itself, in
        which case discovery reports the bundle, its loose browse image and its
        1.1 GB loose full image as three separate products. Taking whichever
        sorted first would pick a stray member and report no usable source at
        all, so the bundle wins, then a loose browse image, and the full image
        only as a last resort.
        """
        products = self.get_raw_inventory().get("source_products", [])
        if not products:
            return None

        for product in products:
            if product.is_archive and product.member_by_kind("browse") is not None:
                return product
        for product in products:
            if product.is_archive:
                return product
        for product in products:
            if "browse" in str(product.path).lower() or "_brw_" in str(product.path).lower():
                return product
        return products[0]

    def _loose_browse_source(self):
        """A browse image sitting loose on disk, when no archive carries one."""
        for product in self.get_raw_inventory().get("source_products", []):
            path = str(product.path).lower()
            if not product.is_archive and ("browse" in path or "_brw_" in path):
                return product
        return None

    def _first_reference_product(self):
        products = self.get_raw_inventory().get("reference_products", [])
        return products[0] if products else None

    def get_working_registration_dir(self) -> Path:
        return self.working_dir / "registration" / self.active_sector

    def get_source_image_ref(self) -> Optional[str]:
        """
        GDAL reference for the moving/source image.

        Prefers a prepared working product. Otherwise falls back to the browse
        image inside the archive, addressed through /vsizip/ so the archive is
        read in place. The full-resolution image is never used here: it is a
        1.1 GB compressed member.
        """
        prepared = self.get_working_registration_dir() / "source.tif"
        if prepared.exists():
            return str(prepared)

        product = self._first_source_product()
        if product is None:
            return None

        browse = product.member_by_kind("browse")
        if browse is None:
            return None
        return f"/vsizip/{product.path}/{browse.name}"

    def get_reference_image_ref(self) -> Optional[str]:
        """GDAL reference for the fixed/reference image."""
        prepared = self.get_working_registration_dir() / "reference.tif"
        if prepared.exists():
            return str(prepared)

        product = self._first_reference_product()
        return str(product.path) if product is not None else None

    def get_navigation_ref(self) -> Optional[str]:
        """
        Reference to the source geometry grid.

        This is sampled lat/lon per (pixel, scan), not a camera model. It
        supports approximate footprint work only.
        """
        product = self._first_source_product()
        if product is None:
            return None
        geometry = product.member_by_kind("geometry")
        if geometry is None:
            return None
        return f"{product.path}!{geometry.name}"

    def get_spice_path(self) -> Optional[str]:
        spice = self.get_raw_inventory().get("spice_files", [])
        return spice[0] if spice else None

    def get_raw_dem_label(self) -> Optional[Path]:
        """
        A DEM under data/raw that can be read through its PDS label.

        The label is what is returned, not the raster: GDAL needs it to know
        the projection, and a bare .img carries none.
        """
        dem_root = self.raw_dir / "dem"
        if not dem_root.exists():
            return None

        for label in sorted(dem_root.rglob("*.lbl")) + sorted(dem_root.rglob("*.LBL")):
            raster = label.with_suffix(".img")
            if not raster.exists():
                raster = label.with_suffix(".IMG")
            if raster.exists():
                return label
        return None

    def has_raw_dem(self) -> bool:
        return self.get_raw_dem_label() is not None

    def get_spice_root(self) -> Path:
        """Where SPICE kernels live. Read-only, like everything under data/raw."""
        return self.raw_dir / "spice"

    def get_spice_kernels(self):
        """
        The discovered SPICE kernel set, or None when there are none.

        Returns the set whether or not it is complete; completeness is a
        property the caller can report rather than a reason to hide it.
        """
        from ..planetary.spice_adapter import SpiceKernelSet

        root = self.get_spice_root()
        if not root.exists():
            return None
        kernels = SpiceKernelSet.discover(root)
        return kernels if kernels.paths else None

    def get_nac_reference(self) -> Optional[Dict[str, Any]]:
        """
        The LROC NAC product and its label, for SPICE geolocation.

        The image itself is not read here; only the label is needed to build
        the camera model, and the image is 252 MB.
        """
        product = self._first_reference_product()
        if product is None:
            return None

        label_path = None
        for suffix in (".xml", ".lbl", ".XML", ".LBL"):
            candidate = product.path.with_suffix(suffix)
            if candidate.exists():
                label_path = candidate
                break
        if label_path is None:
            return None

        return {
            "product_id": product.product_id,
            "image_path": product.path,
            "label_path": label_path,
            "instrument": product.instrument,
            "mission": product.mission,
        }

    def has_dem(self) -> bool:
        return self.get_dem_path().exists()

    def has_spice(self) -> bool:
        return self.get_spice_path() is not None

    def get_capabilities(self) -> Dict[str, Any]:
        """
        What the system can currently do, as explicit flags.

        Registration needs a source and a reference image and nothing else, so
        it stays available with DEM and SPICE absent.
        """
        source = self.get_source_image_ref()
        reference = self.get_reference_image_ref()
        navigation = self.get_navigation_ref()
        dem = self.has_dem()
        spice = self.has_spice()

        source_product = self._first_source_product()
        reference_product = self._first_reference_product()

        return {
            "mode": self.active_mode.value,
            "sector": self.active_sector,
            "source_image": source is not None,
            "reference_image": reference is not None,
            "navigation": navigation is not None,
            "dem": dem,
            "spice": spice,
            "registration": source is not None and reference is not None,
            "source_ref": source,
            "reference_ref": reference,
            "navigation_ref": navigation,
            "source_product": source_product.to_dict() if source_product else None,
            "reference_product": reference_product.to_dict() if reference_product else None,
            "messages": self._capability_messages(dem, spice, source, reference),
        }

    def _capability_messages(self, dem: bool, spice: bool, source, reference) -> list:
        messages = []
        if not dem:
            messages.append("DEM not available - terrain geolocation features disabled.")
        if not spice:
            messages.append(
                "SPICE not available - precise camera/geometry refinement disabled."
            )
        if source is None:
            messages.append("No source image discovered under data/raw.")
        if reference is None:
            messages.append("No reference image discovered under data/raw.")
        return messages
