# Modules

## Data discovery and access

| Module | Responsibility |
|---|---|
| `io/raw_inventory.py` | Discover `data/raw`, classify archive members, parse PDS4 labels, read the NAV geometry grid. Read-only; never extracts. |
| `io/raster_loader.py` | Open a source product read-only, addressing archives via `/vsizip/`. |
| `core/data_manager.py` | Mode selection plus capability flags and registration data references. Single place that resolves paths. |

## Preprocessing

| Module | Responsibility |
|---|---|
| `preprocessing/normalization.py` | Percentile stretch to 8-bit; valid-data fraction. Brings differing dtypes to a common representation. |
| `preprocessing/illumination.py` | `raw`, `clahe`, `local_contrast`, `gradient` representations, selectable by name. Mitigates illumination difference; does not remove it. |

## Registration

| Module | Responsibility |
|---|---|
| `registration/base.py` | `Matcher` protocol. |
| `registration/sift.py` | SIFT + ratio test (existing). |
| `registration/ransac.py` | Similarity / affine / homography with RANSAC or MAGSAC++ (existing). |
| `registration/distribution.py` | Grid-based uniform match selection; occupancy and Gini coverage metrics. |
| `registration/refine.py` | Sub-pixel refinement of inliers by local NCC with a quadratic peak fit. |
| `registration/geometry.py` | Reprojection RMSE, spatial coverage (existing). |
| `registration/pipeline.py` | Orchestrates the stages, applies the quality gate, produces metrics. |
| `registration/products.py` | Writes `registered_image.tif`, `match_points.json`, `matches.csv`, `transform.json`, `metrics.json`, display derivatives. |
| `registration/validation.py` | Known-transform controlled validation on real imagery. |

## Scale, viewpoint, illumination

The three variations named in the problem statement map to specific stages:

- **Illumination** — `preprocessing/illumination.py`, selected per run and recorded in the metrics.
- **Viewpoint** — geometric model in `registration/ransac.py` (`similarity` by default, `affine`/`homography` configurable), verified by RANSAC.
- **Scale** — decimated reads in `pipeline.load_image` bring both products to a comparable working scale; `scaling/pyramid.py` provides multi-scale levels.

None of these are eliminated. They are mitigated and then measured.

## Unchanged subsystems

`terrain/`, `hazards/`, `navigation/` and `simulation/camera.py` behave as
before, as do `DEMProvider`, the DEMO and REAL_LOCAL modes, the A* planner and
the 3D rover visualization.

## Simulation on real data (REAL_RAW)

A later pass added a second path through the perception pipeline that runs on
the bundles under `data/raw` instead of on rendered frames. The synthetic path
is unchanged and still the default; the real path is selected by the REAL_RAW
data mode or per request.

- `io/ohrc_browse.py` — reads the browse raster and the NAV geometry grid out
  of the bundle in place, and interpolates longitude/latitude on the grid's
  lattice.
- `observation/basemap.py` — cuts a sector tile from that raster and anchors it
  on the Moon. The metre scale is measured from the NAV grid rather than taken
  from the label's nominal `pixel_resolution`, and the sun bearing is derived
  from the label's sun azimuth rotated into image coordinates.
- `observation/real_view.py` — cuts the rover's view from the basemap in a
  single resampling. Frames that fall off the edge of the imagery are reported,
  never padded.
- `crater_detection/shadow_pairs.py` — pairs shadows with highlights along the
  sun bearing. `rim_points.py` looks for a bright annulus, which is what a
  crater looks like in the synthetic view and not what it looks like here.
- `localization/image_match.py` — recovers the position by registering the
  observation against the basemap.

### Geometry from SPICE, and real elevation

- `planetary/spice_adapter.py` — builds the LROC NAC pushbroom camera model
  from the mission kernels, forward (pixel to longitude/latitude) and inverse
  (longitude/latitude to pixel). The NAC label carries no geographic metadata
  at all, so this is the only way that image can be placed on the Moon.
- `terrain/lola_dem.py` — samples the LOLA south polar DEM into the sector
  frame through the sector's own coordinates, so elevation and imagery share
  one grid.

Two mistakes in this area produce results that look right and are not, so both
are asserted in the tests rather than left to inspection:

- **Swapping the pushbroom axes.** The instrument frame's x is along-track and
  y is cross-track, which is the opposite of what the instrument kernel's
  `TRANSX`/`TRANSY` names suggest; the kernel's own `ITRANSS`/`ITRANSL` settle
  it. Getting it backwards lays the swath along the orbit instead of across it
  and still yields a footprint of about the right size in about the right
  place. `test_swath_width_matches_the_detector` and
  `test_along_track_is_perpendicular_to_the_detector_row` catch it.
- **A DEM of the wrong region.** SLDEM2015 spans 60°S–60°N and cannot cover a
  sector at 71°S, but such a tile reads cleanly and yields plausible terrain.
  `lola_dem` refuses a sector outside the DEM instead of returning a filled
  array.

The inverse camera model goes through `spkcpt` rather than differencing
positions by hand, so that it carries the same light-time and stellar
aberration convention `sincpt` applies on the way out. Doing the subtraction
manually omits stellar aberration, which is about 1e-4 radians here: a constant
seven-line bias, small enough to look like rounding and far too large to be it.
Round-trip agreement is better than 0.02 px.

### Why localisation changed on real imagery

The crater-constellation matcher in `localization/matching.py` recovers a
position from a handful of landmarks, and it works on the synthetic
observations it was built for. Measured on real OHRC imagery it fixed 1 pose in
30, and that one fix was 156 m wrong. Detections repeat between two views of
the same ground only about a third of the time, which is too few consistent
correspondences to place a rover.

Registering the observation against the basemap uses all the texture in the
frame instead of a dozen fitted circles. Measured over the same poses it fixed
40 of 40 at a median error of about 1.5 m, roughly half a basemap pixel, and
refused all 20 observations taken from an unrelated part of the strip. Both
paths remain in the codebase; the matcher is still what the synthetic mode
uses.
