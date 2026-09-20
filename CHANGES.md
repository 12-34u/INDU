# CHANGES — multi-view scene library

Status: in progress, 2026-09-20
Scope: the middle layer of the target architecture — the scene library and
retrieval that sit between the geometry we already have and the matcher we
already have.

## Why this work

Registration between the Chandrayaan-2 OHRC strip and the LROC NAC frame does
not succeed. The pipeline runs end to end and reports 26 candidate matches and
3 verified inliers, which for a 4-DOF similarity model is RANSAC's noise floor
rather than a signal. The cause is measured, not assumed:

| | acquired | sun elevation | shadow length |
|---|---|---|---|
| LRO NAC | 2019-09-05 18:10 | **2.2°** | ~26x feature height |
| CH-2 OHRC | 2019-09-07 04:38 | **7.5°** | ~7.6x feature height |

Shadows differ by **3.4x**. Both scenes sit within a few degrees of the
terminator, where the image is dominated by cast shadow whose geometry is a
function of sun angle rather than of the surface. All six illumination
representations were measured on the pair and all landed at the noise floor.

The architectural answer is not a better matcher on this pair. It is to hold
**many** views per sector, each with its illumination recorded, and retrieve the
ones lit like the query. That converts an unsolved matching problem into a
lookup.

## Where we stand

Verified against the code, not estimated.

| Stage | State |
|---|---|
| Global data sources | Partial — one product per sensor |
| Offline sector builder | ~60% — geometry proven, no archive search |
| Local lunar sector | Mostly — DEM complete, imagery is a single basemap |
| **Multi-view scene library** | **Absent** |
| **Retrieval / consensus** | **Absent** |
| Matching | Partial — SIFT, RANSAC and MAGSAC++ present; LightGlue empty |
| Terrain analysis | Done |
| Route planning | Done |
| Visualization | Done |

The foundation and the consumers exist. The middle layer does not, and it is
the layer that would unblock the problem above.

## Planned changes

### C1 — View model and scene library — DONE

A `SceneView` record per image, and a `SceneLibrary` that holds them for a
sector, persists to `data/working/scene_library/<sector>/` and loads back.

Each view carries what retrieval and matching need: sensor, product, how to
read its pixels, ground footprint, centre, ground sample distance, acquisition
time, and illumination.

**Done.** `scene/models.py`, `scene/builder.py`. Both products are catalogued
with footprint, ground scale, acquisition time and illumination, written to
`data/working/scene_library/<sector>/library.json` and read back intact.
Ground scales are the measured ones (OHRC 2.853 m/px, NAC 1.304), not the
label nominal that is wrong by 19%.

### C2 — Illumination per view, from SPICE — DONE

Sun azimuth, elevation, incidence, emission and phase at each view's centre,
computed from the kernels rather than read from labels. The NAC label carries
no illumination angles at all, so this is the only way to have them for both
sides of a pair.

**Done.** `scene/illumination.py`. Sun azimuth, elevation and incidence come
from the ephemeris and the local horizontal frame, so they are available even
for a spacecraft with no SPK loaded. Emission and phase need the camera's
position, so OHRC reports them as absent rather than plausible — there is no
Chandrayaan-2 ephemeris here.

Reproduces the measurement: OHRC 7.52°, NAC 2.15°, a 3.5x shadow ratio.

One defect found and fixed in the process: the NAC camera model furnishes the
SPICE kernels as a side effect, so whichever view was built first silently had
no SPICE and got no illumination. The kernels are now loaded up front, and a
test asserts illumination does not depend on build order.

### C3 — Candidate retrieval — DONE

Given a query footprint and illumination, rank the library: views that overlap
the query ground, ordered by how closely their illumination matches. Return
top-K.

**Done.** `scene/retrieval.py`. Ranks on three scores, weighted so
illumination leads (0.45) over overlap (0.35) and scale (0.20), because
illumination is what made the one real pair unmatchable.

Ground sharing is measured by **polygon intersection**, not bounding box.
The difference is not academic: for the OHRC/NAC pair the box test reports
1.00 while the polygons share 0.74 — and 0.74 independently agrees with the
71% we measured by pushing OHRC grid points through the SPICE camera model,
which is a completely different method.

Illumination is scored through shadow length rather than elevation difference,
because the same five-degree gap means a 3.4x change in shadow at 2° sun and
only 1.2x at 40°.

Every candidate explains itself. Asked to place the OHRC strip, retrieval
returns the NAC frame and says why it is a hard pairing:

    nac:M1322346287LC  score 0.475  overlap 0.74  illum 0.28  scale 0.46
      - Shadows differ by 3.5x (7.5° against 2.2° sun elevation);
        the same ground will not look the same.
      - Ground sampling differs by 2.2x; the finer image must be
        resampled down to match.

### C4 — Multi-view matching and consensus — DONE

Register the query against each of the top-K and combine. A position agreed by
several views is stronger evidence than one pair's residuals.

**Done.** `localization/multi_view.py`, plus `scene/geolocate.py` for turning a
view's own pixels into ground coordinates and `FrameMapping` on the
registration outcome for getting from working pixels back to native ones.

Consensus keeps the largest agreeing group rather than averaging everything, so
one bad registration cannot drag the answer toward itself; the dissenter is
reported with its offset rather than dropped. Positions are averaged as
directions, not as numbers — an arithmetic mean of longitudes is wrong across
the meridian and unstable near a pole, and this sector is at 71° south.

A lone contributing view yields a position but **no corroboration**, and is
reported that way at low confidence. A single registration's residuals say how
well a transform explains its own matches, not whether those matches were
right.

Validated end to end against known truth: a 600 px window of the OHRC browse
raster, handed back as though it were a new image, was placed **4.4 m from the
position the NAV grid gives for its centre** — about 1.5 px at this product's
2.85 m sampling — from 131 inliers at 0.822 px rmse. That exercises retrieval,
registration, working-to-native recovery, geolocation and consensus together.

Run against the real cross-instrument pair it fails, honestly and for the known
reason:

    consensus: succeeded=False  attempted 1  registered 0  agreeing 0
    note: No view produced a position. Only 3 verified inliers ...

### C5 — Uncertainty

Report a position uncertainty rather than a bare coordinate.

**Done when:** the localisation output carries a spread, and a poorly
constrained fix says so.

## What this does not fix

More views is the right answer only if the products share matchable ground.
That is still unresolved: the SPICE projection says 71% of source points land
in the reference and the camera model round-trips exactly, but a template
search over 7 x 7 km at 2.85 m/px peaked at NCC 0.024 and a coarse search over
the whole strip at 30 m/px peaked at 0.135 — both at the search boundary, which
is what no peak looks like.

The library makes that question answerable rather than answering it. Adding NAC
frames at varied sun elevation is what settles it: if a like-lit pair
registers, the pipeline is sound and this pair is simply too far apart
photometrically.
