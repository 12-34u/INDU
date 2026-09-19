# PRD — Chandrayaan-2 to Lunar Reference Image Registration

Status: in progress, 2026-09-19 (P0 implemented; see Progress)
Owner: INDU
Scope: the registration subsystem only. Terrain, routing and the rover
simulation are out of scope for this document.

## 1. Problem

Align a Chandrayaan-2 optical image (**source**, moving) to a lunar reference
image (**reference**, fixed) by finding correspondences between them, and
report the registered product, the match points and the accuracy achieved.

Three properties of the data make this hard, and each has to be handled
explicitly rather than hoped away:

| Challenge | What it does to the data |
|---|---|
| **Illumination variation** | Sun azimuth and elevation differ between acquisitions. The same crater reads as a bright annulus in one image and a dark bowl with a lit crescent in the other. Intensity correlation between them is weak. |
| **Viewpoint variation** | Different camera positions and orientations shift, scale, rotate and perspective-distort the surface. Over real relief this shows up as differential displacement, not a single global warp. |
| **Scale variation** | Missions fly at different altitudes with different instruments. OHRC is 0.24 m/px native; LROC NAC is 0.5 m/px native and about 1.3 m/px as 2x-summed. Ratios of 2-6x are normal. |

## 2. Where we actually are

Stated plainly, because it sets the priorities.

**Works.** `ohrc_self_check` — a real OHRC tile registered against a warped
copy of itself under a known transform — passes: 3682 candidates, 182 after
uniform selection, 144 verified inliers, inlier ratio 0.79, spatial coverage
0.48, 173 points sub-pixel refined. This validates the pipeline's plumbing.

**Does not work.** `ch2_ohrc_vs_lronac` — the actual deliverable — reports
`succeeded: false` with **0 candidate matches**. It fails at feature matching,
before geometry is ever attempted.

The cause is visible in that run's own load stage:

```
source    732 x 5722   (decimated 0.611)  ->  ~4.7 m/px
reference 450 x 9301   (decimated 0.178)  ->  ~7.3 m/px
```

Two separate faults, both upstream of the matcher:

1. **The pair is never brought to a common ground scale.** `load_image`
   decimates each image to a per-image pixel budget, computed independently.
   The two land at different metres-per-pixel, so no scale-sensitive matcher
   can succeed.
2. **The search is not restricted to the overlap.** A 22 km source strip is
   matched against a 75 km reference strip. The true overlap is a few percent
   of the reference frame. SPICE now tells us the sector sits at NAC lines
   ~31900-33700, and nothing uses that.

A third, smaller gap: even the passing self-check reports `rmse_px` 1.33 and
`median_error_px` 1.03. That is not sub-pixel, and the configured gate
(`max_rmse_px: 5.0`) does not ask for sub-pixel.

## 2a. Progress

### Done

**P0-1 Common ground scale** - `registration/ground_scale.py`. Each product is
asked how big its pixels really are: the raster's own georeferencing first,
then mission geometry (OHRC NAV grid, LROC NAC via SPICE), then the label, then
"unknown" rather than a guess. Measured: source 2.853 m/px, reference 1.304
m/px, a 2.19x ratio that was previously never reconciled. Both are now
resampled to one working scale.

**P0-2 Overlap-restricted read** - `registration/overlap.py`. The source
footprint is pushed through both geometries into reference pixels. 86 of 121
source grid points land in the reference (71%), restricting the reference read
from 52224 to 20592 lines, a 2.5x reduction.

**Radiometry defect (not in the original plan, and the largest single fault).**
LROC NAC declares `nodata = -32768` and about 1.3% of a typical window carries
it. That value *is* the 1st percentile, so `to_uint8`'s percentile stretch ran
from -32768 to ~560 and mapped all real surface values to 250-255. The
reference reached the matcher as a near-white rectangle, having passed every
shape and validity check on the way. `load_image` now returns declared nodata
as NaN and `valid_data_fraction` counts it as missing.

Effect of the three together: **0 candidate matches -> 26**, and the pipeline
now runs end to end instead of aborting at the matching stage.

**P1-1 Sub-pixel accuracy - met, on the controlled case.** Two different
numbers had been conflated. `rmse_px` is the reprojection residual of the
matched points and reads 1.36 px. The accuracy of the *recovered transform*
against the known truth is what the brief asks for, and it is **0.400 px RMSE**
(mean 0.341, median 0.317, max 0.685 over a 25-point grid). That is sub-pixel.
The two should be reported separately rather than one standing in for the
other.

**P2-1 Illumination representations** - `preprocessing/illumination.py` gains
monogenic `phase_congruency` (log-Gabor + Riesz, contrast invariant by
construction) and `shadow_mask`, and all six representations were measured on
the real pair rather than chosen by assumption.

### Not done: P0-3, and why

The real pair still does not register. Every representation returns 2-3
inliers, which for a 4-DOF similarity model is RANSAC's noise floor, not a
signal:

| representation | matches | inliers |
|---|---|---|
| raw | 33 | 2 |
| clahe | 29 | 3 |
| local_contrast | 46 | 3 |
| gradient | 29 | 2 |
| phase_congruency | 8 | 3 |
| phase_congruency_masked | 19 | 3 |

The illumination difference is now measured rather than assumed. At the shared
ground point, computed from SPICE:

| | acquired | sun elevation | shadow length |
|---|---|---|---|
| LROC NAC | 2019-09-05 18:10 | **2.2 deg** | ~26x feature height |
| CH-2 OHRC | 2019-09-07 04:38 | **7.5 deg** | ~7.6x feature height |

Shadows differ by **3.4x**. Both scenes are within a few degrees of the
terminator, where the image is dominated by cast shadow whose geometry is a
function of sun angle, not of the surface. Intensity-based representations key
on shadow edges, and those edges move.

Two further searches found no correspondence either: a template search over a
7 x 7 km neighbourhood at 2.85 m/px peaked at NCC 0.024, and a coarse search
over the whole strip at 30 m/px peaked at 0.135 - both at the search boundary,
which is what no peak looks like.

**This leaves an open question that has to be settled before more matcher work
is justified:** whether the two products actually share ground. The SPICE
projection says 71% of source points land in the reference, and the NAC camera
model round-trips exactly, but a self-consistent camera model does not prove
agreement between the OHRC NAV grid and MOON_ME. An attempt to arbitrate using
LOLA hillshade was inconclusive (correlations -0.11 and +0.12): a 60 m DEM
cannot predict the metre-scale shadows that dominate a 2-degree-sun scene.

### Revised priorities

1. **Settle the overlap question.** Obtain a second reference product with
   similar illumination to the OHRC scene, or a NAC frame independently known
   to cover this ground. If a like-illuminated pair registers, the pipeline is
   sound and this pair is simply too far apart photometrically.
2. **P2-2 learned matcher**, promoted. Detector-free dense matchers (LoFTR,
   LightGlue) are the current answer to exactly this appearance gap, and are
   the only remaining lever if the products do overlap.
3. Report the controlled sub-pixel figure and the cross-instrument failure
   side by side. A pipeline that refuses to register an unregistrable pair is
   behaving correctly; the gates did their job here.

## 3. Goals

- **G1.** Register the real Chandrayaan-2 OHRC product against the real LROC
  NAC product, successfully and repeatably.
- **G2.** Achieve sub-pixel accuracy, measured, not asserted.
- **G3.** Keep match points uniformly distributed across the overlapping area.
- **G4.** Emit the registered product, the match points and the metrics.
- **G5.** Be generic: any source/reference pair, from a command line.

### Non-goals

- Absolute cartographic accuracy against a geodetic control network.
- Stereo DEM generation.
- Anything in the rover simulation.

## 4. Requirements

### Functional

| ID | Requirement |
|---|---|
| F1 | Resample source and reference to a common ground sample distance before matching, derived from each product's own geometry. |
| F2 | Restrict the reference read to the region the source actually overlaps, when geometry is available to compute it. |
| F3 | Match features, verify geometrically with RANSAC, refine to sub-pixel. |
| F4 | Cap matches per grid cell so the surviving set spans the frame. |
| F5 | Write: registered image, match-point table, metrics, display derivatives. |
| F6 | Expose a CLI accepting an arbitrary source and reference. |
| F7 | Report which geometry sources were used (NAV grid, SPICE, DEM) and which were not. |

### Quality gates

A run is `quality_passed` only if **all** hold:

| Metric | Gate | Why |
|---|---|---|
| Verified inliers | >= 30 | Below this a similarity fit is not meaningfully constrained. |
| Inlier ratio | >= 0.35 | Lower suggests the consensus is coincidental. |
| Reprojection RMSE | **< 1.0 px** | The brief's sub-pixel requirement, made enforceable. |
| Spatial coverage | >= 0.30 | Matches confined to one corner extrapolate badly. |
| Distribution Gini | <= 0.65 | Coverage can be met while still piling into a few cells. |

Metrics reported regardless of pass/fail: candidate matches, uniform-selected
count, inliers, inlier ratio, RMSE, median and max error, spatial coverage,
occupancy grid, Gini, count refined to sub-pixel, working GSD, and the
illumination representation used.

## 5. Approach, per challenge

**Scale.** Resolve each product's true metres-per-pixel from its own geometry -
the OHRC NAV grid for the source, the SPICE camera model for NAC - rather than
from a nominal label value, which for OHRC is wrong by 19%. Resample both to a
single common GSD, defaulting to the coarser of the two so neither is
magnified past the detail it carries.

**Viewpoint.** Keep the configurable global model (similarity / affine /
homography) as the baseline. Relief-induced differential displacement is
acknowledged and deferred; correcting it needs orthorectification against the
DEM, which is a later item.

**Illumination.** Keep the existing representations (CLAHE, local contrast
normalisation, gradient magnitude) but stop guessing: evaluate the candidates
on the pair and report which performed best, rather than taking a config
string on faith.

## 6. Work items

Ordered. Each is independently verifiable.

### P0-1 — Common ground scale
Resolve per-product GSD from NAV grid / SPICE / label, in that order of
preference, and resample both images to a common working GSD.
**Done when:** a run reports the working GSD and both images' native GSD, and
the two working images are at the same metres-per-pixel.

### P0-2 — Overlap-restricted reference read
Project the source footprint into reference pixel coordinates via SPICE and
read only that window plus a margin.
**Done when:** the reference read covers the overlap, and a run records the
window used and how it was derived.

### P0-3 — Make the real pair register
**Done when:** `ch2_ohrc_vs_lronac` reports `succeeded: true` and meets the
inlier and coverage gates.

### P1-1 — Enforce sub-pixel
Tighten the RMSE gate to 1.0 px and add the Gini gate. Verify against the
known-transform self-check, where truth is exact.
**Done when:** both the self-check and the real pair report RMSE < 1.0 px, or
the shortfall is reported honestly as a failure.

### P1-2 — Generic CLI
`scripts/register.py --source <ref> --reference <ref> [--out DIR]`.
**Done when:** it registers an arbitrary pair with no code changes.

### P2-1 — Illumination representation selection
Run the candidate representations and report comparative inlier counts.
**Done when:** the chosen representation is a measured outcome, recorded.

### P2-2 — Learned matcher adapter
Fill `lightglue_adapter.py` behind the existing matcher interface, with SIFT
retained as the default and the fallback.
**Done when:** a matcher can be selected by config and both are measured on
the same pair.

### P3-1 — Orthorectification against the DEM
Remove relief displacement before matching.
**Done when:** residuals are reported before and after, on the same pair.

## 7. Risks

- **The products may not overlap usefully.** SPICE says the OHRC sector falls
  within the NAC footprint (about 90% of a 2400 m sector), but that is an
  ellipsoid-intercept result, not a guarantee of matchable texture.
- **Sub-pixel may not be reachable** across a 2-6x scale ratio and a two-day
  illumination difference with SIFT alone. P2-2 exists for this reason. If it
  is not reachable, the honest outcome is to report the achieved figure and
  say so, not to loosen the gate until it passes.
- **SPICE covers the reference only.** The NAC camera model is available; the
  OHRC side relies on its NAV grid, which is sampled geometry rather than a
  camera model.
