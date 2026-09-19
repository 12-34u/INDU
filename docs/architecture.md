# Architecture

## Current MVP

```
data/raw  (immutable)
    │
    ▼
raw discovery + PDS4 labels          io/raw_inventory.py
    │
    ▼
DataManager capabilities             core/data_manager.py
    │   source · reference · NAV · DEM? · SPICE?
    ▼
registration pipeline                registration/pipeline.py
    │
    ├── load (decimated / windowed)
    ├── radiometric normalisation    preprocessing/normalization.py
    ├── illumination representation  preprocessing/illumination.py
    ├── SIFT matching                registration/sift.py
    ├── uniform spatial selection    registration/distribution.py
    ├── RANSAC verification          registration/ransac.py
    ├── sub-pixel refinement         registration/refine.py
    └── quality gate                 configs/registration.yaml
    │
    ▼
match points · metrics · registered image    registration/products.py
    │
    ▼
outputs/registration/<scene_id>/
    │
    ▼
frontend                             components/RegistrationPanel.tsx
```

## Not yet integrated

| Capability | Status | Consequence |
|---|---|---|
| DEM | absent | Terrain geolocation features disabled. Terrain/hazard/A* run on DEMO data only. |
| SPICE kernels | absent | No camera model. Registration residuals are image-space, not ground accuracy. |
| Camera geometry | absent | Cannot place the NAC strip on the Moon, so OHRC↔NAC overlap is undeterminable. |

Both DEM and SPICE are reported as capability flags and neither gates
registration. `DEMProvider` and the DEMO/REAL_LOCAL modes are unchanged.

## Future

```
registration + DEM + SPICE + camera geometry
        ↓
precise lunar localization
        ↓
terrain / hazard analysis
        ↓
rover route planning
```

## Why the source image is the browse product

The OHRC full image is 93693 × 12000 inside a DEFLATE archive — 1.12 GB
uncompressed. Every member is compressed, so a windowed read would still have
to decompress sequentially to reach the window. The bundle ships a browse PNG
at 1200 × 9369 (a 10× downsample, ≈2.4 m/px) which GDAL reads in place through
`/vsizip/`. Registration therefore runs at browse resolution and the full image
is catalogued but never read. `DataManager.get_source_image_ref()` prefers a
prepared working product when one exists, so raising the resolution later means
preparing that product, not changing the pipeline.

## Two registration pairs

**`source_reference`** — the real cross-instrument attempt, OHRC against NAC.
Currently yields 0 candidate matches at working resolution. This is reported
as-is. With no camera model there is no way to know whether the two strips
overlap, so a low match count is the expected honest outcome rather than a
tuning problem.

**`self_check`** — controlled validation. A real OHRC tile is transformed by a
known similarity transform (7° rotation, 0.82 scale, 14/−9 px translation) and
given a different illumination response (gamma 1.7), then registered back. The
applied transform is known exactly, so the recovered transform can be scored.
This measures the pipeline on real lunar texture. It does **not** validate
cross-instrument registration.

## Quality gate

`succeeded` reflects the configured thresholds in `configs/registration.yaml`,
not merely whether a transform was produced. A degeneracy floor is applied on
top: a model fitted with close to its minimum sample count passes through its
own points almost exactly and reports a near-zero RMSE that says nothing about
alignment, so such fits are rejected with a stated reason.

Thresholds are configurable and are **not** scientifically validated values.
