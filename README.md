# Lunar Localization & Terrain Navigation Platform

A research-oriented MVP for **localizing lunar imagery, understanding local terrain, identifying hazards, and planning rover routes** over a selected lunar sector.

The prototype combines **Chandrayaan-2 OHRC imagery, LRO NAC reference imagery, terrain analysis, crater/hazard mapping, traversability estimation, A* path planning, and interactive 3D rover visualization** into a single modular pipeline.

---

## 1. Problem

Lunar surface data is acquired by different spacecraft and instruments under different:

- spatial resolutions
- illumination conditions
- viewing geometries
- acquisition times
- camera characteristics
- terrain representations

A high-resolution image from Chandrayaan-2 OHRC therefore cannot simply be compared pixel-for-pixel with a lower-resolution LRO NAC image.

The system addresses this through a local image-registration and terrain-processing workflow.

The current MVP focuses on a bounded **~1 × 1 km² lunar sector** rather than attempting to process or navigate the entire Moon.

---

## 2. Core Workflow

```text
                    Lunar Sector
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   Chandrayaan-2 OHRC              LRO NAC
   Input Image                     Reference Image
          │                             │
          └──────────────┬──────────────┘
                         ▼
                Image Preprocessing
                         │
                         ▼
              Illumination Handling
                         │
                         ▼
              GSD-aware Resampling
                         │
                         ▼
                Feature Matching
                  SIFT / LightGlue
                         │
                         ▼
              RANSAC / MAGSAC++
                         │
                         ▼
              Local Image Registration
                         │
                         ▼
                  Local Map Frame
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
             DEM               Crater/Hazards
              │                     │
        ┌─────┼─────┐               │
        ▼     ▼     ▼               │
      Slope Roughness Elevation      │
        │     │     │               │
        └─────┼─────┴───────────────┘
              ▼
       Traversability Map
              │
              ▼
          A* Planner
              │
              ▼
         Planned Route
              │
              ▼
       3D Terrain Viewer
              │
              ▼
         Rover Simulation
              │
              ▼
           Rover POV
3. MVP Scope

The current system is designed around one selected local lunar sector.

Baseline sector
Anchor Longitude: -70.8810°
Anchor Latitude:   22.7840°

Sector Size: approximately 1 × 1 km²
Region: Lunar South Polar region

The coordinates represent the geographic anchor used for the prototype's local working area.

The MVP does not attempt to:

process the complete Moon
construct a global lunar DEM
process the complete OHRC archive
implement full autonomous rover navigation
implement SLAM
implement EKF/state estimation
implement DWA
implement Hybrid A*
perform real-time onboard navigation
perform energy-optimal rover planning
train a new crater-detection model
build a distributed/cloud processing system

These can be future extensions.

4. Dataset Context
Chandrayaan-2 OHRC

The baseline input imagery is from the Chandrayaan-2 Orbiter High Resolution Camera (OHRC).

Selected product:

ch2_ohr_ncp_20190907T0438126359_d_img_g26.zip

Relevant characteristics:

Instrument:       OHRC
Mission:          Chandrayaan-2
GSD:              ~0.24 m/pixel
Image Size:       12000 × 93693
Acquisition:      2019-09-07
Region:           Lunar South Pole

The original product is very large and is treated as immutable raw scientific data.

The application does not extract or process the complete raw product during normal runtime.

LRO NAC

The baseline reference imagery is from the Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO NAC).

Selected product:

M1322346287LC.IMG

Relevant characteristics:

Instrument:       LRO NAC
Mission:          Lunar Reconnaissance Orbiter
Map Resolution:   ~1.353 m/pixel
Acquisition:      2019-09-05
Region:           Lunar South Pole

The LRO NAC product provides the lower-resolution reference context used by the registration pipeline.

5. Why GSD Matters

The selected OHRC and LRO NAC images have significantly different spatial resolutions.

Approximately:

OHRC     ≈ 0.24 m/pixel
LRO NAC  ≈ 1.353 m/pixel

The system therefore uses:

High-resolution OHRC
        │
        ▼
GSD-aware resampling
        │
        ▼
Multi-scale representation
        │
        ▼
Feature extraction / matching

The goal is not to "convert ISRO imagery into NASA imagery".

Instead, both datasets are brought into a suitable common working scale for registration.

6. Image Registration

Image registration is one of the central components of the project.

The baseline registration pipeline is:

Input Image
    │
    ▼
Preprocessing
    │
    ▼
Illumination Normalization
    │
    ▼
GSD-aware Resampling
    │
    ▼
Feature Extraction
    │
    ├── SIFT
    │
    └── LightGlue (optional)
    │
    ▼
Feature Matching
    │
    ▼
RANSAC / MAGSAC++
    │
    ▼
Geometric Transformation
    │
    ▼
Registered Image
    │
    ▼
Quality Metrics

The registration system is designed to support different registration backends without changing the rest of the application.

7. Illumination Variation

The lunar surface can look substantially different under different solar illumination conditions.

This can produce:

intensity differences
contrast differences
shadow differences
changes in crater appearance
changes in local surface texture

The preprocessing layer therefore supports illumination-aware processing before feature matching.

Possible techniques include:

intensity normalization
CLAHE
local contrast normalization
gradient-based representations
multi-scale preprocessing

The goal is to make geometric features more stable despite changes in illumination.

8. Viewpoint and Geometric Variation

Images acquired by different spacecraft or at different viewing geometries can contain:

translation
rotation
scale differences
local geometric distortion
viewpoint-dependent appearance

The project separates appearance handling from geometric estimation.

Illumination / appearance
          │
          ▼
     Preprocessing
          │
          ▼
Feature extraction + matching
          │
          ▼
Robust geometric estimation
          │
          ▼
      Registration

The initial local registration approach uses robust feature matching and transformation estimation rather than assuming that the images are already geometrically aligned.

9. Registration Quality

A registration result should not be considered valid only because an overlay visually appears reasonable.

The system is designed to calculate measurable registration quality indicators, including:

total feature matches
number of geometric inliers
inlier ratio
reprojection error
RMSE
transformation parameters
spatial coverage
registered footprint

This provides quantitative information about whether a registration is reliable.

10. Terrain Processing

The terrain subsystem converts elevation data into navigation-relevant information.

The current processing pipeline is:

DEM
 │
 ├── Elevation
 │
 ├── Slope
 │
 ├── Roughness
 │
 └── Hillshade / Visualization

The terrain representation is then combined with hazard information.

11. DEM Architecture

The project uses a provider-based DEM architecture.

                  DEMProvider
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
      LocalDEMProvider   SyntheticDEMProvider
              │                 │
              ▼                 ▼
        Real GeoTIFF        Demo Terrain
DEMO Mode

DEMO mode uses a lightweight synthetic terrain representation.

This allows the complete terrain → hazard → route → visualization pipeline to operate offline.

REAL_LOCAL Mode

REAL_LOCAL mode is designed to consume a prepared local DEM for the selected lunar sector.

Large raw scientific products are not used directly by the application.

12. Terrain Layers

The backend produces terrain information including:

Elevation

The underlying height field of the local terrain.

Slope

Represents terrain inclination and can be incorporated into traversability cost.

Roughness

Represents local terrain variability and surface irregularity.

Hazards

Represents areas that should be avoided by the rover.

These layers can be visualized independently in the frontend.

13. Crater and Hazard Mapping

The hazard subsystem provides a representation of unsafe areas.

Potential hazards include:

craters
steep terrain
rough terrain
configurable exclusion regions

Conceptually:

              DEM
               │
       ┌───────┼────────┐
       │       │        │
       ▼       ▼        ▼
     Slope  Roughness Elevation
       │       │        │
       └───────┼────────┘
               ▼
         Terrain Cost
               │
               +
         Hazard Map
               │
               ▼
      Traversability Map
14. Traversability

The route planner does not treat every terrain cell as equally traversable.

Instead, terrain properties and hazards contribute to a cost representation.

Conceptually:

Traversability Cost =
    terrain cost
  + slope cost
  + roughness cost
  + hazard cost

The exact weighting is configurable and can be extended as the navigation subsystem evolves.

15. A* Route Planning

The current path planner uses A* over the local traversability grid.

Start
  │
  ▼
Traversability Grid
  │
  ▼
    A*
  │
  ▼
Collision-aware Route
  │
  ▼
3D Visualization

The planner operates in the local sector rather than attempting global lunar routing.

The resulting path is returned by the backend and visualized in the Three.js frontend.

16. Three.js Terrain Visualization

The frontend uses:

React
TypeScript
Three.js
React Three Fiber

The backend remains the source of truth for scientific terrain calculations.

The frontend is primarily responsible for:

rendering
interaction
visualization
camera control
rover animation

Architecture:

Python Backend
     │
     ├── DEM
     ├── Slope
     ├── Roughness
     ├── Hazards
     └── A* Route
            │
            ▼
       JSON Payload
            │
            ▼
React / Three.js
     │
     ├── Terrain Mesh
     ├── Heatmap Layers
     ├── Route
     └── Rover
17. Terrain Downsampling

The /terrain endpoint converts the terrain representation into a lightweight visualization grid.

The current target is approximately:

100 × 100

The grid preserves the local terrain aspect ratio.

This avoids sending unnecessarily large scientific arrays to the browser while maintaining interactive 3D visualization.

Terrain results are cached locally under:

data/cache/terrain/

The cache is generated data and is not part of the scientific source dataset.

18. Rover Simulation

The frontend contains a lightweight procedural rover model.

The rover includes:

chassis
wheels
mast
camera viewpoint

The rover follows the route generated by the backend.

A* Route
    │
    ▼
Waypoint Interpolation
    │
    ▼
Terrain Height Sampling
    │
    ▼
Rover Pose
    │
    ▼
Three.js Scene

The rover therefore follows the terrain surface rather than moving on a completely flat plane.

19. Rover POV

The application includes a Rover POV mode.

The camera follows the rover's route and looks toward the next route waypoint.

This provides a simple visual simulation of what a rover traversing the planned path would experience.

This is a visualization/simulation feature, not a complete autonomous navigation system.

20. Data Architecture

The repository separates source data, prepared sector data, demo data, metadata, and generated cache.

data/
│
├── raw/
│   ├── chandrayaan2/
│   │   └── ohrc/
│   └── reference/
│       └── lro/
│           └── nac/
│
├── sector/
│   └── sector_001/
│       └── manifest.yaml
│
├── demo/
│   ├── input/
│   ├── reference/
│   ├── dem/
│   └── craters/
│
├── reference_catalog/
│   └── *.yaml
│
└── cache/
    └── terrain/
21. Raw Data

data/raw/ contains large original scientific products.

Examples:

Chandrayaan-2 OHRC
LRO NAC

These datasets can be extremely large.

They are treated as immutable source/archive data.

The application must not extract or process complete raw datasets during normal startup.

The intended workflow is:

Large Raw Product
       │
       ▼
Explicit Preparation Workflow
       │
       ▼
Small Local Crop / Prepared Product
       │
       ▼
data/sector/sector_001/
       │
       ▼
Application Runtime
22. Demo Data

data/demo/ contains lightweight assets that allow the system to run without large external datasets.

DEMO mode is intended for:

development
testing
demonstrations
offline execution
frontend integration
navigation pipeline validation

The demo pipeline does not represent scientific validation of the lunar terrain.

23. Real Local Data

data/sector/sector_001/ is intended to contain prepared local assets for the selected sector.

The application should operate on small prepared assets rather than the complete raw archives.

A sector manifest is used to describe the local sector.

Example:

data/sector/sector_001/manifest.yaml
24. Data Manager

The application uses a DataManager to control which dataset mode is active.

                 DataManager
                     │
             ┌───────┴───────┐
             │               │
             ▼               ▼
           DEMO          REAL_LOCAL
             │               │
             ▼               ▼
        Demo assets      Local assets

The data mode can be inspected through:

GET /data/status

and changed through:

POST /data/mode

If REAL_LOCAL assets are unavailable, the system should report the missing data instead of silently replacing the requested mode with DEMO.

25. Backend API

The backend is implemented using FastAPI.

Current important endpoints include:

GET  /data/status
POST /data/mode
GET  /terrain
POST /plan_path

Additional registration endpoints can be added as the image-registration subsystem is integrated.

26. Repository Structure
lunar-localization-platform/
│
├── backend/
│   ├── app/
│   └── tests/
│
├── configs/
│   ├── app.yaml
│   ├── navigation.yaml
│   ├── registration.yaml
│   ├── scaling.yaml
│   └── sector.yaml
│
├── data/
│   ├── cache/
│   ├── demo/
│   ├── raw/
│   ├── reference_catalog/
│   └── sector/
│
├── docs/
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts
│
├── outputs/
│
├── scripts/
│   ├── build_sector.py
│   ├── build_terrain.py
│   ├── create_demo.py
│   ├── plan_route.py
│   ├── register.py
│   └── run_demo.py
│
├── src/
│   └── lunar_platform/
│       ├── core/
│       ├── io/
│       ├── scaling/
│       ├── preprocessing/
│       ├── registration/
│       ├── planetary/
│       ├── map/
│       ├── terrain/
│       ├── hazards/
│       ├── navigation/
│       ├── simulation/
│       └── utils/
│
├── tests/
│
├── .gitignore
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
27. Technology Stack
Backend
Python
FastAPI
NumPy
OpenCV
Rasterio
PyYAML
Pytest
Frontend
React
TypeScript
Vite
Three.js
React Three Fiber
@react-three/drei
Registration

Baseline:

SIFT
RANSAC / MAGSAC++
OpenCV

Optional:

LightGlue
Planetary Processing

Planned/integration-ready:

ISIS3
Ames Stereo Pipeline (ASP)
Navigation

Current:

A*
28. Installation
Requirements

Recommended:

Python 3.10+
Node.js 18+
npm

A GPU is not required for the basic DEMO pipeline.

Backend

Create a virtual environment:

python3 -m venv .venv

Activate it:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Frontend
cd frontend
npm install
29. Running the Backend

From the project root:

source .venv/bin/activate

PYTHONPATH=backend:src \
uvicorn app.main:app --reload --port 8000

The backend should be available at:

http://127.0.0.1:8000
30. Running the Frontend

Open a second terminal:

cd frontend
npm run dev

Open the URL displayed by Vite.

The frontend communicates with the FastAPI backend running on port 8000.

31. Testing

Run the API and offline integration tests:

source .venv/bin/activate

PYTHONPATH=backend:src \
pytest backend/tests/test_api.py backend/tests/test_offline_pipeline.py

Run the complete test suite:

pytest

The offline pipeline is designed to operate without:

network access
large raw scientific datasets
GPU acceleration
32. Offline Pipeline

The offline integration test validates the basic end-to-end terrain and navigation workflow:

DEMO Data
   │
   ▼
DEM
   │
   ▼
Terrain Processing
   │
   ├── Slope
   ├── Roughness
   └── Hazards
          │
          ▼
   Traversability
          │
          ▼
          A*
          │
          ▼
        Route

This verifies that the core software architecture can run independently of large lunar datasets.

33. Scientific Data Strategy

The project intentionally uses an integration-first approach.

Large scientific products are not copied into the application unnecessarily.

Instead:

NASA / ISRO Source Products
           │
           ▼
Preparation / Extraction
           │
           ▼
Small Local Sector
           │
           ▼
Registration
           │
           ▼
Terrain Processing
           │
           ▼
Navigation

This makes the prototype manageable on a development laptop while preserving a path toward higher-fidelity processing.

34. ISIS3 Integration

ISIS3 is intended as an optional planetary image-processing backend.

Potential responsibilities include:

planetary image ingestion
camera models
spacecraft geometry
SPICE geometry
map projection
control networks
image processing

The project contains an adapter-oriented architecture so ISIS3 can be introduced without coupling the complete application to it.

ISIS3 is not required for the basic DEMO pipeline.

35. Ames Stereo Pipeline Integration

ASP can be used for more advanced planetary stereo processing and DEM generation.

A potential future workflow is:

LRO NAC Stereo Images
          │
          ▼
       ISIS / ASP
          │
          ▼
     Local High-res DEM
          │
          ▼
    Terrain Processing
          │
          ▼
      Route Planning

ASP integration is intended as a higher-fidelity research path.

36. LightGlue Integration

LightGlue is an optional learned feature-matching component.

The project retains a classical SIFT + RANSAC baseline because it is:

CPU-friendly
easier to debug
easier to validate
suitable as a baseline for comparison

Potential registration flow:

SIFT
 │
 └── Baseline
     
LightGlue
 │
 └── Optional advanced matcher

Both can feed into the same robust geometric verification layer.

37. Current Status
Completed
[x] Project architecture
[x] Backend structure
[x] Frontend structure
[x] DataManager
[x] DEMO / REAL_LOCAL data modes
[x] Synthetic DEM provider
[x] Local DEM provider
[x] Terrain processing
[x] Slope calculation
[x] Roughness calculation
[x] Hazard representation
[x] Traversability cost map
[x] A* path planning
[x] Offline integration tests
[x] /terrain endpoint
[x] Terrain downsampling
[x] Terrain caching
[x] React frontend
[x] Three.js terrain visualization
[x] Terrain layer toggles
[x] Procedural rover
[x] Rover route simulation
[x] Rover POV
[x] DEMO mode
[x] REAL_LOCAL architecture
[x] Frontend production build
38. Next Development Phase

The next major scientific component is real lunar image registration.

The intended workflow is:

OHRC Local Crop
       │
       ▼
LRO NAC Local Reference
       │
       ▼
GSD-aware Resampling
       │
       ▼
Illumination-aware Preprocessing
       │
       ▼
SIFT Baseline
       │
       ▼
Feature Matching
       │
       ▼
RANSAC / MAGSAC++
       │
       ▼
Registered Overlay
       │
       ▼
Registration Metrics

After the baseline registration is working, optional research integrations can be evaluated.

39. Validation Philosophy

The project distinguishes between software demonstration and scientific validation.

A DEMO terrain can validate:

data flow
terrain rendering
slope processing
roughness processing
hazard integration
traversability
A* planning
rover visualization

It cannot by itself validate the accuracy of lunar terrain reconstruction.

Real lunar imagery and appropriately prepared terrain products are required for scientific validation.

Similarly, successful visual image alignment alone does not establish scientific registration accuracy; quantitative registration metrics are required.

40. Design Principles
Local First

The MVP focuses on a small known lunar sector rather than global lunar mapping.

Data-Aware

Large scientific datasets remain outside the runtime pipeline.

Modular

Registration, terrain, hazards, and navigation are separated into replaceable components.

Backend as Scientific Source of Truth

Scientific calculations remain in Python.

Frontend for Visualization

Three.js is used primarily for rendering and interaction.

Offline Demonstrability

The DEMO mode should work without external datasets or network access.

Quantitative Validation

Registration and navigation components should expose measurable metrics wherever possible.

Integration Over Reinvention

Existing planetary processing and research algorithms should be integrated through adapters rather than unnecessarily reimplemented.

41. Future Extensions

Possible future development includes:

real local lunar DEM integration
ISIS3 camera and geometry integration
ASP stereo DEM generation
LightGlue registration
improved illumination-invariant matching
orthorectification
subpixel registration refinement
crater detection
boulder detection
uncertainty-aware localization
improved traversability models
rover kinematic constraints
Hybrid A*
local obstacle avoidance
SLAM
visual odometry
onboard localization
energy-aware route planning
multi-sector lunar mapping

These features are outside the current MVP scope.

42. Project Objective

The final objective of the prototype is to demonstrate an integrated local lunar workflow:

        Lunar Image
             │
             ▼
      Image Localization
             │
             ▼
       Local Map Frame
             │
             ▼
      Terrain Understanding
             │
             ▼
       Hazard Mapping
             │
             ▼
      Traversability Model
             │
             ▼
        A* Route Planning
             │
             ▼
      3D Rover Simulation

The result is a lightweight foundation for experimenting with lunar image registration, terrain-aware localization, hazard-aware route planning, and rover traversal visualization on a selected lunar sector.

License

See LICENSE.