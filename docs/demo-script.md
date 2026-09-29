# INDU — video demonstration script

Target length: **3 min 57 s**. The narration is 545 words, written to be read
at ~145 words per minute; the timings below assume that pace plus the three
deliberate silent beats noted at the end.

Every number quoted in the narration is one the application or the repository
reports. Nothing here is rounded up for effect — if a run shows different
figures on the day, read the figures on screen instead.

---

## Before recording

1. `make dev-backend` — FastAPI on :8000 (use this target, not bare uvicorn).
2. `make dev-frontend` — Vite on :5173.
3. Confirm **Real (data/raw)** is enabled in the Data section. If it is greyed
   out the Chandrayaan-2 bundle is missing and the real-data beats (0:46,
   2:55) have to be cut.
4. Pre-run **Detect Craters** once and discard it, so the recorded run is warm
   and the terrain cache is populated.
5. Window at 1920×1080, browser chrome hidden, cursor movements slow.

---

## 00:00 – 00:25 · Open

**Visual:** The home screen. "Explore the Moon" type, the four numbered
entries, the max-slope figure ticking in at the bottom right. Hold it still for
three seconds before any cursor moves. Then click **Enter Console**.

> A rover on the Moon has no GPS. No magnetic compass. If it loses track of
> where it is, nothing downstream works — not the map, not the route, not the
> science. INDU is a working prototype of how a rover finds itself again using
> nothing but what its camera can see.

## 00:25 – 00:46 · The problem

**Visual:** Console loads. Slow orbit around the 3D terrain. Let the pipeline
trail — OBSERVE → DETECT → MAP → LOCALIZE → ANALYZE → NAVIGATE — sit in frame.

> The method is visual localization. The rover photographs the terrain around
> it, finds craters in that image, and matches them against craters already
> catalogued from orbit. Where those two sets agree is where it's standing.
> All of it runs end to end over one lunar sector near the south pole.

## 00:46 – 01:10 · The data

**Visual:** Data section. Click **Real (data/raw)**. Then open **Registration →
Status & Results** so the capability dots are readable — source, reference,
NAV, DEM, SPICE.

> Two modes. Synthetic, where ground truth is known, so error can actually be
> measured. And real, on mission data: a Chandrayaan-2 OHRC strip at
> twenty-four centimetres per pixel, an LRO NAC reference frame, and elevation
> from the LOLA south polar DEM. The two-gigabyte raw archive is never
> extracted — the browse product and the navigation grid are read in place, and
> nothing under data/raw is written to.

## 01:10 – 01:41 · Observe and detect

**Visual:** Observation section. Click **Detect Craters**. While it runs, step
through **Rover Image → 4 Quadrants → Crater Points → Crater Map** so each view
lands as the panel fills. End on the crater table.

> The rover looks around in four quadrants. The detector works from shadow — a
> crater near the terminator is a bright rim against a dark floor, and that
> pairing survives changing light better than the outline does. Rim points from
> all four views are fitted into circles. Every column here is a measured
> property of that fit. Confidence is fit quality times angular coverage — not
> a detection probability, and the panel says so.

## 01:41 – 02:13 · Localize

**Visual:** Localization section. **Reference Map**, then **Match** — hold on
the match lines — then **Estimated Position**. Let the ground-truth and
estimated columns and the position-error figure read clearly.

> Now the match. Detected craters are compared against the reference
> catalogue, and the position the most candidates agree on wins. In synthetic
> mode the true position is known, so the panel shows both — and the error
> between them is a real measurement of this pipeline, not a quoted figure.
> Outside synthetic mode there's no truth to compare against, and those rows
> read as dashes rather than inventing a number.

## 02:13 – 02:37 · Terrain and route

**Visual:** Terrain section — click **Elevation**, **Slope**, **Roughness**,
**Traversability** with a beat on each. Then Navigation — **Hazards**, then
**A\* Route**. Pan to the Navigation metrics: path length, max route slope,
high-cost cells, waypoints.

> That fix becomes the start of the route. Elevation, slope and roughness are
> computed in the backend, combined with hazards into a single traversability
> cost, and A-star plans across it. Steep ground isn't forbidden — it's
> expensive. The planner goes around it when it can and through it when it
> must, and the metrics say what that cost.

## 02:37 – 03:07 · Drive it

**Visual:** Rover section — **Follow route**, **Play**, then switch camera to
**Rover POV**. Let the POV run for five seconds. Switch to **Free drive** and
steer with WASD onto a slope until the traction warning appears.

> And then you can drive it. Follow the route, or take the wheel — lunar
> gravity, real grades, a climb limit of twenty-six point six degrees. Push
> past what the wheels hold and it tells you traction is gone. In point-of-view
> on real data, this image is cut from the Chandrayaan-2 product itself. It
> isn't rendered terrain — it's the Moon.

## 03:07 – 03:57 · Close, honestly

**Visual:** Registration panel. Click **Run controlled self-check** and let the
metrics land. Then cut to the home screen for the final line.

> One last thing, because it matters. On a controlled self-check over real
> lunar texture, registration recovers a known transform to within a fifth of a
> pixel — a hundred and seventy-five inliers, RMSE one point two five. Across
> the two instruments, it fails: three inliers, the noise floor. And the cause
> is measured, not guessed — sun elevations of two and seven degrees, shadows
> differing by three and a half times. Six illumination-invariant
> representations were tried; all six hit the same floor. So the answer isn't a
> better matcher. It's holding many views per sector and retrieving the ones
> lit like the query — and that layer is built, waiting on more views to choose
> between. This one reports what it cannot do.

---

## Capture notes

- **Don't narrate the cursor.** Every click above is already covered by a line
  of script; saying "now I'll click" spends words you don't have.
- **Three dead-air beats are deliberate and are already in the timings**: the
  three-second opening hold, the match lines around 02:00, and the five-second
  POV run around 02:55. Don't fill them.
- **The self-check takes a few seconds.** Start it just before the 03:07 line
  so the numbers land under the narration rather than after it.
- If a run produces worse numbers than those quoted, read the screen. The whole
  argument of the closing section is that the system doesn't dress up results.
