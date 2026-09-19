import { useState, useEffect, useMemo, useRef, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'
import { Terrain3D } from './components/Terrain3D'
import { RoverView } from './components/RoverView'
import { InspectorPanel } from './components/InspectorPanel'
import { HomeScreen } from './components/HomeScreen'
import type {
  DataMode,
  TerrainLayer,
  CameraMode,
  TerrainData,
  RoutePoint,
  RouteMetrics,
  PerceptionResult,
  ReferenceMap,
  DataCapabilities,
  RegistrationResult,
  RoverPose,
  RoverTelemetry,
  DriveMode,
  StartSite,
  InspectorView,
} from './types'
import './App.css'

const API_BASE = 'http://localhost:8000'

/**
 * DEM geometry the planner searches. These fields come from /terrain; they are
 * declared here rather than in types.ts to keep this integration to two files.
 */
type PlannerTerrain = TerrainData & {
  gsd_m: number
  dem_width: number
  dem_height: number
}

/** Planner-grid cell (col, row), matching A*'s cost_map[start[1], start[0]]. */
type PlannerCell = [number, number]

/** Where the cell handed to A* came from. Shown in the UI so the two are never confused. */
type RouteStartSource = 'demo-fallback' | 'localization' | 'start-site'

/**
 * Demo start used only before any localization fix exists. It is a fallback,
 * not a localization result, and is labelled as such wherever it is displayed.
 */
const DEMO_START_CELL: PlannerCell = [100, 100]
const DEMO_GOAL_CELL: PlannerCell = [800, 800]

/**
 * Places to start from, as fractions of the sector rather than fixed cells.
 *
 * The sector is 1000 m wide in the demo and about 2400 m on real imagery, so
 * hard-coded cells would land in different places - or off the edge - between
 * modes. Fractions keep one list meaningful for both.
 */
const START_SITES: StartSite[] = [
  { id: 'northwest', label: 'NW Landing', fx: 0.12, fy: 0.12, heading_deg: 45 },
  { id: 'north-ridge', label: 'N Ridge', fx: 0.55, fy: 0.14, heading_deg: 110 },
  { id: 'centre', label: 'Central Flats', fx: 0.5, fy: 0.5, heading_deg: 0 },
  { id: 'east-basin', label: 'E Basin', fx: 0.86, fy: 0.42, heading_deg: 200 },
  { id: 'southwest', label: 'SW Approach', fx: 0.16, fy: 0.84, heading_deg: -35 },
]

const siteToMetres = (site: StartSite, terrain: PlannerTerrain) => ({
  x_m: site.fx * terrain.width_m,
  y_m: site.fy * terrain.height_m,
  heading_deg: site.heading_deg,
})

const clampCell = (value: number, size: number) => Math.min(size - 1, Math.max(0, value))

/**
 * Sector metres -> full-DEM planner cell.
 *
 * The sector frame is metres from the DEM's top-left corner with y increasing
 * downward - the same convention the crater catalogue, the camera model and
 * Terrain3D's sectorToLocal already use - so this is a pure scale by gsd_m with
 * no axis flip. Introducing one here would silently mirror the route.
 */
const metresToPlannerCell = (x_m: number, y_m: number, terrain: PlannerTerrain): PlannerCell => [
  clampCell(Math.round(x_m / terrain.gsd_m), terrain.dem_width),
  clampCell(Math.round(y_m / terrain.gsd_m), terrain.dem_height),
]

const plannerCellToMetres = (cell: PlannerCell, terrain: PlannerTerrain) => ({
  x_m: cell[0] * terrain.gsd_m,
  y_m: cell[1] * terrain.gsd_m,
})

function App() {
  const [dataMode, setDataMode] = useState<DataMode>('DEMO')
  // Whether data/raw actually holds a bundle the simulation can run on. The
  // backend decides this; offering the mode without it would only fail later.
  const [realRawAvailable, setRealRawAvailable] = useState(false)
  // Same idea for REAL_LOCAL: it needs prepared sector products under
  // data/sector, and offering the mode without them only produces a 404 and a
  // terrain view that silently stays on whatever mode was active before.
  const [realLocalAvailable, setRealLocalAvailable] = useState(false)
  const [terrainData, setTerrainData] = useState<PlannerTerrain | null>(null)
  const [route, setRoute] = useState<RoutePoint[]>([])
  const [routeMetrics, setRouteMetrics] = useState<RouteMetrics | null>(null)
  const [routeStart, setRouteStart] = useState<{ cell: PlannerCell; source: RouteStartSource } | null>(null)
  const [routeError, setRouteError] = useState<string | null>(null)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [activeLayers, setActiveLayers] = useState<Record<TerrainLayer, boolean>>({
    Elevation: true,
    Slope: false,
    Roughness: false,
    Traversability: false,
    Hazards: false,
    Route: true
  })

  const [cameraMode, setCameraMode] = useState<CameraMode>('Orbit')
  const [roverPlaying, setRoverPlaying] = useState(false)
  // Bumped to teleport the rover back to the selected site.
  const [roverReset, setRoverReset] = useState(0)
  const [driveMode, setDriveMode] = useState<DriveMode>('free')
  const [startSiteId, setStartSiteId] = useState<string>(START_SITES[2].id)
  const [telemetry, setTelemetry] = useState<RoverTelemetry | null>(null)

  // Perception / localization
  const [perception, setPerception] = useState<PerceptionResult | null>(null)
  const [perceptionLoading, setPerceptionLoading] = useState(false)
  const [perceptionError, setPerceptionError] = useState<string | null>(null)
  const [referenceMap, setReferenceMap] = useState<ReferenceMap | null>(null)
  const [inspectorView, setInspectorView] = useState<InspectorView>('image')
  // Landing page and working console are distinct screens, not one blended view.
  const [appView, setAppView] = useState<'home' | 'console'>('home')
  const [showLandmarks, setShowLandmarks] = useState(false)

  // Real-data capabilities and registration
  const [capabilities, setCapabilities] = useState<DataCapabilities | null>(null)
  const [registration, setRegistration] = useState<RegistrationResult | null>(null)
  const [registrationLoading, setRegistrationLoading] = useState(false)
  const [registrationError, setRegistrationError] = useState<string | null>(null)

  // Single source of truth for where the rover is, written by the 3D loop.
  const roverPoseRef = useRef<RoverPose | null>(null)
  // Written every frame; read on a timer, because rendering the HUD at frame
  // rate would re-render the whole console sixty times a second.
  const telemetryRef = useRef<RoverTelemetry | null>(null)

  useEffect(() => {
    const tick = setInterval(() => setTelemetry(telemetryRef.current), 100)
    return () => clearInterval(tick)
  }, [])

  const activeSite = START_SITES.find((site) => site.id === startSiteId) ?? START_SITES[0]

  /*
   * Memoised deliberately. The HUD refreshes ten times a second, so this
   * component re-renders constantly; a fresh object here would be a new prop
   * identity every time, re-firing the rover's placement effect and teleporting
   * it back to the start before it could move. Driving looked completely dead.
   */
  const startPoint = useMemo(
    () => (terrainData ? siteToMetres(activeSite, terrainData) : null),
    [activeSite, terrainData]
  )

  const startSiteMarkers = useMemo(
    () =>
      terrainData
        ? START_SITES.map((site) => ({ site, ...siteToMetres(site, terrainData) }))
        : [],
    [terrainData]
  )

  /**
   * Move the rover to a site and replan from it.
   *
   * The route start follows the rover rather than staying on the demo cell, so
   * picking a site changes both where you are and what the planner solved.
   */
  const selectStartSite = async (site: StartSite) => {
    setStartSiteId(site.id)
    setRoverPlaying(false)
    setRoverReset((n) => n + 1)
    if (!terrainData) return
    const metres = siteToMetres(site, terrainData)
    await fetchRoute(
      terrainData,
      metresToPlannerCell(metres.x_m, metres.y_m, terrainData),
      'start-site'
    )
  }

  useEffect(() => {
    fetchStatusAndTerrain()
    fetchReferenceMap()
    fetchCapabilities()
  }, [])

  const fetchStatusAndTerrain = async () => {
    try {
      setLoading(true)
      setError(null)

      const statusRes = await fetch(`${API_BASE}/data/status`)
      const status = await statusRes.json()
      setDataMode(status.mode as DataMode)
      setRealRawAvailable(Boolean(status.real_raw_available))
      setRealLocalAvailable(Boolean(status.real_local_available))

      const terrainRes = await fetch(`${API_BASE}/terrain?max_grid_size=100`)
      if (!terrainRes.ok) throw new Error("Failed to fetch terrain")

      const terrain: PlannerTerrain = await terrainRes.json()
      setTerrainData(terrain)
      // No localization fix exists on first load, so the route starts from the
      // labelled demo cell.
      await fetchRoute(terrain, DEMO_START_CELL, 'demo-fallback')

    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchCapabilities = async () => {
    try {
      const res = await fetch(`${API_BASE}/data/capabilities`)
      if (res.ok) setCapabilities(await res.json())
    } catch (err: any) {
      console.warn(err)
    }
  }

  const runRegistration = async (pair: 'source_reference' | 'self_check') => {
    try {
      setRegistrationLoading(true)
      setRegistrationError(null)
      setInspectorView('registration')
      const res = await fetch(`${API_BASE}/registration/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pair, max_dimension_px: 1400 })
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        throw new Error(detail.detail || 'Registration failed')
      }
      setRegistration(await res.json())
    } catch (err: any) {
      setRegistrationError(err.message)
    } finally {
      setRegistrationLoading(false)
    }
  }

  const fetchReferenceMap = async () => {
    try {
      const res = await fetch(`${API_BASE}/reference_map`)
      if (!res.ok) return
      setReferenceMap(await res.json())
    } catch (err: any) {
      console.warn(err)
    }
  }

  const setBackendMode = async (mode: DataMode) => {
    try {
      setLoading(true)
      setError(null)
      const res = await fetch(`${API_BASE}/data/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      })
      if (!res.ok) {
        const errorData = await res.json()
        throw new Error(errorData.detail || "Failed to set mode")
      }

      setRoute([])
      setRouteMetrics(null)
      setRouteStart(null)
      setRouteError(null)
      setPerception(null)
      await fetchStatusAndTerrain()
      await fetchReferenceMap()
    } catch (err: any) {
      setError(err.message)
      setLoading(false)
    }
  }

  // Takes terrain as an argument: reading it from state here would see the
  // pre-update value on the initial load and silently drop the route.
  const fetchRoute = async (
    terrain: PlannerTerrain,
    startCell: PlannerCell,
    source: RouteStartSource
  ) => {
    try {
      setRouteError(null)

      const goal: PlannerCell = [
        clampCell(DEMO_GOAL_CELL[0], terrain.dem_width),
        clampCell(DEMO_GOAL_CELL[1], terrain.dem_height),
      ]

      const res = await fetch(`${API_BASE}/plan_path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: startCell, goal })
      })

      if (!res.ok) {
        // 400 carries the planner's own reason, e.g. an impassable start cell.
        const detail = await res.json().catch(() => ({}))
        throw new Error(detail.detail || `Path planning failed (${res.status})`)
      }

      const data = await res.json()
      // Rescale the planner's full-resolution path onto the display grid, using
      // the DEM size reported by the backend rather than an assumed one.
      const colSpan = Math.max(1, terrain.dem_width - 1)
      const rowSpan = Math.max(1, terrain.dem_height - 1)
      const scaledRoute = data.path.map((p: number[]) => ({
        x: (p[0] / colSpan) * (terrain.grid.width - 1),
        y: (p[1] / rowSpan) * (terrain.grid.height - 1)
      }))
      setRoute(scaledRoute)
      setRouteMetrics(data.metrics ?? null)
      setRouteStart({ cell: startCell, source })
    } catch (err: any) {
      // The previous route and its start label are left untouched: replacing a
      // failed localization-based plan with the demo route would misrepresent
      // which position the shown route was planned from.
      setRouteError(err.message)
      console.warn(err)
    }
  }

  const runPerception = async () => {
    if (!terrainData) return
    try {
      setPerceptionLoading(true)
      setPerceptionError(null)

      // Observe from wherever the rover currently is; fall back to the demo
      // start so the pipeline is runnable before the rover has moved.
      const fallback = plannerCellToMetres(DEMO_START_CELL, terrainData)
      const pose: RoverPose = roverPoseRef.current ?? {
        x_m: fallback.x_m,
        y_m: fallback.y_m,
        heading_deg: 45,
      }

      const res = await fetch(`${API_BASE}/perception`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pose)
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        throw new Error(detail.detail || 'Perception pipeline failed')
      }

      const result: PerceptionResult = await res.json()
      setPerception(result)
      setShowLandmarks(true)

      // Replan from where localization says the rover is, not from where the
      // simulation knows it is. A no-fix result is reported, never quietly
      // replaced by the ground-truth pose.
      const loc = result.localization
      if (loc.estimated_x_m !== null && loc.estimated_y_m !== null) {
        const startCell = metresToPlannerCell(loc.estimated_x_m, loc.estimated_y_m, terrainData)
        await fetchRoute(terrainData, startCell, 'localization')
      } else {
        // Say WHY. The backend reports the stage that failed and its own
        // account of the failure; dropping both leaves the operator with a
        // dead end and no idea whether to move the rover, switch data mode, or
        // report a bug.
        const failed = result.stages.find((s) => s.status === 'failed')
        const reason = result.localization.note || failed?.detail || 'no reason reported'
        const where = failed ? `${failed.label}: ` : ''
        setRouteError(
          `Localization returned no fix, so the route start was not updated. ` +
          `${where}${reason} ` +
          `(${result.localization.n_candidates} crater candidates, ` +
          `${result.localization.n_reference_in_range} landmarks in range). ` +
          `The displayed route is still the previous one.`
        )
      }
    } catch (err: any) {
      setPerceptionError(err.message)
    } finally {
      setPerceptionLoading(false)
    }
  }

  const selectBaseLayer = (layer: TerrainLayer) => {
    setActiveLayers(prev => ({
      ...prev,
      Elevation: layer === 'Elevation',
      Slope: layer === 'Slope',
      Roughness: layer === 'Roughness',
      Traversability: layer === 'Traversability',
    }))
  }

  const toggleOverlay = (layer: TerrainLayer) => {
    setActiveLayers(prev => ({ ...prev, [layer]: !prev[layer] }))
  }

  const openView = (view: InspectorView) => setInspectorView(view)

  const loc = perception?.localization ?? null

  // How far through OBSERVE -> DETECT -> LOCALIZE -> NAVIGATE the run has got.
  // Driven by actual results, not by which panel happens to be open.
  const stage: 'observe' | 'detect' | 'localize' | 'navigate' =
    routeStart?.source === 'localization'
      ? 'navigate'
      : loc?.estimated_x_m != null
        ? 'localize'
        : perception && perception.candidates.length > 0
          ? 'detect'
          : 'observe'

  return (
    <>
      {appView === 'home' && (
        <HomeScreen
          terrain={terrainData}
          onEnter={(view) => { setInspectorView(view); setAppView('console') }}
        />
      )}

      <div className="canvas-container">
        {loading && appView === 'console' && (
          <div className="loading-overlay">Loading terrain...</div>
        )}

        <Canvas camera={{ position: [0, 300, 660], fov: 52, near: 0.1, far: 5000 }}>
          <color attach="background" args={['#0b0b0d']} />
          <hemisphereLight args={['#c8cfdb', '#17171c', 1.0]} />
          <ambientLight intensity={0.62} color="#dfe3ea" />
          <directionalLight position={[600, 900, 400]} intensity={2.9} color="#ffffff" />
          <directionalLight position={[-500, 400, -600]} intensity={0.6} color="#7d8798" />
          <Stars radius={3000} depth={200} count={6000} factor={6} saturation={0} fade speed={1} />

          <Suspense fallback={null}>
            <Terrain3D
              data={terrainData}
              activeLayers={activeLayers}
              route={route}
              exaggeration={2.0}
              cameraMode={cameraMode}
              localization={loc}
              referenceCraters={referenceMap?.craters}
              showLandmarks={showLandmarks}
              startSites={startSiteMarkers}
              activeStartSiteId={startSiteId}
            />

            <RoverView
              data={terrainData}
              route={route}
              exaggeration={2.0}
              cameraMode={cameraMode}
              driveMode={driveMode}
              isPlaying={roverPlaying}
              resetSignal={roverReset}
              startPoint={startPoint}
              poseRef={roverPoseRef}
              telemetryRef={telemetryRef}
            />
          </Suspense>

          {cameraMode === 'Orbit' && (
            <OrbitControls
              makeDefault
              minPolarAngle={0}
              maxPolarAngle={Math.PI / 2 - 0.05}
              maxDistance={2000}
            />
          )}
        </Canvas>

        {appView === 'console' && telemetry && (
          <div className="rover-hud">
            <div className="hud-speed">
              <span className="hud-number">{Math.abs(telemetry.speed_mps).toFixed(1)}</span>
              <span className="hud-unit">m/s</span>
            </div>
            <div className="hud-grid">
              <div><span>Position</span><b>{telemetry.x_m.toFixed(0)}, {telemetry.y_m.toFixed(0)} m</b></div>
              <div><span>Heading</span><b>{telemetry.heading_deg.toFixed(0)}&deg;</b></div>
              <div><span>Elevation</span><b>{telemetry.elevation_m.toFixed(1)} m</b></div>
              <div><span>Slope</span><b className={telemetry.slope_deg > 18 ? 'hud-warn' : ''}>{telemetry.slope_deg.toFixed(1)}&deg;</b></div>
              <div><span>Driven</span><b>{(telemetry.distance_m / 1000).toFixed(2)} km</b></div>
              {telemetry.traversability !== null && (
                <div>
                  <span>Ground</span>
                  <b className={telemetry.traversability < 0.25 ? 'hud-warn' : ''}>
                    {telemetry.traversability <= 0.001
                      ? 'impassable'
                      : `${(telemetry.traversability * 100).toFixed(0)}%`}
                  </b>
                </div>
              )}
            </div>
            {telemetry.mode === 'free' && (
              <div className="hud-keys">WASD / arrows &middot; space to brake</div>
            )}
          </div>
        )}
      </div>

      {appView === 'console' && (<>
      <header className="top-bar">
        <button className="brand" onClick={() => setAppView('home')}>INDU</button>
        <nav className="top-nav">
          <span className={stage === 'observe' ? 'on' : ''}>Observe</span>
          <span className={stage === 'detect' ? 'on' : ''}>Detect</span>
          <span className={stage === 'localize' ? 'on' : ''}>Localize</span>
          <span className={stage === 'navigate' ? 'on' : ''}>Navigate</span>
          <span className={dataMode !== 'DEMO' ? 'on' : ''}>{dataMode}</span>
          <div className="burger"><i /><i /><i /></div>
        </nav>
      </header>

      <div className="mission-control">
        <h1>Lunar Navigation MVP</h1>
        <div className="pipeline-trail">
          OBSERVE → DETECT → MAP → LOCALIZE → ANALYZE → NAVIGATE
        </div>

        <div className="section">
          <h2>Data</h2>
          <div className="button-group row">
            <button
              className={dataMode === 'DEMO' ? 'active' : ''}
              onClick={() => setBackendMode('DEMO')}
            >
              Demo (Synthetic)
            </button>
            <button
              className={dataMode === 'REAL_LOCAL' ? 'active' : ''}
              onClick={() => setBackendMode('REAL_LOCAL')}
              disabled={!realLocalAvailable}
              title={
                realLocalAvailable
                  ? 'Use prepared sector products from data/sector'
                  : 'No prepared sector products under data/sector. Build them with scripts/build_sector.py and scripts/build_terrain.py.'
              }
            >
              Real Local
            </button>
            <button
              className={dataMode === 'REAL_RAW' ? 'active' : ''}
              onClick={() => setBackendMode('REAL_RAW')}
              disabled={!realRawAvailable}
              title={
                realRawAvailable
                  ? 'Run the simulation on Chandrayaan-2 OHRC imagery from data/raw'
                  : 'No Chandrayaan-2 bundle was found under data/raw'
              }
            >
              Real (data/raw)
            </button>
          </div>
          {error && <div className="error-message">{error}</div>}
        </div>

        <div className="section">
          <h2>Observation</h2>
          <div className="button-group row">
            <button className={inspectorView === 'image' ? 'active' : ''} onClick={() => openView('image')}>Rover Image</button>
            <button className={inspectorView === 'quadrants' ? 'active' : ''} onClick={() => openView('quadrants')}>4 Quadrants</button>
            <button className={inspectorView === 'points' ? 'active' : ''} onClick={() => openView('points')}>Crater Points</button>
            <button className={inspectorView === 'craters' ? 'active' : ''} onClick={() => openView('craters')}>Crater Map</button>
          </div>
          <button className="run-button wide" onClick={runPerception} disabled={perceptionLoading || !terrainData}>
            {perceptionLoading ? 'Detecting...' : 'Detect Craters'}
          </button>
        </div>

        <div className="section">
          <h2>Localization</h2>
          <div className="button-group row">
            <button className={inspectorView === 'reference' ? 'active' : ''} onClick={() => openView('reference')}>Reference Map</button>
            <button className={inspectorView === 'match' ? 'active' : ''} onClick={() => openView('match')}>Match</button>
            <button className={inspectorView === 'position' ? 'active' : ''} onClick={() => openView('position')}>Estimated Position</button>
          </div>
          <div className="button-group row">
            <button className={showLandmarks ? 'active' : ''} onClick={() => setShowLandmarks(v => !v)}>
              Landmarks in 3D
            </button>
          </div>
        </div>

        <div className="section">
          <h2>Registration</h2>
          <div className="button-group row">
            <button className={inspectorView === 'registration' ? 'active' : ''} onClick={() => openView('registration')}>
              Status &amp; Results
            </button>
          </div>
          <button className="run-button wide" onClick={() => runRegistration('source_reference')} disabled={registrationLoading}>
            {registrationLoading ? 'Registering...' : 'Register Source vs Reference'}
          </button>
        </div>

        <div className="section">
          <h2>Terrain</h2>
          <div className="button-group row">
            <button className={activeLayers.Elevation ? 'active' : ''} onClick={() => selectBaseLayer('Elevation')}>Elevation</button>
            <button className={activeLayers.Slope ? 'active' : ''} onClick={() => selectBaseLayer('Slope')}>Slope</button>
            <button className={activeLayers.Roughness ? 'active' : ''} onClick={() => selectBaseLayer('Roughness')}>Roughness</button>
            <button className={activeLayers.Traversability ? 'active' : ''} onClick={() => selectBaseLayer('Traversability')}>Traversability</button>
          </div>
        </div>

        <div className="section">
          <h2>Navigation</h2>
          <div className="button-group row">
            <button className={activeLayers.Hazards ? 'active' : ''} onClick={() => toggleOverlay('Hazards')}>Hazards</button>
            <button className={activeLayers.Route ? 'active' : ''} onClick={() => toggleOverlay('Route')}>A* Route</button>
          </div>
          {routeError && <div className="error-message">{routeError}</div>}
        </div>

        <div className="section">
          <h2>Camera</h2>
          <div className="button-group row">
            <button className={cameraMode === 'Orbit' ? 'active' : ''} onClick={() => setCameraMode('Orbit')}>Orbit</button>
            <button className={cameraMode === 'Rover POV' ? 'active' : ''} onClick={() => setCameraMode('Rover POV')}>Rover POV</button>
          </div>
        </div>

        <div className="section">
          <h2>Rover</h2>
          <div className="button-group row">
            <button
              className={driveMode === 'free' ? 'active' : ''}
              onClick={() => setDriveMode('free')}
              title="Steer the rover yourself"
            >
              Free drive
            </button>
            <button
              className={driveMode === 'follow' ? 'active' : ''}
              onClick={() => setDriveMode('follow')}
              title="Replay the planned A* route"
            >
              Follow route
            </button>
          </div>

          {driveMode === 'free' ? (
            <div className="drive-help">
              <span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> or arrows to drive</span>
              <span><kbd>Space</kbd> to brake</span>
            </div>
          ) : (
            <div className="button-group row">
              <button onClick={() => setRoverPlaying(!roverPlaying)} disabled={route.length === 0}>
                {roverPlaying ? 'Pause' : 'Play'}
              </button>
            </div>
          )}

          <div className="button-group row">
            <button onClick={() => { setRoverPlaying(false); setRoverReset((n) => n + 1) }}>
              Reset to {activeSite.label}
            </button>
          </div>
        </div>

        <div className="section">
          <h2>Start point</h2>
          <div className="button-group wrap">
            {START_SITES.map((site) => (
              <button
                key={site.id}
                className={site.id === startSiteId ? 'active' : ''}
                onClick={() => selectStartSite(site)}
                disabled={!terrainData}
              >
                {site.label}
              </button>
            ))}
          </div>
          <div className="hint">
            Moves the rover there and replans the route from that position.
          </div>
        </div>

        {terrainData && (
          <div className="section">
            <h2>Terrain metrics</h2>
            <div className="metric-card">
              <div className="metric-label">Elevation Range (m)</div>
              <div className="metric-value">{terrainData.min_elevation.toFixed(1)} to {terrainData.max_elevation.toFixed(1)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Max Slope (deg)</div>
              <div className="metric-value">{terrainData.max_slope.toFixed(1)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Grid Downsample</div>
              <div className="metric-value">{terrainData.grid.width} x {terrainData.grid.height}</div>
            </div>
            {terrainData.cost_stats && (
              <div className="metric-card">
                <div className="metric-label">Impassable / High cost</div>
                <div className="metric-value">
                  {(terrainData.cost_stats.impassable_fraction * 100).toFixed(1)}% / {(terrainData.cost_stats.high_cost_fraction * 100).toFixed(1)}%
                </div>
              </div>
            )}
          </div>
        )}

        {perception && (
          <div className="section">
            <h2>Observation metrics</h2>
            <div className="metric-card">
              <div className="metric-label">Quadrants</div>
              <div className="metric-value">{perception.quadrants.length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Points detected</div>
              <div className="metric-value">{perception.points.length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Aggregated craters</div>
              <div className="metric-value">{perception.candidates.length}</div>
            </div>
          </div>
        )}

        {loc && (
          <div className="section">
            <h2>Localization metrics</h2>
            <div className="metric-card">
              <div className="metric-label">Matched craters</div>
              <div className="metric-value">{loc.matches.length} / {loc.n_candidates}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Position error (synthetic)</div>
              <div className="metric-value">
                {loc.position_error_m === null ? '--' : `${loc.position_error_m.toFixed(2)} m`}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Synthetic match confidence</div>
              <div className="metric-value">
                {loc.match_confidence === null ? '--' : loc.match_confidence.toFixed(3)}
              </div>
            </div>
          </div>
        )}

        {routeMetrics && (
          <div className="section">
            <h2>Navigation metrics</h2>
            {routeStart && (
              <div className="metric-card">
                <div className="metric-label">
                  {routeStart.source === 'localization'
                    ? 'A* start (from localization)'
                    : 'A* start (demo fallback)'}
                </div>
                <div className="metric-value">
                  cell {routeStart.cell[0]}, {routeStart.cell[1]}
                </div>
              </div>
            )}
            <div className="metric-card">
              <div className="metric-label">Path length</div>
              <div className="metric-value">{routeMetrics.length_m.toFixed(0)} m</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Max route slope</div>
              <div className="metric-value">
                {routeMetrics.max_slope_deg === null ? '--' : `${routeMetrics.max_slope_deg.toFixed(1)}°`}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">High-cost cells on route</div>
              <div className="metric-value">{routeMetrics.high_cost_cells}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Route waypoints</div>
              <div className="metric-value">{routeMetrics.n_waypoints}</div>
            </div>
          </div>
        )}
      </div>

      <div className="orbit-controls-hint">
        <button
          className="round-btn"
          title="Orbit camera"
          onClick={() => setCameraMode('Orbit')}
        >
          ▲
        </button>
        <button
          className="round-btn"
          title="Rover POV"
          onClick={() => setCameraMode('Rover POV')}
        >
          ▼
        </button>
      </div>

      <InspectorPanel
        view={inspectorView}
        perception={perception}
        referenceMap={referenceMap}
        loading={perceptionLoading}
        error={perceptionError}
        onRun={runPerception}
        capabilities={capabilities}
        registration={registration}
        registrationLoading={registrationLoading}
        registrationError={registrationError}
        onRunRegistration={runRegistration}
      />
      </>)}
    </>
  )
}

export default App
