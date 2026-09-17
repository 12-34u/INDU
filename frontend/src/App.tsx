import { useState, useEffect, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'
import { Terrain3D } from './components/Terrain3D'
import { RoverView } from './components/RoverView'
import type { DataMode, TerrainLayer, CameraMode, TerrainData, RoutePoint } from './types'
import './App.css'

const API_BASE = 'http://localhost:8000'

function App() {
  const [dataMode, setDataMode] = useState<DataMode>('DEMO')
  const [terrainData, setTerrainData] = useState<TerrainData | null>(null)
  const [route, setRoute] = useState<RoutePoint[]>([])
  
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [activeLayers, setActiveLayers] = useState<Record<TerrainLayer, boolean>>({
    Elevation: true,
    Slope: false,
    Roughness: false,
    Hazards: false,
    Route: true
  })

  const [cameraMode, setCameraMode] = useState<CameraMode>('Orbit')
  const [roverPlaying, setRoverPlaying] = useState(false)
  const [roverReset, setRoverReset] = useState(false)

  // Initialize
  useEffect(() => {
    fetchStatusAndTerrain()
  }, [])

  const fetchStatusAndTerrain = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const statusRes = await fetch(`${API_BASE}/data/status`)
      const status = await statusRes.json()
      setDataMode(status.mode as DataMode)

      const terrainRes = await fetch(`${API_BASE}/terrain?max_grid_size=100`)
      if (!terrainRes.ok) throw new Error("Failed to fetch terrain")
      
      const terrain = await terrainRes.json()
      setTerrainData(terrain)
      
      // Auto-fetch route to have it ready
      fetchRoute()
      
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
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
      
      setRoute([]) // Clear route on mode change
      await fetchStatusAndTerrain()
    } catch (err: any) {
      setError(err.message)
      setLoading(false)
    }
  }

  const fetchRoute = async () => {
    try {
      // Hardcoded start/goal for MVP demonstration
      const res = await fetch(`${API_BASE}/plan_path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: [100, 100], goal: [800, 800] }) // These are in the original 1000x1000 grid space!
      })
      
      if (!res.ok) throw new Error("Path not found")
      
      const data = await res.json()
      // Path is returned in full 1000x1000 coordinates.
      // The Rover/Terrain component expects coordinates in the 0..1 range scaled by grid width,
      // But since we are downsampling the grid to 100x100 for display, we need to scale the route
      // to match the original 1000x1000 space relative to the grid.
      // The backend A* uses the full DEM size. 
      // The frontend uses the small DEM (e.g. 100x100).
      // If we know the full size was 1000x1000, we map it relative to width_m.
      // Wait, Terrain3D expects route pt.x to be in downsampled grid coordinates OR we should map it as a percentage.
      // Let's modify the way we pass route points: 
      // We will map the backend path [x,y] (in 1000x1000 space) to percentages right here, assuming the original was 1000x1000.
      // Actually, we can fetch the original dims from the backend if we want, but for MVP it's fixed.
      // Let's map it to the downsampled grid dimensions.
      
      // Assume original is 1000x1000 for MVP
      const originalW = 1000
      const originalH = 1000
      
      if (terrainData) {
         const scaledRoute = data.path.map((p: number[]) => ({
            x: (p[0] / originalW) * (terrainData.grid.width - 1),
            y: (p[1] / originalH) * (terrainData.grid.height - 1)
         }))
         setRoute(scaledRoute)
      }
    } catch (err: any) {
      console.warn(err)
    }
  }

  // We need a refetch of route if terrainData loads after route attempt
  useEffect(() => {
    if (terrainData && route.length === 0 && !loading && !error) {
       fetchRoute()
    }
  }, [terrainData])

  const toggleLayer = (layer: TerrainLayer) => {
    setActiveLayers(prev => {
      // If turning on elevation, slope, roughness, or hazards, turn off the others
      const isBaseLayer = layer !== 'Route' && layer !== 'Hazards'
      if (isBaseLayer) {
        return {
          ...prev,
          Elevation: layer === 'Elevation',
          Slope: layer === 'Slope',
          Roughness: layer === 'Roughness',
          [layer]: true
        }
      }
      return { ...prev, [layer]: !prev[layer] }
    })
  }

  return (
    <>
      <div className="mission-control">
        <h1>Lunar Terrain MVP</h1>
        
        <div className="section">
          <h2>Data Mode</h2>
          <div className="button-group row">
            <button 
              className={dataMode === 'DEMO' ? 'active' : ''}
              onClick={() => setBackendMode('DEMO')}
            >
              DEMO (Synthetic)
            </button>
            <button 
              className={dataMode === 'REAL_LOCAL' ? 'active' : ''}
              onClick={() => setBackendMode('REAL_LOCAL')}
            >
              REAL LOCAL (Sector 001)
            </button>
          </div>
          {error && <div className="error-message">{error}</div>}
        </div>

        <div className="section">
          <h2>Terrain Base</h2>
          <div className="button-group row">
            <button className={activeLayers.Elevation ? 'active' : ''} onClick={() => toggleLayer('Elevation')}>Elevation</button>
            <button className={activeLayers.Slope ? 'active' : ''} onClick={() => toggleLayer('Slope')}>Slope</button>
            <button className={activeLayers.Roughness ? 'active' : ''} onClick={() => toggleLayer('Roughness')}>Roughness</button>
          </div>
        </div>

        <div className="section">
          <h2>Overlays</h2>
          <div className="button-group row">
            <button className={activeLayers.Hazards ? 'active' : ''} onClick={() => toggleLayer('Hazards')}>Hazards</button>
            <button className={activeLayers.Route ? 'active' : ''} onClick={() => toggleLayer('Route')}>A* Route</button>
          </div>
        </div>

        <div className="section">
          <h2>Camera Mode</h2>
          <div className="button-group row">
            <button className={cameraMode === 'Orbit' ? 'active' : ''} onClick={() => setCameraMode('Orbit')}>Orbit</button>
            <button className={cameraMode === 'Rover POV' ? 'active' : ''} onClick={() => setCameraMode('Rover POV')}>Rover POV</button>
          </div>
        </div>

        <div className="section">
          <h2>Rover Controls</h2>
          <div className="button-group row">
            <button onClick={() => setRoverPlaying(!roverPlaying)}>
              {roverPlaying ? 'Pause' : 'Play'}
            </button>
            <button onClick={() => { setRoverPlaying(false); setRoverReset(r => !r) }}>
              Reset
            </button>
          </div>
        </div>

        {terrainData && (
          <div className="section">
            <h2>Metrics</h2>
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
            <div className="metric-card">
              <div className="metric-label">Route Points</div>
              <div className="metric-value">{route.length}</div>
            </div>
          </div>
        )}
      </div>

      <div className="canvas-container">
        {loading && <div className="loading-overlay">Initializing DataManager...</div>}
        
        <Canvas camera={{ position: [0, 500, 800], fov: 45 }}>
          <color attach="background" args={['#050505']} />
          <ambientLight intensity={0.4} />
          <directionalLight position={[100, 500, 100]} intensity={1.5} />
          <Stars radius={1000} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
          
          <Suspense fallback={null}>
            <Terrain3D 
              data={terrainData} 
              activeLayers={activeLayers} 
              route={route}
              exaggeration={2.0}
              cameraMode={cameraMode}
            />
            
            <RoverView 
              data={terrainData}
              route={route}
              exaggeration={2.0}
              cameraMode={cameraMode}
              isPlaying={roverPlaying}
              onReset={roverReset}
            />
          </Suspense>

          {cameraMode === 'Orbit' && (
            <OrbitControls 
              makeDefault 
              minPolarAngle={0} 
              maxPolarAngle={Math.PI / 2 - 0.05} // Prevent going under the terrain
              maxDistance={2000}
            />
          )}
        </Canvas>
      </div>
    </>
  )
}

export default App
