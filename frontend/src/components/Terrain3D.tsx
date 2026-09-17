import { useMemo, useRef } from 'react'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import type { TerrainData, TerrainLayer, RoutePoint } from '../types'

interface Terrain3DProps {
  data: TerrainData | null
  activeLayers: Record<TerrainLayer, boolean>
  exaggeration?: number
  route?: RoutePoint[]
  roverPos?: THREE.Vector3
  cameraMode?: 'Orbit' | 'Rover POV'
}

export const Terrain3D: React.FC<Terrain3DProps> = ({ 
  data, 
  activeLayers, 
  exaggeration = 2,
  route,
}) => {
  const meshRef = useRef<THREE.Mesh>(null)

  // 1. Generate Geometry
  const geometry = useMemo(() => {
    if (!data) return null
    const { width_m, height_m, grid, elevation, min_elevation } = data
    // PlaneGeometry creates vertices left-to-right, top-to-bottom
    const geo = new THREE.PlaneGeometry(width_m, height_m, grid.width - 1, grid.height - 1)
    
    const pos = geo.attributes.position
    for (let i = 0; i < pos.count; i++) {
      // Elevate relative to minimum elevation to keep it near origin
      const h = (elevation[i] - min_elevation) * exaggeration
      // PlaneGeometry is on XY plane, so Z is height
      pos.setZ(i, h)
    }
    geo.computeVertexNormals()
    return geo
  }, [data, exaggeration])

  // 2. Generate Texture
  const texture = useMemo(() => {
    if (!data) return null
    const { grid, elevation, slope, roughness, hazards, min_elevation, max_elevation, min_slope, max_slope } = data
    const canvas = document.createElement('canvas')
    canvas.width = grid.width
    canvas.height = grid.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    
    const imgData = ctx.createImageData(grid.width, grid.height)
    
    for (let i = 0; i < grid.width * grid.height; i++) {
      let r = 0, g = 0, b = 0
      
      if (activeLayers.Hazards && hazards[i] > 0.5) {
        // Red for hazards
        r = 255; g = 50; b = 50
      } else if (activeLayers.Slope) {
        // Heatmap for slope
        const norm = (slope[i] - min_slope) / (max_slope - min_slope + 0.001)
        r = Math.floor(norm * 255)
        g = Math.floor((1 - norm) * 255)
        b = 50
      } else if (activeLayers.Roughness) {
        // Blue/purple for roughness
        const roughNorm = Math.min(1, roughness[i] * 10)
        r = Math.floor(roughNorm * 200)
        g = 50
        b = 255
      } else if (activeLayers.Elevation) {
        // Grayscale for elevation
        const norm = (elevation[i] - min_elevation) / (max_elevation - min_elevation + 0.001)
        r = g = b = Math.floor(norm * 255)
      } else {
        // Default dark gray base
        r = g = b = 40
      }
      
      const px = i * 4
      imgData.data[px] = r
      imgData.data[px + 1] = g
      imgData.data[px + 2] = b
      imgData.data[px + 3] = 255
    }
    
    ctx.putImageData(imgData, 0, 0)
    const tex = new THREE.CanvasTexture(canvas)
    tex.magFilter = THREE.NearestFilter // Blocky is good for grid data
    return tex
  }, [data, activeLayers])

  // 3. Convert Route to 3D Line
  const routePoints3D = useMemo(() => {
    if (!data || !route || route.length === 0 || !activeLayers.Route) return []
    const { width_m, height_m, grid, elevation, min_elevation } = data
    
    const pts: THREE.Vector3[] = []
    
    for (const pt of route) {
      // pt.x and pt.y are grid coordinates (0 to grid.width-1)
      const nx = pt.x / (grid.width - 1)
      const ny = pt.y / (grid.height - 1)
      
      // Map to PlaneGeometry local space (centered at 0,0)
      const localX = (nx - 0.5) * width_m
      const localY = -(ny - 0.5) * height_m // Plane top is +Y, image top is 0
      
      const idx = pt.y * grid.width + pt.x
      const h = (elevation[idx] - min_elevation) * exaggeration
      
      // In Three.js world space (after Plane is rotated -PI/2 on X):
      // localX -> world X
      // localY -> world -Z
      // h -> world Y
      
      // We're drawing the line inside the same coordinate space as the Plane, 
      // before rotation, so we just use localX, localY, h
      pts.push(new THREE.Vector3(localX, localY, h + 2.0)) // Slightly above terrain
    }
    
    return pts
  }, [data, route, activeLayers, exaggeration])

  if (!data || !geometry) return null

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* Terrain Mesh */}
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial 
          map={texture} 
          wireframe={false} 
          roughness={0.8}
        />
      </mesh>
      
      {/* Wireframe overlay for grid context */}
      <mesh geometry={geometry}>
        <meshBasicMaterial color="#333" wireframe={true} transparent opacity={0.2} />
      </mesh>

      {/* Route */}
      {routePoints3D.length > 0 && (
        <Line 
          points={routePoints3D} 
          color="cyan" 
          lineWidth={3} 
        />
      )}

      {/* Start/Goal Markers */}
      {routePoints3D.length > 0 && (
        <>
          <mesh position={routePoints3D[0]}>
            <sphereGeometry args={[4, 16, 16]} />
            <meshBasicMaterial color="lime" />
          </mesh>
          <mesh position={routePoints3D[routePoints3D.length - 1]}>
            <sphereGeometry args={[4, 16, 16]} />
            <meshBasicMaterial color="red" />
          </mesh>
        </>
      )}
      
      {/* Craters */}
      {data.craters?.map((c, i) => {
        const nx = c.x / (data.grid.width - 1)
        const ny = c.y / (data.grid.height - 1)
        const localX = (nx - 0.5) * data.width_m
        const localY = -(ny - 0.5) * data.height_m
        const idx = Math.floor(c.y) * data.grid.width + Math.floor(c.x)
        const h = (data.elevation[idx] - data.min_elevation) * exaggeration
        const radius_m = c.radius * (data.width_m / data.grid.width)
        
        return (
          <mesh key={`crater-${i}`} position={[localX, localY, h + 1.0]}>
            <ringGeometry args={[radius_m - 2, radius_m, 32]} />
            <meshBasicMaterial color="orange" transparent opacity={0.6} side={THREE.DoubleSide} />
          </mesh>
        )
      })}
    </group>
  )
}
