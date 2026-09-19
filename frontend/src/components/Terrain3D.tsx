import { useMemo, useRef } from 'react'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import type {
  LocalizationResult,
  ReferenceCrater,
  RoutePoint,
  StartSite,
  TerrainData,
  TerrainLayer,
} from '../types'
import { sampleHeight, sectorToPlane } from '../lib/terrain'

/**
 * Sector metres (origin top-left, y down) to this group's local plane space.
 *
 * Height is sampled bilinearly through the shared helper, so markers sit on
 * the surface the mesh actually draws rather than on the nearest vertex - on a
 * grid this coarse those differ by metres.
 */
const sectorToLocal = (
  data: TerrainData,
  x_m: number,
  y_m: number,
  exaggeration: number
): [number, number, number] =>
  sectorToPlane(data, x_m, y_m, sampleHeight(data, x_m, y_m, exaggeration))

interface Terrain3DProps {
  data: TerrainData | null
  activeLayers: Record<TerrainLayer, boolean>
  exaggeration?: number
  route?: RoutePoint[]
  startSites?: { site: StartSite; x_m: number; y_m: number }[]
  activeStartSiteId?: string
  roverPos?: THREE.Vector3
  cameraMode?: 'Orbit' | 'Rover POV'
  localization?: LocalizationResult | null
  referenceCraters?: ReferenceCrater[]
  showLandmarks?: boolean
}

export const Terrain3D: React.FC<Terrain3DProps> = ({
  data,
  activeLayers,
  exaggeration = 2,
  route,
  localization,
  referenceCraters,
  showLandmarks = false,
  startSites,
  activeStartSiteId,
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
    const { grid, elevation, slope, roughness, hazards, traversability, min_elevation, max_elevation, min_slope, max_slope } = data
    const src = document.createElement('canvas')
    src.width = grid.width
    src.height = grid.height
    const sctx = src.getContext('2d')
    if (!sctx) return null

    const imgData = sctx.createImageData(grid.width, grid.height)

    for (let i = 0; i < grid.width * grid.height; i++) {
      let r = 0, g = 0, b = 0

      if (activeLayers.Hazards && hazards[i] > 0.5) {
        // Red for hazards
        r = 255; g = 60; b = 60
      } else if (activeLayers.Slope) {
        // Heatmap for slope
        const norm = (slope[i] - min_slope) / (max_slope - min_slope + 0.001)
        r = Math.floor(60 + norm * 195)
        g = Math.floor(220 - norm * 190)
        b = 70
      } else if (activeLayers.Roughness) {
        // Blue/purple for roughness
        const roughNorm = Math.min(1, roughness[i] * 10)
        r = Math.floor(70 + roughNorm * 185)
        g = 60
        b = 235
      } else if (activeLayers.Traversability && traversability) {
        // Green = easy ground, red = costly, dark = impassable to the planner.
        const t = traversability[i]
        if (t <= 0.001) {
          r = 30; g = 30; b = 38
        } else if (t < 0.5) {
          r = 235
          g = Math.floor(60 + (t / 0.5) * 150)
          b = 55
        } else {
          r = Math.floor(235 - ((t - 0.5) / 0.5) * 165)
          g = 210
          b = 70
        }
      } else {
        // Regolith albedo ramp. Elevation only modulates brightness within the
        // gray range: a pure-black low point would swallow all incident light.
        const norm = activeLayers.Elevation
          ? (elevation[i] - min_elevation) / (max_elevation - min_elevation + 0.001)
          : 0.5
        r = Math.floor(92 + norm * 116)
        g = Math.floor(89 + norm * 113)
        b = Math.floor(83 + norm * 107)
      }

      const px = i * 4
      imgData.data[px] = r
      imgData.data[px + 1] = g
      imgData.data[px + 2] = b
      imgData.data[px + 3] = 255
    }

    sctx.putImageData(imgData, 0, 0)

    const SCALE = 6
    const canvas = document.createElement('canvas')
    canvas.width = grid.width * SCALE
    canvas.height = grid.height * SCALE
    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(src, 0, 0, canvas.width, canvas.height)

    const grain = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const gd = grain.data
    for (let p = 0; p < gd.length; p += 4) {
      const n = (Math.random() - 0.5) * 34
      gd[p] = Math.max(0, Math.min(255, gd[p] + n))
      gd[p + 1] = Math.max(0, Math.min(255, gd[p + 1] + n))
      gd[p + 2] = Math.max(0, Math.min(255, gd[p + 2] + n))
    }
    ctx.putImageData(grain, 0, 0)

    const tex = new THREE.CanvasTexture(canvas)
    tex.colorSpace = THREE.SRGBColorSpace
    tex.magFilter = THREE.LinearFilter
    tex.minFilter = THREE.LinearMipmapLinearFilter
    tex.anisotropy = 8
    return tex
  }, [data, activeLayers])

  // 3. Convert Route to 3D Line
  const routePoints3D = useMemo(() => {
    if (!data || !route || route.length === 0 || !activeLayers.Route) return []
    const { width_m, height_m, grid } = data

    const pts: THREE.Vector3[] = []

    for (const pt of route) {
      // Route points arrive on the display grid; convert to sector metres and
      // let the shared helper place them, so the line and the rover driving it
      // cannot end up on different surfaces.
      const x_m = (pt.x / Math.max(1, grid.width - 1)) * width_m
      const y_m = (pt.y / Math.max(1, grid.height - 1)) * height_m
      const [localX, localY, h] = sectorToLocal(data, x_m, y_m, exaggeration)
      // Lifted just clear of the surface so the line is not z-fought by it.
      pts.push(new THREE.Vector3(localX, localY, h + 1.2))
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
          roughness={1.0}
          metalness={0.0}
        />
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
      
      {/* Places the rover can be dropped. The active one is filled. */}
      {startSites?.map(({ site, x_m, y_m }) => {
        const [lx, ly, h] = sectorToLocal(data, x_m, y_m, exaggeration)
        const active = site.id === activeStartSiteId
        return (
          <group key={`start-${site.id}`} position={[lx, ly, h]}>
            <mesh position={[0, 0, 1]}>
              <ringGeometry args={[9, 13, 24]} />
              <meshBasicMaterial
                color={active ? '#7ee787' : '#68707f'}
                transparent
                opacity={active ? 0.95 : 0.5}
                side={THREE.DoubleSide}
              />
            </mesh>
            {active && (
              <mesh position={[0, 0, 16]}>
                <coneGeometry args={[5, 14, 4]} />
                <meshBasicMaterial color="#7ee787" transparent opacity={0.85} />
              </mesh>
            )}
          </group>
        )
      })}

      {/* Reference landmarks used for localization; matched ones stand out. */}
      {showLandmarks &&
        referenceCraters?.map((c) => {
          const [lx, ly, h] = sectorToLocal(data, c.x_m, c.y_m, exaggeration)
          const matched = localization?.matches.some((m) => m.reference_id === c.id)
          return (
            <mesh key={`landmark-${c.id}`} position={[lx, ly, h + 1.5]}>
              <ringGeometry args={[Math.max(0, c.diameter_m / 2 - 2.5), c.diameter_m / 2, 24]} />
              <meshBasicMaterial
                // Matched vs unmatched separated by brightness, not hue, so the
                // lunar surface stays monochrome.
                color={matched ? '#ffffff' : '#5c6270'}
                transparent
                opacity={matched ? 0.95 : 0.35}
                side={THREE.DoubleSide}
              />
            </mesh>
          )
        })}

      {/* Ground truth rover position (synthetic mode only). */}
      {localization?.ground_truth_x_m != null && (
        <mesh
          position={sectorToLocal(
            data,
            localization.ground_truth_x_m,
            localization.ground_truth_y_m as number,
            exaggeration
          )}
        >
          <sphereGeometry args={[6, 16, 16]} />
          <meshBasicMaterial color="#7ee787" transparent opacity={0.8} />
        </mesh>
      )}

      {/* Position recovered by the localization pipeline. */}
      {localization?.estimated_x_m != null && (
        <mesh
          position={sectorToLocal(
            data,
            localization.estimated_x_m,
            localization.estimated_y_m as number,
            exaggeration
          )}
        >
          <coneGeometry args={[6, 18, 4]} />
          <meshBasicMaterial color="#ffb347" />
        </mesh>
      )}

      {/* Craters */}
      {data.craters?.map((c, i) => {
        // Backend crater x/y are metres from the sector's top-left corner.
        const [localX, localY, h] = sectorToLocal(data, c.x, c.y, exaggeration)
        const radius_m = c.diameter_m / 2

        return (
          <mesh key={c.id ?? `crater-${i}`} position={[localX, localY, h + 1.0]}>
            <ringGeometry args={[Math.max(0, radius_m - 2), radius_m, 32]} />
            <meshBasicMaterial color="#b9bec9" transparent opacity={0.55} side={THREE.DoubleSide} />
          </mesh>
        )
      })}
    </group>
  )
}
