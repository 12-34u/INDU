import React, { useRef, useMemo, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import type { DriveMode, RoutePoint, RoverPose, RoverTelemetry, TerrainData } from '../types'
import {
  roverDisplayScale,
  sampleHeight,
  sampleLayer,
  sampleNormal,
  sampleSlopeDeg,
  sectorToWorld,
  surfaceQuaternion,
} from '../lib/terrain'

/** Driving model, in sector metres and seconds. */
const MAX_SPEED_MPS = 42
const REVERSE_SPEED_MPS = 16
const ACCELERATION_MPS2 = 26
const BRAKE_MPS2 = 46
const COAST_DRAG_MPS2 = 12
const TURN_RATE_DEG = 78
// Below this the wheels have nothing to bite on, so steering stops responding.
const STEERING_CUTOFF_MPS = 0.4
// A slope the rover cannot climb. Matches the spirit of the planner's cost map
// without pretending to reproduce it.
const IMPASSABLE_SLOPE_DEG = 28

interface RoverViewProps {
  data: TerrainData | null
  route?: RoutePoint[]
  exaggeration?: number
  cameraMode?: 'Orbit' | 'Rover POV'
  driveMode?: DriveMode
  isPlaying?: boolean
  /** Changing this number teleports the rover to `startPoint`. */
  resetSignal?: number
  startPoint?: { x_m: number; y_m: number; heading_deg?: number } | null
  /**
   * Written every frame with the rover pose in sector metres. A ref rather
   * than state so the 3D loop never triggers a React render; the perception
   * request reads it on demand.
   */
  poseRef?: React.MutableRefObject<RoverPose | null>
  /** Same idea, for the HUD: sampled on a timer rather than per frame. */
  telemetryRef?: React.MutableRefObject<RoverTelemetry | null>
}

export const RoverView: React.FC<RoverViewProps> = ({
  data,
  route,
  exaggeration = 2,
  cameraMode = 'Orbit',
  driveMode = 'free',
  isPlaying = false,
  resetSignal = 0,
  startPoint = null,
  poseRef,
  telemetryRef,
}) => {
  const groupRef = useRef<THREE.Group>(null)

  // The rover's whole state lives in refs. Driving is a per-frame integration
  // and putting it in React state would re-render the tree sixty times a second.
  const pose = useRef({ x_m: 0, y_m: 0, heading_deg: 0 })
  const speed = useRef(0)
  const distance = useRef(0)
  const followProgress = useRef(0)
  const keys = useRef<Record<string, boolean>>({})

  const scale = useMemo(() => (data ? roverDisplayScale(data) : 1), [data])

  // Route in sector metres. The route arrives on the display grid, so this is
  // the one place that conversion happens.
  const routeMetres = useMemo(() => {
    if (!data || !route || route.length === 0) return []
    return route.map((point) => ({
      x_m: (point.x / Math.max(1, data.grid.width - 1)) * data.width_m,
      y_m: (point.y / Math.max(1, data.grid.height - 1)) * data.height_m,
    }))
  }, [data, route])

  // -- keyboard ------------------------------------------------------------
  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      if (DRIVE_KEYS.has(key)) {
        keys.current[key] = true
        // Arrow keys scroll the console behind the canvas otherwise.
        event.preventDefault()
      }
    }
    const up = (event: KeyboardEvent) => {
      keys.current[event.key.toLowerCase()] = false
    }
    const blur = () => {
      keys.current = {}
    }

    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    // Without this a key held while the tab loses focus stays stuck down.
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
    }
  }, [])

  // -- placement -----------------------------------------------------------
  useEffect(() => {
    if (!data) return
    const target = startPoint ?? routeMetres[0] ?? {
      x_m: data.width_m / 2,
      y_m: data.height_m / 2,
    }
    pose.current = {
      x_m: target.x_m,
      y_m: target.y_m,
      heading_deg: startPoint?.heading_deg ?? pose.current.heading_deg ?? 0,
    }
    speed.current = 0
    distance.current = 0
    followProgress.current = 0
  }, [data, startPoint, resetSignal, routeMetres])

  useFrame((_, rawDelta) => {
    if (!groupRef.current || !data) return
    // A backgrounded tab resumes with a huge delta, which would teleport the
    // rover across the sector in one step.
    const delta = Math.min(rawDelta, 0.1)

    if (driveMode === 'follow') {
      advanceAlongRoute(delta)
    } else {
      drive(delta)
    }

    const { x_m, y_m, heading_deg } = pose.current
    const height = sampleHeight(data, x_m, y_m, exaggeration)
    const [wx, wy, wz] = sectorToWorld(data, x_m, y_m, height)

    // Sit ON the surface: the model's wheels bottom out at the group origin,
    // so no lift is added. The old constant offset is what left it hovering.
    groupRef.current.position.set(wx, wy, wz)

    const normal = sampleNormal(data, x_m, y_m, exaggeration)
    const target = surfaceQuaternion(normal, (heading_deg * Math.PI) / 180)
    // Ease into the new orientation so crossing a cell boundary does not snap.
    groupRef.current.quaternion.slerp(target, Math.min(1, delta * 12))

    if (poseRef) poseRef.current = { x_m, y_m, heading_deg }
    if (telemetryRef) {
      telemetryRef.current = {
        x_m,
        y_m,
        heading_deg: ((heading_deg % 360) + 360) % 360,
        speed_mps: speed.current,
        elevation_m: sampleLayer(data, data.elevation, x_m, y_m),
        slope_deg: sampleSlopeDeg(data, x_m, y_m),
        traversability: data.traversability
          ? sampleLayer(data, data.traversability, x_m, y_m)
          : null,
        distance_m: distance.current,
        mode: driveMode,
      }
    }

    if (cameraMode === 'Rover POV') chaseCamera(delta)
  })

  const { camera } = useThree()

  /** Free driving: the player steers, the terrain pushes back. */
  function drive(delta: number) {
    if (!data) return
    const held = keys.current
    const forward = held['w'] || held['arrowup']
    const backward = held['s'] || held['arrowdown']
    const left = held['a'] || held['arrowleft']
    const right = held['d'] || held['arrowright']
    const handbrake = held[' ']

    const slope = sampleSlopeDeg(data, pose.current.x_m, pose.current.y_m)
    // Steep ground is slow ground. This is what makes the terrain layers worth
    // looking at while driving rather than just decoration.
    const slopeFactor = Math.max(0.25, 1 - slope / IMPASSABLE_SLOPE_DEG)
    const topSpeed = MAX_SPEED_MPS * slopeFactor

    if (handbrake) {
      speed.current -= Math.sign(speed.current) * BRAKE_MPS2 * delta * 1.6
      if (Math.abs(speed.current) < 0.6) speed.current = 0
    } else if (forward) {
      speed.current = Math.min(topSpeed, speed.current + ACCELERATION_MPS2 * delta)
    } else if (backward) {
      speed.current = Math.max(-REVERSE_SPEED_MPS, speed.current - BRAKE_MPS2 * delta)
    } else {
      const drag = COAST_DRAG_MPS2 * delta
      speed.current =
        Math.abs(speed.current) <= drag ? 0 : speed.current - Math.sign(speed.current) * drag
    }

    if (Math.abs(speed.current) > STEERING_CUTOFF_MPS) {
      // Reversing steers the other way round, as a real vehicle does.
      const direction = Math.sign(speed.current)
      const rate = TURN_RATE_DEG * delta * direction
      if (left) pose.current.heading_deg -= rate
      if (right) pose.current.heading_deg += rate
    }

    const step = speed.current * delta
    const radians = (pose.current.heading_deg * Math.PI) / 180
    const nextX = pose.current.x_m + Math.cos(radians) * step
    const nextY = pose.current.y_m + Math.sin(radians) * step

    // Refuse a step onto ground too steep to climb, rather than letting the
    // rover walk up a wall.
    const nextSlope = sampleSlopeDeg(data, clampToSector(nextX, data.width_m), clampToSector(nextY, data.height_m))
    if (nextSlope > IMPASSABLE_SLOPE_DEG && speed.current > 0) {
      speed.current = 0
      return
    }

    pose.current.x_m = clampToSector(nextX, data.width_m)
    pose.current.y_m = clampToSector(nextY, data.height_m)
    distance.current += Math.abs(step)
  }

  /** Route following: the original behaviour, kept as a mode rather than the only one. */
  function advanceAlongRoute(delta: number) {
    if (routeMetres.length < 2) {
      speed.current = 0
      return
    }
    const last = routeMetres.length - 1

    if (isPlaying && followProgress.current < last) {
      followProgress.current = Math.min(followProgress.current + delta * 14, last)
    }

    const index = Math.floor(followProgress.current)
    const next = Math.min(index + 1, last)
    const t = followProgress.current - index
    const a = routeMetres[index]
    const b = routeMetres[next]

    const x = a.x_m + (b.x_m - a.x_m) * t
    const y = a.y_m + (b.y_m - a.y_m) * t
    const moved = Math.hypot(x - pose.current.x_m, y - pose.current.y_m)

    if (b !== a && (b.x_m !== a.x_m || b.y_m !== a.y_m)) {
      pose.current.heading_deg = (Math.atan2(b.y_m - a.y_m, b.x_m - a.x_m) * 180) / Math.PI
    }
    pose.current.x_m = x
    pose.current.y_m = y
    speed.current = delta > 0 ? moved / delta : 0
    distance.current += moved
  }

  function chaseCamera(delta: number) {
    if (!groupRef.current) return
    // Distances are relative to the rover's DISPLAY size, not to metres. The
    // model is drawn several times life size so it is visible on a sector
    // kilometres wide, and a camera parked a fixed number of metres behind it
    // ends up inside the chassis.
    const back = scale * (17 + Math.abs(speed.current) * 0.35)
    const offset = new THREE.Vector3(0, scale * 8.5, back).applyQuaternion(
      groupRef.current.quaternion
    )
    const desired = new THREE.Vector3().copy(groupRef.current.position).add(offset)

    camera.position.lerp(desired, Math.min(1, delta * 4))

    const ahead = new THREE.Vector3(0, scale * 2, -scale * 14).applyQuaternion(
      groupRef.current.quaternion
    )
    camera.lookAt(new THREE.Vector3().copy(groupRef.current.position).add(ahead))
  }

  // The rover is drawn whether or not a route exists: it is a vehicle to drive,
  // not an animation attached to a plan.
  if (!data) return null

  return (
    <group ref={groupRef}>
      <group scale={scale}>
        {/* Chassis */}
        <mesh position={[0, 1.5, 0]} castShadow>
          <boxGeometry args={[4, 1, 6]} />
          <meshStandardMaterial color="silver" metalness={0.8} roughness={0.35} />
        </mesh>

        {/* Solar panel */}
        <mesh position={[0, 2.1, 0]}>
          <boxGeometry args={[3.8, 0.1, 5.8]} />
          <meshStandardMaterial color="#1a2b4c" roughness={0.2} metalness={0.9} />
        </mesh>

        {/* Mast */}
        <mesh position={[0, 3, 2]}>
          <cylinderGeometry args={[0.2, 0.2, 3]} />
          <meshStandardMaterial color="white" />
        </mesh>

        {/* Camera head, pointing along travel (-Z). */}
        <mesh position={[0, 4.5, 2]}>
          <boxGeometry args={[1, 0.5, 0.5]} />
          <meshStandardMaterial color="gold" metalness={0.6} />
        </mesh>

        {/* Wheels. Radius 1 centred at y=1, so they bottom out exactly on the
            group origin - which is what lets the rover sit on the surface. */}
        {([[-2.2, 1, 2], [2.2, 1, 2], [-2.2, 1, -2], [2.2, 1, -2]] as const).map((position, i) => (
          <mesh key={i} position={position} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[1, 1, 0.5, 16]} />
            <meshStandardMaterial color="#222" roughness={0.9} />
          </mesh>
        ))}
      </group>
    </group>
  )
}

const DRIVE_KEYS = new Set([
  'w', 'a', 's', 'd',
  'arrowup', 'arrowdown', 'arrowleft', 'arrowright',
  ' ',
])

const clampToSector = (value: number, extent: number) =>
  Math.min(extent, Math.max(0, value))
