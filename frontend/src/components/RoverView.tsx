import React, { useRef, useMemo, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import type { DriveMode, RoutePoint, RoverPose, RoverTelemetry, TerrainData } from '../types'
import { profileById, type SpeedProfile } from '../lib/roverProfiles'
import {
  roverDisplayScale,
  sampleContactAttitude,
  sampleGradient,
  sampleLayer,
  sampleSlopeDeg,
  sectorToWorld,
  surfaceQuaternion,
} from '../lib/terrain'

/**
 * Driving model, in real units: metres, seconds, and lunar gravity.
 *
 * The previous model capped top speed by slope and refused any step onto
 * ground steeper than a fixed 28 degrees. That made climbing and descending
 * feel identical - the only thing slope did was make you slower in both
 * directions - and the climb limit was a number rather than a consequence.
 *
 * Here the wheels produce a force, gravity pulls along the slope, and what
 * the rover can and cannot climb falls out of the two. Driving uphill costs
 * speed, driving downhill gains it, and a crater wall stops you because
 * traction runs out, not because a constant said so.
 */
const LUNAR_GRAVITY_MPS2 = 1.62

// Coefficient of friction of a wheel on regolith. It sets the climb limit:
// the steepest grade the rover can hold is atan(TRACTION) ~ 26.6 degrees,
// which is now where the limit comes from rather than a declared constant.
const TRACTION = 0.5
// Rolling resistance on loose regolith.
const ROLLING_RESISTANCE = 0.08
// Top speed and the acceleration scale both come from the selected speed
// profile; see lib/roverProfiles.ts for why there is a deliberate non-physical
// option and what it does and does not change.
// Fraction of available traction the motors can actually demand.
const THROTTLE_AUTHORITY = 0.85
const BRAKE_AUTHORITY = 1.0
const TURN_RATE_DEG = 42
// Below this the wheels have nothing to bite on, so steering stops responding.
const STEERING_CUTOFF_MPS = 0.05
// Below this, with no throttle, static friction holds the rover rather than
// letting rolling resistance oscillate it around zero.
const STATIC_HOLD_MPS = 0.02

// Contact patch of the model: wheels sit at +/-2 along travel and +/-2.2
// across it, before the display scale is applied.
const WHEELBASE_HALF = 2.0
const TRACK_HALF = 2.2
// How quickly the body settles onto a new attitude. Lower is softer.
const SUSPENSION_RATE = 6

interface RoverViewProps {
  data: TerrainData | null
  route?: RoutePoint[]
  exaggeration?: number
  cameraMode?: 'Orbit' | 'Rover POV'
  driveMode?: DriveMode
  isPlaying?: boolean
  /**
   * Simulation speed-up. The physics stays in real units - lunar gravity, real
   * grades, a top speed anchored to the Apollo rover - and this only advances
   * the clock faster. A real robotic rover does a few centimetres a second, so
   * without this the sector takes hours to cross.
   */
  timeScale?: number
  /** Which speed profile to drive under. See lib/roverProfiles.ts. */
  speedProfileId?: string
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
  timeScale = 1,
  speedProfileId = 'fast',
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
  const followDistance = useRef(0)
  const keys = useRef<Record<string, boolean>>({})
  const traction = useRef({ grade: 0, grip: 0, gravityAlong: 0, slipping: false })

  const scale = useMemo(() => (data ? roverDisplayScale(data) : 1), [data])
  const profile: SpeedProfile = useMemo(() => profileById(speedProfileId), [speedProfileId])

  // Route in sector metres. The route arrives on the display grid, so this is
  // the one place that conversion happens.
  const routeMetres = useMemo(() => {
    if (!data || !route || route.length === 0) return []
    return route.map((point) => ({
      x_m: (point.x / Math.max(1, data.grid.width - 1)) * data.width_m,
      y_m: (point.y / Math.max(1, data.grid.height - 1)) * data.height_m,
    }))
  }, [data, route])

  /**
   * Cumulative distance along the route, in metres.
   *
   * Following by route INDEX was the bug behind a rover that ignored its own
   * speed limit: advancing a fixed number of waypoints per second means the
   * speed is whatever the waypoint spacing happens to be, which on this grid
   * worked out at about 56 m/s regardless of the profile. Arc length makes the
   * speed a speed.
   */
  const routeArcLength = useMemo(() => {
    const lengths = [0]
    for (let i = 1; i < routeMetres.length; i++) {
      const a = routeMetres[i - 1]
      const b = routeMetres[i]
      lengths.push(lengths[i - 1] + Math.hypot(b.x_m - a.x_m, b.y_m - a.y_m))
    }
    return lengths
  }, [routeMetres])

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
    followDistance.current = 0
  }, [data, startPoint, resetSignal, routeMetres])

  useFrame((_, rawDelta) => {
    if (!groupRef.current || !data) return
    // A backgrounded tab resumes with a huge delta, which would teleport the
    // rover across the sector in one step. Clamp first, then scale, so a long
    // stall cannot be multiplied into a bigger one.
    const delta = Math.min(rawDelta, 0.1) * Math.max(0.1, timeScale)

    if (driveMode === 'follow') {
      advanceAlongRoute(delta)
    } else {
      drive(delta)
    }

    const { x_m, y_m, heading_deg } = pose.current
    const headingRad = (heading_deg * Math.PI) / 180

    // Ride on the four contact patches rather than on one point under the
    // centre. This is what makes a crater feel like a crater: the nose drops
    // going in and lifts coming out, and a rim tips the body as it crosses.
    const { normal, height } = sampleContactAttitude(
      data,
      x_m,
      y_m,
      headingRad,
      WHEELBASE_HALF * scale,
      TRACK_HALF * scale,
      exaggeration
    )
    const [wx, wy, wz] = sectorToWorld(data, x_m, y_m, height)

    // Sit ON the surface: the model's wheels bottom out at the group origin,
    // so no lift is added. The old constant offset is what left it hovering.
    groupRef.current.position.set(wx, wy, wz)

    const target = surfaceQuaternion(normal, headingRad)
    // Suspension: the body follows the ground with lag rather than snapping to
    // it, so a bump reads as a bump instead of a jump.
    groupRef.current.quaternion.slerp(target, Math.min(1, delta * SUSPENSION_RATE))

    if (poseRef) poseRef.current = { x_m, y_m, heading_deg }
    if (telemetryRef) {
      telemetryRef.current = {
        x_m,
        y_m,
        heading_deg: ((heading_deg % 360) + 360) % 360,
        speed_mps: speed.current,
        elevation_m: sampleLayer(data, data.elevation, x_m, y_m),
        slope_deg: sampleSlopeDeg(data, x_m, y_m),
        grade_percent: traction.current.grade * 100,
        slipping: traction.current.slipping,
        speed_profile: profile.label,
        speed_is_physical: profile.physical,
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

  /**
   * Free driving: the player asks for force, the slope decides what happens.
   *
   * Accelerations are summed rather than speeds being clamped, so the rover
   * has to fight gravity up a crater wall and has to brake coming down one.
   */
  function drive(delta: number) {
    if (!data) return
    const held = keys.current
    const forward = held['w'] || held['arrowup']
    const backward = held['s'] || held['arrowdown']
    const left = held['a'] || held['arrowleft']
    const right = held['d'] || held['arrowright']
    const handbrake = held[' ']

    const headingRad = (pose.current.heading_deg * Math.PI) / 180
    // Same surface the body is drawn sitting on, so what you see is what you
    // drive. sampleSlopeDeg below reports the TRUE slope for the readout.
    const { dzdx, dzdy } = sampleGradient(
      data, pose.current.x_m, pose.current.y_m, exaggeration
    )

    // Grade along travel, and the steepest grade at this point. The first is
    // what gravity acts through; the second sets how much weight is on the
    // wheels and therefore how much grip there is.
    const grade = dzdx * Math.cos(headingRad) + dzdy * Math.sin(headingRad)
    const steepest = Math.hypot(dzdx, dzdy)
    const sinAlong = grade / Math.sqrt(1 + grade * grade)
    const cosSlope = 1 / Math.sqrt(1 + steepest * steepest)

    // Grip available at this attitude. Everything the wheels do is bounded by
    // it. The profile factor multiplies grip and gravity together, so their
    // ratio - and therefore how much a slope costs - is identical in both
    // profiles, and so is the slip threshold.
    const gravity = LUNAR_GRAVITY_MPS2 * profile.factor
    const grip = TRACTION * gravity * cosSlope
    const gravityAlong = -gravity * sinAlong

    let drive = 0
    if (handbrake) {
      drive = -Math.sign(speed.current) * grip * BRAKE_AUTHORITY
    } else if (forward) {
      drive = grip * THROTTLE_AUTHORITY
    } else if (backward) {
      drive = -grip * THROTTLE_AUTHORITY
    }

    const resistance =
      Math.abs(speed.current) > STATIC_HOLD_MPS
        ? -Math.sign(speed.current) * ROLLING_RESISTANCE * gravity * cosSlope
        : 0

    let acceleration = drive + gravityAlong + resistance

    // Parked on a slope: static friction holds it until gravity exceeds grip.
    const parked = Math.abs(speed.current) <= STATIC_HOLD_MPS && !forward && !backward
    if (parked) {
      if (Math.abs(gravityAlong) <= grip) {
        speed.current = 0
        acceleration = 0
      }
    }

    speed.current += acceleration * delta

    // Motor limit, not a force: the rover simply cannot spin its wheels faster.
    // Gravity can still carry it past this downhill, which is why the cap is
    // only applied to powered travel.
    if (!handbrake) {
      if (speed.current > profile.maxForward && drive > 0) speed.current = profile.maxForward
      if (speed.current < -profile.maxReverse && drive < 0) speed.current = -profile.maxReverse
    }
    if (handbrake && Math.abs(speed.current) < STATIC_HOLD_MPS) speed.current = 0

    traction.current = {
      grade,
      grip,
      gravityAlong,
      // Slipping when gravity along the slope exceeds what the wheels can hold.
      slipping: Math.abs(gravityAlong) > grip,
    }

    if (Math.abs(speed.current) > STEERING_CUTOFF_MPS) {
      // Reversing steers the other way round, as a real vehicle does.
      const direction = Math.sign(speed.current)
      // Turning is limited by grip too, so a steep slope also costs agility.
      const authority = Math.min(1, grip / (TRACTION * gravity))
      const rate = TURN_RATE_DEG * delta * direction * authority
      if (left) pose.current.heading_deg -= rate
      if (right) pose.current.heading_deg += rate
    }

    const step = speed.current * delta
    pose.current.x_m = clampToSector(pose.current.x_m + Math.cos(headingRad) * step, data.width_m)
    pose.current.y_m = clampToSector(pose.current.y_m + Math.sin(headingRad) * step, data.height_m)
    distance.current += Math.abs(step)
  }

  /**
   * Route following, driven at the same speed the player would manage.
   *
   * The rover walks the planned path at its profile's top speed, accelerating
   * and braking with the profile's authority and losing speed on climbs the
   * same way free driving does. It is the same vehicle; only the steering is
   * taken over.
   */
  function advanceAlongRoute(delta: number) {
    if (routeMetres.length < 2) {
      speed.current = 0
      return
    }
    const total = routeArcLength[routeArcLength.length - 1]
    const finished = followDistance.current >= total

    if (isPlaying && !finished) {
      const headingRad = (pose.current.heading_deg * Math.PI) / 180
      const { dzdx, dzdy } = sampleGradient(
        data!, pose.current.x_m, pose.current.y_m, exaggeration
      )
      const grade = dzdx * Math.cos(headingRad) + dzdy * Math.sin(headingRad)
      const sinAlong = grade / Math.sqrt(1 + grade * grade)
      const steepest = Math.hypot(dzdx, dzdy)
      const cosSlope = 1 / Math.sqrt(1 + steepest * steepest)

      const gravity = LUNAR_GRAVITY_MPS2 * profile.factor
      const grip = TRACTION * gravity * cosSlope
      const acceleration =
        grip * THROTTLE_AUTHORITY
        - gravity * sinAlong
        - ROLLING_RESISTANCE * gravity * cosSlope

      speed.current = Math.max(
        0, Math.min(profile.maxForward, speed.current + acceleration * delta)
      )
      // Ease to a stop at the goal rather than halting mid-stride.
      const remaining = total - followDistance.current
      const stopping = (speed.current * speed.current) / (2 * Math.max(grip, 1e-6))
      if (remaining <= stopping) {
        speed.current = Math.max(0, speed.current - grip * delta)
      }

      followDistance.current = Math.min(total, followDistance.current + speed.current * delta)
      distance.current += speed.current * delta

      traction.current = { grade, grip, gravityAlong: -gravity * sinAlong, slipping: false }
    } else {
      speed.current = 0
    }

    // Position by arc length: find the segment this distance falls in.
    const target = followDistance.current
    let index = 1
    while (index < routeArcLength.length - 1 && routeArcLength[index] < target) index++
    const segmentStart = routeArcLength[index - 1]
    const segmentLength = Math.max(routeArcLength[index] - segmentStart, 1e-6)
    const t = Math.min(1, Math.max(0, (target - segmentStart) / segmentLength))

    const a = routeMetres[index - 1]
    const b = routeMetres[index]
    pose.current.x_m = a.x_m + (b.x_m - a.x_m) * t
    pose.current.y_m = a.y_m + (b.y_m - a.y_m) * t

    if (b.x_m !== a.x_m || b.y_m !== a.y_m) {
      pose.current.heading_deg = (Math.atan2(b.y_m - a.y_m, b.x_m - a.x_m) * 180) / Math.PI
    }
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
