import * as THREE from 'three'
import type { TerrainData } from '../types'

/**
 * One definition of where a point in the sector sits in the 3D scene.
 *
 * The terrain mesh lives inside a group rotated -PI/2 about X, which maps
 * local (x, y, z) to world (x, z, -y). Anything drawn outside that group -
 * the rover, its markers - has to apply that mapping itself, and the two
 * descriptions have to agree. They did not: the rover used the group's local
 * expression for the north-south axis while standing in world space, which
 * mirrored it in Z. It sampled its height at one place and stood at another,
 * so it hovered over unrelated ground.
 *
 * Everything here therefore works in the sector frame - metres from the
 * top-left corner, y increasing downward, the same convention the backend,
 * the camera model and the crater catalogue use - and converts once, here.
 */

/** Sector metres to the terrain group's LOCAL space (height in z). */
export const sectorToPlane = (
  data: TerrainData,
  x_m: number,
  y_m: number,
  height: number
): [number, number, number] => [x_m - data.width_m / 2, data.height_m / 2 - y_m, height]

/** Sector metres to WORLD space (height in y), for anything outside the group. */
export const sectorToWorld = (
  data: TerrainData,
  x_m: number,
  y_m: number,
  height: number
): [number, number, number] => [x_m - data.width_m / 2, height, y_m - data.height_m / 2]

/** World space back to sector metres. The inverse of sectorToWorld. */
export const worldToSector = (data: TerrainData, worldX: number, worldZ: number) => ({
  x_m: worldX + data.width_m / 2,
  y_m: worldZ + data.height_m / 2,
})

const clamp = (value: number, low: number, high: number) =>
  value < low ? low : value > high ? high : value

/**
 * Bilinear sample of a per-cell layer at a point in sector metres.
 *
 * Nearest-neighbour would step: the display grid is coarse - a hundred cells
 * across a sector kilometres wide - so snapping to the closest vertex puts a
 * rover metres above or below the surface actually being drawn, because the
 * mesh interpolates between those same vertices.
 */
export const sampleLayer = (
  data: TerrainData,
  layer: ArrayLike<number>,
  x_m: number,
  y_m: number
): number => {
  const { width, height } = data.grid
  const fx = clamp((x_m / data.width_m) * (width - 1), 0, width - 1)
  const fy = clamp((y_m / data.height_m) * (height - 1), 0, height - 1)

  const col = Math.min(Math.floor(fx), width - 2 < 0 ? 0 : width - 2)
  const row = Math.min(Math.floor(fy), height - 2 < 0 ? 0 : height - 2)
  const tx = fx - col
  const ty = fy - row

  const colNext = Math.min(col + 1, width - 1)
  const rowNext = Math.min(row + 1, height - 1)

  const v00 = layer[row * width + col]
  const v10 = layer[row * width + colNext]
  const v01 = layer[rowNext * width + col]
  const v11 = layer[rowNext * width + colNext]

  return (
    v00 * (1 - tx) * (1 - ty) +
    v10 * tx * (1 - ty) +
    v01 * (1 - tx) * ty +
    v11 * tx * ty
  )
}

/** Rendered surface height at a point, matching the mesh exactly. */
export const sampleHeight = (
  data: TerrainData,
  x_m: number,
  y_m: number,
  exaggeration: number
): number =>
  (sampleLayer(data, data.elevation, x_m, y_m) - data.min_elevation) * exaggeration

/**
 * Surface normal in WORLD space, from central differences on the rendered height.
 *
 * Taken over a cell rather than an epsilon: a smaller step just measures the
 * interpolation inside one cell and makes the rover twitch at every boundary.
 */
export const sampleNormal = (
  data: TerrainData,
  x_m: number,
  y_m: number,
  exaggeration: number,
  step?: number
): THREE.Vector3 => {
  const span = step ?? Math.max(data.width_m / data.grid.width, 1)

  const east = sampleHeight(data, x_m + span, y_m, exaggeration)
  const west = sampleHeight(data, x_m - span, y_m, exaggeration)
  const south = sampleHeight(data, x_m, y_m + span, exaggeration)
  const north = sampleHeight(data, x_m, y_m - span, exaggeration)

  // World x runs with sector x, world z runs with sector y.
  return new THREE.Vector3(-(east - west) / (2 * span), 1, -(south - north) / (2 * span))
    .normalize()
}

/** True slope of the rendered surface in degrees, before exaggeration. */
export const sampleSlopeDeg = (data: TerrainData, x_m: number, y_m: number): number => {
  const normal = sampleNormal(data, x_m, y_m, 1)
  return (Math.acos(clamp(normal.y, -1, 1)) * 180) / Math.PI
}

/**
 * Orientation that sits a vehicle flat on the surface.
 *
 * Three's objects look down -Z, so the basis puts the travel direction on -Z.
 * The heading is projected onto the surface tangent plane rather than applied
 * about world up, which is what makes the rover pitch into a slope instead of
 * staying level and cutting through it.
 */
export const surfaceQuaternion = (
  normal: THREE.Vector3,
  headingRad: number
): THREE.Quaternion => {
  const forward = new THREE.Vector3(Math.cos(headingRad), 0, Math.sin(headingRad))
  forward.addScaledVector(normal, -forward.dot(normal))

  if (forward.lengthSq() < 1e-8) {
    // Looking straight up or down a cliff: any tangent will do.
    forward.set(1, 0, 0).addScaledVector(normal, -normal.x)
  }
  forward.normalize()

  const right = new THREE.Vector3().crossVectors(forward, normal).normalize()
  const basis = new THREE.Matrix4().makeBasis(right, normal, forward.clone().negate())
  return new THREE.Quaternion().setFromRotationMatrix(basis)
}

/**
 * Surface gradient at a point: rise per metre, in the sector frame.
 *
 * `exaggeration` must be whatever the terrain is DRAWN with. It is tempting to
 * give the physics true slopes and leave exaggeration to the renderer, but
 * that makes the rover disagree with the world it is standing in: its body
 * tilts to the drawn surface while gravity pulls on a shallower one, so it
 * leans into a hill it then climbs as though flat.
 *
 * Pass 1 for the true gradient - the honest number to report - and the render
 * exaggeration for the surface actually being driven.
 */
export const sampleGradient = (
  data: TerrainData,
  x_m: number,
  y_m: number,
  exaggeration = 1,
  step?: number
): { dzdx: number; dzdy: number } => {
  const span = step ?? Math.max(data.width_m / data.grid.width, 1)
  const east = sampleHeight(data, x_m + span, y_m, exaggeration)
  const west = sampleHeight(data, x_m - span, y_m, exaggeration)
  const south = sampleHeight(data, x_m, y_m + span, exaggeration)
  const north = sampleHeight(data, x_m, y_m - span, exaggeration)
  return {
    dzdx: (east - west) / (2 * span),
    dzdy: (south - north) / (2 * span),
  }
}

/**
 * Grade along a heading, as rise over run.
 *
 * Positive is uphill. This is the number gravity acts on: the slope of the
 * ground taken in the direction of travel, which is what decides whether a
 * climb is possible, not the steepest slope at the point.
 */
export const gradeAlongHeading = (
  data: TerrainData,
  x_m: number,
  y_m: number,
  headingRad: number,
  exaggeration = 1
): number => {
  const { dzdx, dzdy } = sampleGradient(data, x_m, y_m, exaggeration)
  return dzdx * Math.cos(headingRad) + dzdy * Math.sin(headingRad)
}

/**
 * Attitude from where the four wheels actually touch down.
 *
 * Sampling one point under the middle of the rover makes it glide over a
 * crater rim as though the rim were not there. Sampling the contact patches
 * and fitting pitch and roll through them is what makes a depression feel like
 * one: the nose drops going in and lifts coming out.
 *
 * Heights here ARE exaggerated, because this drives what is drawn.
 */
export const sampleContactAttitude = (
  data: TerrainData,
  x_m: number,
  y_m: number,
  headingRad: number,
  halfLength_m: number,
  halfWidth_m: number,
  exaggeration: number
): { normal: THREE.Vector3; height: number } => {
  const forwardX = Math.cos(headingRad)
  const forwardY = Math.sin(headingRad)
  // Right of travel, in the sector frame (y increases downward).
  const rightX = -Math.sin(headingRad)
  const rightY = Math.cos(headingRad)

  const at = (alongSign: number, acrossSign: number) => {
    const x = x_m + forwardX * halfLength_m * alongSign + rightX * halfWidth_m * acrossSign
    const y = y_m + forwardY * halfLength_m * alongSign + rightY * halfWidth_m * acrossSign
    return sampleHeight(data, x, y, exaggeration)
  }

  const frontLeft = at(1, -1)
  const frontRight = at(1, 1)
  const rearLeft = at(-1, -1)
  const rearRight = at(-1, 1)

  // Slopes across the contact patch, in rendered units per metre.
  const alongRise = (frontLeft + frontRight - rearLeft - rearRight) / (4 * halfLength_m)
  const acrossRise = (frontRight + rearRight - frontLeft - rearLeft) / (4 * halfWidth_m)

  // Build the normal from the two tangents of the fitted plane.
  const along = new THREE.Vector3(forwardX, alongRise, forwardY).normalize()
  const across = new THREE.Vector3(rightX, acrossRise, rightY).normalize()
  const normal = new THREE.Vector3().crossVectors(across, along).normalize()
  if (normal.y < 0) normal.negate()

  return {
    normal,
    // The body rides on the mean of its contact patches, not on the point
    // under its centre, so it sits level across a rut instead of sinking in.
    height: (frontLeft + frontRight + rearLeft + rearRight) / 4,
  }
}

/** Display scale for the rover, so a 6 m vehicle is visible on a km-wide sector. */
export const roverDisplayScale = (data: TerrainData): number =>
  Math.max(1, data.width_m / 420)
