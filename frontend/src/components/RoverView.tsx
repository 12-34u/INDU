import React, { useRef, useMemo, useEffect, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import type { RoutePoint, TerrainData } from '../types'

interface RoverViewProps {
  data: TerrainData | null
  route?: RoutePoint[]
  exaggeration?: number
  cameraMode?: 'Orbit' | 'Rover POV'
  isPlaying?: boolean
  onReset?: boolean // Signal to reset position
}

export const RoverView: React.FC<RoverViewProps> = ({ 
  data, 
  route, 
  exaggeration = 2,
  cameraMode = 'Orbit',
  isPlaying = false,
  onReset = false
}) => {
  const groupRef = useRef<THREE.Group>(null)
  const { camera } = useThree()
  
  // State for animation
  const [progress, setProgress] = useState(0) // 0 to route.length - 1

  // Pre-calculate 3D route points once
  const pathPoints = useMemo(() => {
    if (!data || !route || route.length === 0) return []
    const pts: THREE.Vector3[] = []
    
    for (const pt of route) {
      const nx = pt.x / (data.grid.width - 1)
      const ny = pt.y / (data.grid.height - 1)
      
      const localX = (nx - 0.5) * data.width_m
      // Note: We are placing the rover directly in world space!
      // In the Terrain3D component, we drew the line inside a Group rotated -PI/2 on X.
      // So local X = world X.
      // local Y (in plane space) = world -Z.
      // height = world Y.
      const localY = -(ny - 0.5) * data.height_m
      
      const idx = pt.y * data.grid.width + pt.x
      const h = (data.elevation[idx] - data.min_elevation) * exaggeration
      
      pts.push(new THREE.Vector3(localX, h, localY))
    }
    return pts
  }, [data, route, exaggeration])

  // Reset animation if needed or if route changes
  useEffect(() => {
    setProgress(0)
  }, [route, onReset])

  useFrame((_, delta) => {
    if (!groupRef.current || pathPoints.length < 2) return

    // Move rover
    if (isPlaying && progress < pathPoints.length - 1) {
      const speed = 5.0 // points per second (arbitrary speed)
      setProgress(p => Math.min(p + delta * speed, pathPoints.length - 1))
    }

    const currentIdx = Math.floor(progress)
    const nextIdx = Math.min(currentIdx + 1, pathPoints.length - 1)
    const t = progress - currentIdx

    const p1 = pathPoints[currentIdx]
    const p2 = pathPoints[nextIdx]

    // Interpolate position
    const currentPos = new THREE.Vector3().lerpVectors(p1, p2, t)
    
    // Offset rover body slightly above ground
    currentPos.y += 2.0 
    
    groupRef.current.position.copy(currentPos)

    // Calculate heading (look at next point)
    if (currentIdx < pathPoints.length - 1) {
      // Look slightly ahead
      const lookTarget = p2.clone()
      lookTarget.y += 2.0
      groupRef.current.lookAt(lookTarget)
    }

    // Camera logic
    if (cameraMode === 'Rover POV') {
      // Position camera slightly behind and above rover
      const offset = new THREE.Vector3(0, 4, 10)
      
      // We want the camera to be relative to the rover's rotation
      offset.applyQuaternion(groupRef.current.quaternion)
      
      camera.position.copy(currentPos).add(offset)
      
      // Look at where the rover is going
      const camTarget = new THREE.Vector3().copy(currentPos)
      const forward = new THREE.Vector3(0, 0, -20).applyQuaternion(groupRef.current.quaternion)
      camTarget.add(forward)
      
      camera.lookAt(camTarget)
    }
  })

  if (!data || !route || route.length === 0) return null

  return (
    <group ref={groupRef}>
      {/* Procedural Rover Mesh */}
      {/* Chassis */}
      <mesh position={[0, 1.5, 0]}>
        <boxGeometry args={[4, 1, 6]} />
        <meshStandardMaterial color="silver" metalness={0.8} />
      </mesh>
      
      {/* Solar Panel */}
      <mesh position={[0, 2.1, 0]}>
        <boxGeometry args={[3.8, 0.1, 5.8]} />
        <meshStandardMaterial color="#1a2b4c" roughness={0.2} metalness={0.9} />
      </mesh>

      {/* Mast */}
      <mesh position={[0, 3, 2]}>
        <cylinderGeometry args={[0.2, 0.2, 3]} />
        <meshStandardMaterial color="white" />
      </mesh>
      
      {/* Camera Head */}
      <mesh position={[0, 4.5, 2]}>
        <boxGeometry args={[1, 0.5, 0.5]} />
        <meshStandardMaterial color="gold" />
      </mesh>

      {/* Wheels (4) */}
      {[[-2.2, 1, 2], [2.2, 1, 2], [-2.2, 1, -2], [2.2, 1, -2]].map((pos, i) => (
        <mesh key={i} position={pos as [number, number, number]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[1, 1, 0.5, 16]} />
          <meshStandardMaterial color="#222" roughness={0.9} />
        </mesh>
      ))}
    </group>
  )
}
