export type DataMode = 'DEMO' | 'REAL_LOCAL'

export type TerrainLayer = 'Elevation' | 'Slope' | 'Roughness' | 'Hazards' | 'Route'

export type CameraMode = 'Orbit' | 'Rover POV'

export interface TerrainData {
  sector_id: string
  width_m: number
  height_m: number
  grid: {
    width: number
    height: number
  }
  elevation: number[]
  slope: number[]
  roughness: number[]
  hazards: number[]
  craters?: CraterData[]
  min_elevation: number
  max_elevation: number
  min_slope: number
  max_slope: number
}

export interface CraterData {
  x: number
  y: number
  radius: number
}

export interface RoutePoint {
  x: number
  y: number
}
