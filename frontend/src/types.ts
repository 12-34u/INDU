/**
 * REAL_RAW drives the simulation from the mission bundles under data/raw:
 * the rover's view is cut from real Chandrayaan-2 OHRC imagery rather than
 * rendered, and the landmarks are craters detected in it.
 */
export type DataMode = 'DEMO' | 'REAL_LOCAL' | 'REAL_RAW'

export type TerrainLayer =
  | 'Elevation'
  | 'Slope'
  | 'Roughness'
  | 'Traversability'
  | 'Hazards'
  | 'Route'

export type CameraMode = 'Orbit' | 'Rover POV'

/**
 * How the rover moves. 'follow' replays the planned route, which is what the
 * rover used to do and nothing else; 'free' hands steering to the player.
 */
export type DriveMode = 'follow' | 'free'

/** A named place to start from, as a fraction of the sector. */
export interface StartSite {
  id: string
  label: string
  /** Fractions of the sector width and height, so one list fits any sector. */
  fx: number
  fy: number
  heading_deg: number
}

/** Live rover state for the HUD, sampled on a timer rather than per frame. */
export interface RoverTelemetry {
  x_m: number
  y_m: number
  heading_deg: number
  speed_mps: number
  elevation_m: number
  slope_deg: number
  /** Rise over run along the direction of travel. Positive is uphill. */
  grade_percent: number
  /** Gravity along the slope exceeds what the wheels can hold. */
  slipping: boolean
  /** Name of the active speed profile. */
  speed_profile: string
  /** False when the profile's speeds are deliberately faster than reality. */
  speed_is_physical: boolean
  traversability: number | null
  distance_m: number
  mode: DriveMode
}

/** Which stage of the pipeline the inspector panel is showing. */
export type InspectorView =
  | 'image'
  | 'quadrants'
  | 'points'
  | 'craters'
  | 'reference'
  | 'match'
  | 'position'
  | 'registration'

export type QuadrantName = 'Q1' | 'Q2' | 'Q3' | 'Q4'

export interface TerrainData {
  /** True when the craters below were detected in real imagery. */
  craters_are_detected?: boolean
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
  traversability?: number[]
  craters?: CraterData[]
  min_elevation: number
  max_elevation: number
  min_slope: number
  max_slope: number
  cost_stats?: CostStats
}

export interface CostStats {
  impassable_fraction: number
  mean_cost: number | null
  min_cost: number | null
  max_cost: number | null
  high_cost_fraction: number
  high_cost_threshold?: number
}

export interface CraterData {
  id?: string
  x: number
  y: number
  diameter_m: number
}

export interface RoutePoint {
  x: number
  y: number
}

export interface RouteMetrics {
  n_waypoints: number
  length_m: number
  max_slope_deg: number | null
  mean_slope_deg: number | null
  mean_cost: number | null
  high_cost_cells: number
}

/** A landmark in the reference map, in sector metres (y down). */
export interface ReferenceCrater {
  id: string
  x_m: number
  y_m: number
  diameter_m: number
  provenance: 'synthetic' | 'catalog' | 'detected'
}

export interface ReferenceMap {
  /** Present when the landmarks were detected in real imagery. */
  basemap?: SectorBasemapInfo
  sector_id: string
  width_m: number
  height_m: number
  is_synthetic: boolean
  note: string
  craters: ReferenceCrater[]
}

export interface QuadrantBounds {
  name: QuadrantName
  x: number
  y: number
  width: number
  height: number
}

export interface DetectedPoint {
  u: number
  v: number
  strength: number
  quadrant: QuadrantName
}

export interface CraterCandidate {
  id: string
  center_u: number
  center_v: number
  radius_px: number
  diameter_px: number
  n_points: number
  fit_rmse_px: number
  confidence: number
  source_quadrants: QuadrantName[]
  offset_dx_m: number
  offset_dy_m: number
  diameter_m: number
  provenance: string
}

export interface CraterMatch {
  candidate_id: string
  reference_id: string
  residual_m: number
}

export interface PipelineStage {
  key: string
  label: string
  status: 'ok' | 'failed' | 'skipped'
  detail: string
  duration_ms: number
}

export interface LocalizationResult {
  estimated_x_m: number | null
  estimated_y_m: number | null
  ground_truth_x_m: number | null
  ground_truth_y_m: number | null
  position_error_m: number | null
  heading_deg: number
  matches: CraterMatch[]
  n_candidates: number
  n_reference_in_range: number
  match_confidence: number | null
  method: string
  is_synthetic: boolean
  note: string
}

export interface VisibleTruth {
  reference_id: string
  u: number
  v: number
  radius_px: number
  diameter_m: number
  x_m: number
  y_m: number
}

/** Which elevation product the sector is using, and what else was available. */
export interface DemCandidateInfo {
  product_id: string
  path: string
  kind: 'polar' | 'equirectangular'
  posting_m: number
  /** null when the mode does not resolve a DEM by coverage at all. */
  covers_point?: boolean | null
}

export interface DemSelection {
  point: { longitude: number; latitude: number }
  selected: DemCandidateInfo | null
  available: DemCandidateInfo[]
  note: string
  sector_id?: string
  mode?: string
  elevation?: {
    product?: string
    is_synthetic?: boolean
    source_gsd_m?: number
    gsd_m?: number
    resample_factor?: number
    relief_m?: number
    notes?: string[]
  }
}

/** Provenance of the real imagery the simulation is standing on. */
export interface SectorBasemapInfo {
  product_id: string
  source_ref: string
  is_synthetic: boolean
  width_px: number
  height_px: number
  width_m: number
  height_m: number
  gsd_m: number
  browse_decimation: number
  origin_pixel: number
  origin_scan: number
  sun_direction_deg: number
  georeferenced: boolean
  centre_longitude: number | null
  centre_latitude: number | null
  anchor_error_m: number | null
  notes: string[]
  sector_id?: string
  n_reference_craters?: number
}

/** Selenographic coordinates, interpolated from the bundle's NAV grid. */
export interface GeodeticPosition {
  longitude: number
  latitude: number
}

export interface Geodetic {
  ground_truth?: GeodeticPosition
  estimated?: GeodeticPosition
  note?: string
}

/** One feature correspondence between the observation and the basemap. */
export interface FeatureCorrespondence {
  u: number
  v: number
  basemap_u: number
  basemap_v: number
}

export interface PerceptionResult {
  pose: { x_m: number; y_m: number; heading_deg: number }
  camera: {
    width_px: number
    height_px: number
    gsd_m: number
    footprint_m: [number, number]
    max_range_m: number
  }
  image_png_base64: string
  quadrants: QuadrantBounds[]
  points: DetectedPoint[]
  candidates: CraterCandidate[]
  localization: LocalizationResult
  visible_truth: VisibleTruth[]
  stages: PipelineStage[]
  detector: { name: string; is_synthetic: boolean }
  /** Present on every response; the fields below describe the real-data path. */
  sector?: { width_m: number; height_m: number }
  is_real_imagery?: boolean
  /** Empty in the demo, which has no basemap. */
  basemap?: SectorBasemapInfo
  correspondences?: FeatureCorrespondence[]
  geodetic?: Geodetic
}

/** Rover pose in sector metres, y down - the single source of truth for where the rover is. */
export interface RoverPose {
  x_m: number
  y_m: number
  heading_deg: number
}

// ---- Real-data capabilities and registration ----

export interface DataCapabilities {
  mode: DataMode
  sector: string
  source_image: boolean
  reference_image: boolean
  navigation: boolean
  dem: boolean
  spice: boolean
  registration: boolean
  source_ref: string | null
  reference_ref: string | null
  navigation_ref: string | null
  source_product: RawProduct | null
  reference_product: RawProduct | null
  messages: string[]
}

export interface RawProduct {
  product_id: string
  role: string
  path: string
  is_archive: boolean
  mission: string | null
  instrument: string | null
  size_bytes: number
  label: Record<string, string | number>
  notes: string[]
  members: { name: string; size_bytes: number; kind: string; cheap_to_read: boolean }[]
}

export interface RegistrationMetrics {
  n_candidate_matches: number
  n_uniform_selected: number
  n_verified_inliers: number
  inlier_ratio: number
  rmse_px: number | null
  median_error_px: number | null
  max_error_px: number | null
  spatial_coverage: number
  illumination_representation: string
  model_type: string
  quality_passed: boolean
  quality_reasons: string[]
  n_subpixel_refined: number
  source_decimation: number | null
  reference_decimation: number | null
  inlier_distribution?: DistributionMetrics
  candidate_distribution?: DistributionMetrics
}

export interface DistributionMetrics {
  grid_cols: number
  grid_rows: number
  occupied_cells: number
  total_cells: number
  occupied_fraction: number
  max_points_in_cell: number
  gini: number
}

export interface MatchPoint {
  index: number
  source_x: number
  source_y: number
  reference_x: number
  reference_y: number
  error_px: number | null
  inlier: boolean
  match_score: number
  refine_correlation: number
}

export interface RegistrationStageInfo {
  key: string
  label: string
  status: 'ok' | 'failed' | 'skipped'
  detail: string
  duration_ms: number
}

export interface DisplayDerivative {
  file: string
  width: number
  height: number
  scale_from_working: number
}

export interface RegistrationResult {
  scene_id: string
  generated_at: string
  succeeded: boolean
  source_ref: string
  reference_ref: string
  uses_dem: boolean
  uses_spice: boolean
  caveat: string
  metrics: RegistrationMetrics
  stages: RegistrationStageInfo[]
  notes: string[]
  display: Record<string, DisplayDerivative>
  n_matches?: number
  n_inliers?: number
  matches_preview?: MatchPoint[]
  known_transform_applied?: Record<string, number | number[]>
  known_transform_error?: { mean_px: number; median_px: number; max_px: number; rmse_px: number }
}
