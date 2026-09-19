import type { PerceptionResult, ReferenceMap } from '../types'

interface Props {
  perception: PerceptionResult
  referenceMap: ReferenceMap | null
}

/**
 * Observed crater configuration against the reference map.
 *
 * Both panels are drawn in sector metres. The observation panel places each
 * candidate at estimated_position + its offset, which is exactly what the
 * matcher solved for, so a correct match shows as a candidate sitting on its
 * reference landmark rather than as a decorative connector.
 */
export const MatchView: React.FC<Props> = ({ perception, referenceMap }) => {
  const { candidates, localization, camera, pose } = perception
  const matches = localization.matches
  const matchedRefIds = new Set(matches.map((m) => m.reference_id))
  const matchedCandidateIds = new Set(matches.map((m) => m.candidate_id))

  if (!referenceMap) {
    return <div className="panel-empty">Reference map not loaded.</div>
  }

  const { width_m: W, height_m: H } = referenceMap

  // Anchor the observation panel on the estimate when there is one, otherwise
  // on the pose used to observe, so candidates are always placeable.
  const anchorX = localization.estimated_x_m ?? pose.x_m
  const anchorY = localization.estimated_y_m ?? pose.y_m
  const range = camera.max_range_m

  return (
    <div className="match-view">
      <div className="match-pair">
        <div className="match-side">
          <div className="match-side-title">Observation</div>
          <svg viewBox={`0 0 ${range * 2} ${range * 2}`} className="match-svg">
            <rect x={0} y={0} width={range * 2} height={range * 2} className="match-bg" />
            <circle cx={range} cy={range} r={range - 1} className="match-fov" />
            {candidates.map((c) => (
              <circle
                key={c.id}
                cx={range + c.offset_dx_m}
                cy={range + c.offset_dy_m}
                r={Math.max(4, c.diameter_m / 2)}
                className={
                  matchedCandidateIds.has(c.id) ? 'constellation matched' : 'constellation'
                }
              />
            ))}
            <circle cx={range} cy={range} r={6} className="rover-dot" />
          </svg>
          <div className="match-side-note">
            {candidates.length} candidates, rover at centre
          </div>
        </div>

        <div className="match-side">
          <div className="match-side-title">Reference map</div>
          <svg viewBox={`0 0 ${W} ${H}`} className="match-svg">
            <rect x={0} y={0} width={W} height={H} className="match-bg" />
            {referenceMap.craters.map((c) => (
              <circle
                key={c.id}
                cx={c.x_m}
                cy={c.y_m}
                r={Math.max(4, c.diameter_m / 2)}
                className={matchedRefIds.has(c.id) ? 'constellation matched' : 'constellation'}
              />
            ))}
            {/* The region the match resolved to. */}
            <circle cx={anchorX} cy={anchorY} r={range} className="match-region" />
            {localization.estimated_x_m !== null && (
              <circle
                cx={localization.estimated_x_m}
                cy={localization.estimated_y_m as number}
                r={8}
                className="estimate-dot"
              />
            )}
          </svg>
          <div className="match-side-note">
            {referenceMap.craters.length} landmarks, {matches.length} matched
          </div>
        </div>
      </div>

      <div className="match-result">
        <div className="match-result-title">Matching result</div>
        {matches.length === 0 ? (
          <div className="panel-empty">No consistent correspondences found.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Reference</th>
                <th>Residual</th>
              </tr>
            </thead>
            <tbody>
              {matches.map((m) => (
                <tr key={`${m.candidate_id}-${m.reference_id}`}>
                  <td>{m.candidate_id}</td>
                  <td>{m.reference_id}</td>
                  <td>{m.residual_m.toFixed(2)} m</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
