import type { LocalizationResult } from '../types'

interface Props {
  localization: LocalizationResult
}

const fmt = (v: number | null, unit = ' m') => (v === null ? '--' : `${v.toFixed(2)}${unit}`)

/**
 * Ground truth against the recovered position.
 *
 * In synthetic mode the ground truth is the pose the observation was rendered
 * from, so the error is a real measurement of this pipeline rather than a
 * quoted figure. Outside synthetic mode there is no truth to compare against
 * and those rows read "--".
 */
export const LocalizationPanel: React.FC<Props> = ({ localization: loc }) => {
  const failed = loc.estimated_x_m === null

  return (
    <div className="localization-panel">
      <div className="position-grid">
        <div className="position-block">
          <div className="position-title">Ground truth</div>
          <div className="position-row">
            <span>X</span>
            <span>{fmt(loc.ground_truth_x_m)}</span>
          </div>
          <div className="position-row">
            <span>Y</span>
            <span>{fmt(loc.ground_truth_y_m)}</span>
          </div>
          <div className="position-note">
            {loc.is_synthetic ? 'Synthetic mode' : 'Unavailable outside synthetic mode'}
          </div>
        </div>

        <div className="position-block">
          <div className="position-title">Estimated</div>
          <div className="position-row">
            <span>X</span>
            <span>{fmt(loc.estimated_x_m)}</span>
          </div>
          <div className="position-row">
            <span>Y</span>
            <span>{fmt(loc.estimated_y_m)}</span>
          </div>
          <div className="position-note">{loc.method}</div>
        </div>
      </div>

      <div className={`error-block ${failed ? 'failed' : ''}`}>
        <div className="metric-label">Position error</div>
        <div className="error-value">{fmt(loc.position_error_m)}</div>
      </div>

      <div className="metric-card">
        <div className="metric-label">Synthetic match confidence</div>
        <div className="metric-value">
          {loc.match_confidence === null ? '--' : loc.match_confidence.toFixed(3)}
        </div>
        <div className="panel-footnote">
          Fraction of crater candidates that agreed on this position
          ({loc.matches.length}/{loc.n_candidates}).
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-label">Reference landmarks in camera range</div>
        <div className="metric-value">{loc.n_reference_in_range}</div>
      </div>

      <div className="panel-note">{loc.note}</div>
      <div className="panel-footnote">
        Heading is taken as known in this MVP; only position is solved.
      </div>
    </div>
  )
}
