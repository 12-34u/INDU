import type { DataCapabilities, RegistrationResult } from '../types'

const API_BASE = 'http://localhost:8000'

interface Props {
  capabilities: DataCapabilities | null
  result: RegistrationResult | null
  loading: boolean
  error: string | null
  onRun: (pair: 'source_reference' | 'self_check') => void
}

const Flag: React.FC<{ label: string; value: boolean; detail?: string }> = ({
  label,
  value,
  detail,
}) => (
  <div className="capability-row">
    <span className={`capability-dot ${value ? 'on' : 'off'}`} />
    <span className="capability-label">{label}</span>
    <span className="capability-value">{detail ?? (value ? 'available' : 'unavailable')}</span>
  </div>
)

/**
 * Real-data status and registration results.
 *
 * Candidate matches and verified inliers are shown as separate numbers
 * throughout: a large candidate count with few inliers is a failed
 * registration, and collapsing them into one figure would hide that.
 */
export const RegistrationPanel: React.FC<Props> = ({
  capabilities,
  result,
  loading,
  error,
  onRun,
}) => {
  if (!capabilities) {
    return <div className="panel-empty">Capabilities not loaded.</div>
  }

  const source = capabilities.source_product
  const reference = capabilities.reference_product
  const metrics = result?.metrics

  const registrationState = loading
    ? 'running'
    : result
      ? result.succeeded
        ? 'completed'
        : 'completed (quality gate failed)'
      : capabilities.registration
        ? 'ready'
        : 'blocked'

  return (
    <div className="registration-panel">
      <div className="capability-block">
        <div className="position-title">Data status</div>
        <Flag label="Data source" value detail={capabilities.mode} />
        <Flag
          label="Source"
          value={capabilities.source_image}
          detail={source ? `${source.instrument ?? 'unknown'} · ${source.product_id.slice(0, 22)}…` : 'unavailable'}
        />
        <Flag
          label="Reference"
          value={capabilities.reference_image}
          detail={reference ? `${reference.instrument ?? 'unknown'} · ${reference.product_id}` : 'unavailable'}
        />
        <Flag label="NAV" value={capabilities.navigation} />
        <Flag label="DEM" value={capabilities.dem} />
        <Flag label="SPICE" value={capabilities.spice} />
        <Flag label="Registration" value={capabilities.registration} detail={registrationState} />
      </div>

      {capabilities.messages.map((m) => (
        <div key={m} className="panel-note">{m}</div>
      ))}

      <div className="button-group row registration-actions">
        <button onClick={() => onRun('source_reference')} disabled={loading || !capabilities.registration}>
          Run source vs reference
        </button>
        <button onClick={() => onRun('self_check')} disabled={loading || !capabilities.source_image}>
          Run controlled self-check
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}
      {loading && <div className="panel-empty">Registration running…</div>}

      {result && metrics && (
        <>
          <ol className="stage-list">
            {result.stages.map((s) => (
              <li key={s.key} className={`stage ${s.status}`}>
                <span className="stage-label">{s.label}</span>
                <span className="stage-detail">{s.detail}</span>
                <span className="stage-time">{s.duration_ms.toFixed(0)} ms</span>
              </li>
            ))}
          </ol>

          <div className={`error-block ${metrics.quality_passed ? '' : 'failed'}`}>
            <div className="metric-label">Quality gate</div>
            <div className="error-value">{metrics.quality_passed ? 'PASSED' : 'FAILED'}</div>
          </div>

          {metrics.quality_reasons.length > 0 && (
            <ul className="reason-list">
              {metrics.quality_reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}

          <table className="data-table">
            <tbody>
              <tr><td>Candidate matches</td><td>{metrics.n_candidate_matches}</td></tr>
              <tr><td>After uniform selection</td><td>{metrics.n_uniform_selected}</td></tr>
              <tr className="row-matched"><td>Verified inliers</td><td>{metrics.n_verified_inliers}</td></tr>
              <tr><td>Inlier ratio</td><td>{metrics.inlier_ratio.toFixed(3)}</td></tr>
              <tr><td>RMSE</td><td>{metrics.rmse_px === null ? '--' : `${metrics.rmse_px.toFixed(3)} px`}</td></tr>
              <tr><td>Median / max error</td><td>
                {metrics.median_error_px === null ? '--' : `${metrics.median_error_px.toFixed(2)} / ${metrics.max_error_px?.toFixed(2)} px`}
              </td></tr>
              <tr><td>Spatial coverage</td><td>{metrics.spatial_coverage.toFixed(3)}</td></tr>
              <tr><td>Sub-pixel refined</td><td>{metrics.n_subpixel_refined}</td></tr>
              <tr><td>Illumination</td><td>{metrics.illumination_representation}</td></tr>
              <tr><td>Model</td><td>{metrics.model_type}</td></tr>
            </tbody>
          </table>

          {result.known_transform_error && (
            <div className="metric-card">
              <div className="metric-label">Recovered vs known transform</div>
              <div className="metric-value">
                {result.known_transform_error.mean_px.toFixed(3)} px mean
              </div>
              <div className="panel-footnote">
                max {result.known_transform_error.max_px.toFixed(3)} px. Controlled check:
                the applied transform is known exactly, so this is a measured accuracy.
              </div>
            </div>
          )}

          <div className="registration-images">
            {(['source', 'reference', 'registered'] as const).map((key) =>
              result.display[key] ? (
                <figure key={key}>
                  <img
                    src={`${API_BASE}/registration/image/${result.scene_id}/${result.display[key].file}`}
                    alt={key}
                  />
                  <figcaption>{key}</figcaption>
                </figure>
              ) : null
            )}
          </div>

          <div className="panel-note">{result.caveat}</div>
          {result.notes.map((n) => (
            <div key={n} className="panel-footnote">{n}</div>
          ))}
        </>
      )}
    </div>
  )
}
