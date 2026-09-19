import type {
  DataCapabilities,
  InspectorView,
  PerceptionResult,
  ReferenceMap,
  RegistrationResult,
} from '../types'
import { RegistrationPanel } from './RegistrationPanel'
import { CraterMapPanel } from './CraterMapPanel'
import { LocalizationPanel } from './LocalizationPanel'
import { MatchView } from './MatchView'
import { ObservationView } from './ObservationView'

const VIEW_TITLE: Record<InspectorView, string> = {
  image: 'Rover image',
  quadrants: '4 quadrants',
  points: 'Crater points',
  craters: 'Crater map',
  reference: 'Reference map',
  match: 'Match',
  position: 'Estimated position',
  registration: 'Registration',
}

interface Props {
  view: InspectorView
  perception: PerceptionResult | null
  referenceMap: ReferenceMap | null
  loading: boolean
  error: string | null
  onRun: () => void
  capabilities: DataCapabilities | null
  registration: RegistrationResult | null
  registrationLoading: boolean
  registrationError: string | null
  onRunRegistration: (pair: 'source_reference' | 'self_check') => void
}

/** Right-hand inspector: one stage of the perception pipeline at a time. */
export const InspectorPanel: React.FC<Props> = ({
  view,
  perception,
  referenceMap,
  loading,
  error,
  onRun,
  capabilities,
  registration,
  registrationLoading,
  registrationError,
  onRunRegistration,
}) => {
  const matchedCandidateIds = new Set(
    perception?.localization.matches.map((m) => m.candidate_id) ?? []
  )

  const renderReferenceMap = () => {
    if (!referenceMap) return <div className="panel-empty">Reference map not loaded.</div>
    const { width_m: W, height_m: H } = referenceMap
    return (
      <div>
        <svg viewBox={`0 0 ${W} ${H}`} className="match-svg wide">
          <rect x={0} y={0} width={W} height={H} className="match-bg" />
          {referenceMap.craters.map((c) => (
            <circle
              key={c.id}
              cx={c.x_m}
              cy={c.y_m}
              r={Math.max(4, c.diameter_m / 2)}
              className={c.provenance === 'catalog' ? 'constellation catalog' : 'constellation'}
            />
          ))}
        </svg>
        <div className="panel-footnote">
          {referenceMap.craters.length} landmarks.{' '}
          {referenceMap.craters.filter((c) => c.provenance === 'catalog').length} from the sector
          catalog, the rest generated. {referenceMap.note}
        </div>
      </div>
    )
  }

  const renderBody = () => {
    if (view === 'registration') {
      return (
        <RegistrationPanel
          capabilities={capabilities}
          result={registration}
          loading={registrationLoading}
          error={registrationError}
          onRun={onRunRegistration}
        />
      )
    }

    if (view === 'reference') return renderReferenceMap()

    if (!perception) {
      return (
        <div className="panel-empty">
          No observation yet. Run the perception pipeline to populate this view.
        </div>
      )
    }

    switch (view) {
      case 'image':
      case 'quadrants':
      case 'points':
        return <ObservationView perception={perception} view={view} />
      case 'craters':
        return (
          <>
            <ObservationView perception={perception} view={view} />
            <CraterMapPanel
              candidates={perception.candidates}
              matchedIds={matchedCandidateIds}
            />
          </>
        )
      case 'match':
        return <MatchView perception={perception} referenceMap={referenceMap} />
      case 'position':
        return <LocalizationPanel localization={perception.localization} />
    }
  }

  return (
    <div className="inspector">
      <div className="inspector-header">
        <h2>{VIEW_TITLE[view]}</h2>
        {view !== 'registration' && (
          <button className="run-button" onClick={onRun} disabled={loading}>
            {loading ? 'Running...' : 'Run pipeline'}
          </button>
        )}
      </div>

      {error && <div className="error-message">{error}</div>}

      {perception && view !== 'registration' && (
        <ol className="stage-list">
          {perception.stages.map((s) => (
            <li key={s.key} className={`stage ${s.status}`}>
              <span className="stage-label">{s.label}</span>
              <span className="stage-detail">{s.detail}</span>
              <span className="stage-time">{s.duration_ms.toFixed(1)} ms</span>
            </li>
          ))}
        </ol>
      )}

      <div className="inspector-body">{renderBody()}</div>

      {perception && view !== 'registration' && (
        <div className="panel-footnote">
          Detector: {perception.detector.name}
          {perception.detector.is_synthetic && ' - validated only on synthetic imagery'}
        </div>
      )}
    </div>
  )
}
