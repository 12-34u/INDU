import type { CraterCandidate } from '../types'

interface Props {
  candidates: CraterCandidate[]
  matchedIds: Set<string>
}

/**
 * Aggregated crater candidates. Every column is a measured property of the
 * circle fit; confidence is fit quality times angular coverage, not a
 * detection probability.
 */
export const CraterMapPanel: React.FC<Props> = ({ candidates, matchedIds }) => {
  if (candidates.length === 0) {
    return <div className="panel-empty">No crater candidates aggregated.</div>
  }

  return (
    <div className="crater-map">
      <table className="data-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Center (px)</th>
            <th>Diam.</th>
            <th>Conf.</th>
            <th>Pts</th>
            <th>Src</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c) => (
            <tr key={c.id} className={matchedIds.has(c.id) ? 'row-matched' : undefined}>
              <td>{c.id}</td>
              <td>
                {c.center_u.toFixed(0)}, {c.center_v.toFixed(0)}
              </td>
              <td>{c.diameter_m.toFixed(1)} m</td>
              <td>{c.confidence.toFixed(2)}</td>
              <td>{c.n_points}</td>
              <td>{c.source_quadrants.join('+')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="panel-footnote">
        Confidence = circle-fit quality x angular coverage. Synthetic imagery; not a
        detection probability.
      </div>
    </div>
  )
}
