import type { InspectorView, PerceptionResult, QuadrantName } from '../types'

const QUADRANT_COLOR: Record<QuadrantName, string> = {
  Q1: '#66fcf1',
  Q2: '#ffb347',
  Q3: '#c084fc',
  Q4: '#7ee787',
}

interface Props {
  perception: PerceptionResult
  view: InspectorView
}

/**
 * The rover observation with pipeline overlays drawn in image pixel space.
 * The SVG shares the image's coordinate system via viewBox, so overlay
 * positions are the raw detector outputs with no rescaling in the UI.
 */
export const ObservationView: React.FC<Props> = ({ perception, view }) => {
  const { camera, image_png_base64, quadrants, points, candidates } = perception
  const { width_px: w, height_px: h } = camera

  const showQuadrants = view === 'quadrants' || view === 'points' || view === 'craters'
  const showPoints = view === 'points' || view === 'craters'
  const showCraters = view === 'craters'

  return (
    <div className="observation-frame">
      <img
        src={`data:image/png;base64,${image_png_base64}`}
        alt="Rover observation"
        width={w}
        height={h}
      />
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
        {showQuadrants && (
          <g>
            <line x1={w / 2} y1={0} x2={w / 2} y2={h} className="quadrant-divider" />
            <line x1={0} y1={h / 2} x2={w} y2={h / 2} className="quadrant-divider" />
            {quadrants.map((q) => (
              <text
                key={q.name}
                x={q.x + 10}
                y={q.y + 26}
                className="quadrant-label"
                fill={QUADRANT_COLOR[q.name]}
              >
                {q.name}
              </text>
            ))}
          </g>
        )}

        {showPoints &&
          points.map((p, i) => (
            <circle
              key={i}
              cx={p.u}
              cy={p.v}
              r={2.2}
              fill={QUADRANT_COLOR[p.quadrant]}
              opacity={0.85}
            />
          ))}

        {showCraters &&
          candidates.map((c) => (
            <g key={c.id}>
              <circle
                cx={c.center_u}
                cy={c.center_v}
                r={c.radius_px}
                className="candidate-circle"
              />
              <circle cx={c.center_u} cy={c.center_v} r={3} fill="#ff4a4a" />
              <text x={c.center_u + c.radius_px + 5} y={c.center_v} className="candidate-label">
                {c.id}
              </text>
            </g>
          ))}
      </svg>
    </div>
  )
}

export { QUADRANT_COLOR }
