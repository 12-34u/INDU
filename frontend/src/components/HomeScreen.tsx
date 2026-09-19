import type { InspectorView, TerrainData } from '../types'

interface Props {
  terrain: (TerrainData & { gsd_m: number }) | null
  onEnter: (view: InspectorView) => void
}

/**
 * Landing screen.
 *
 * Deliberately separate from the console: this is an editorial index of the
 * pipeline, not an instrument panel. Nothing here runs anything - each entry
 * hands a starting view to the console and gets out of the way.
 */
const ENTRIES: { no: string; title: string; sub: string; view: InspectorView; action: string }[] = [
  { no: '01', title: 'Rover', sub: 'Observation', view: 'image', action: 'Explore' },
  { no: '02', title: 'Crater', sub: 'Map', view: 'craters', action: 'Explore' },
  { no: '03', title: 'Rover', sub: 'Localization', view: 'position', action: 'Read More' },
  { no: '04', title: 'Reference', sub: 'Matching', view: 'match', action: 'Read More' },
]

export const HomeScreen: React.FC<Props> = ({ terrain, onEnter }) => (
  <div className="home">
    <div className="home-rules">
      <span /><span /><span /><span />
    </div>

    <header className="home-top">
      <div className="home-mark">
        <i className="home-moon-glyph" />
        <span>indu</span>
      </div>
      <div className="home-top-right">
        <span>EN</span>
        <span>⌕</span>
      </div>
    </header>

    <div className="home-grid">
      <h1 className="home-title">
        Explore<br />the<br />Moon
      </h1>

      {ENTRIES.map((e) => (
        <button key={e.no} className="home-entry" onClick={() => onEnter(e.view)}>
          <span className="home-entry-no">{e.no}</span>
          <span className="home-entry-title">
            {e.title}
            <br />
            {e.sub}
          </span>
          <span className="home-entry-action">{e.action}</span>
        </button>
      ))}
    </div>

    <div className="home-side">
      <span>Observe</span>
      <span>Localize</span>
      <span>Navigate</span>
    </div>

    <footer className="home-bottom">
      <div className="home-pager">
        <span className="on">01</span>
        <span>02</span>
        <span>03</span>
      </div>
      <button className="home-about" onClick={() => onEnter('image')}>
        Enter Console
      </button>
      <div className="home-stat">
        {terrain ? (
          <>
            {terrain.max_slope.toFixed(0)}
            <small>°</small>
          </>
        ) : (
          <>
            --<small>°</small>
          </>
        )}
      </div>
    </footer>
  </div>
)
