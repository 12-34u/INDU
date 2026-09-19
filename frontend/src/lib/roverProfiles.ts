/**
 * How fast the rover is allowed to be.
 *
 * The physics underneath is real: lunar gravity, traction-limited drive,
 * rolling resistance, and a climb limit that falls out of the friction
 * coefficient rather than being declared. A real lunar vehicle is slow - the
 * Apollo LRV managed about 4.5 m/s and a robotic rover is a hundred times
 * slower still - which is faithful and not much fun to drive across a 2.4 km
 * sector.
 *
 * So there is a deliberate, named exception rather than a quietly inflated
 * constant. `fast` multiplies EVERY acceleration by the same factor - drive,
 * gravity and rolling resistance alike - which is equivalent to driving a
 * stronger vehicle on a heavier world. Because the terms scale together:
 *
 *   - the slope penalty is unchanged. A 9% grade costs 21% of available drive
 *     in both profiles, so climbs and descents feel exactly as they should.
 *   - the climb limit is unchanged at atan(traction) = 26.6 degrees, because
 *     slip compares two quantities that scale identically.
 *
 * What is NOT faithful is the speed itself, and the UI says so.
 */

export interface SpeedProfile {
  id: string
  label: string
  /** Motor-limited top speed, m/s. */
  maxForward: number
  maxReverse: number
  /** Multiplies every acceleration. 1 is real lunar physics. */
  factor: number
  /** Whether the speeds are physically faithful. */
  physical: boolean
  note: string
}

export const SPEED_PROFILES: SpeedProfile[] = [
  {
    id: 'lrv',
    label: 'Apollo LRV',
    maxForward: 4.5,
    maxReverse: 1.8,
    factor: 1,
    physical: true,
    note: 'Real lunar physics. Top speed matches the Apollo Lunar Roving Vehicle.',
  },
  {
    id: 'fast',
    label: 'Fast',
    maxForward: 18,
    maxReverse: 6,
    factor: 4,
    physical: false,
    note:
      'Not a real rover speed. Every acceleration is scaled by 4, so slope still ' +
      'costs the same share of drive and the 26.6° climb limit is unchanged — only ' +
      'the speeds are faster than anything that has driven on the Moon.',
  },
]

export const DEFAULT_PROFILE_ID = 'fast'

export const profileById = (id: string): SpeedProfile =>
  SPEED_PROFILES.find((p) => p.id === id) ?? SPEED_PROFILES[0]
