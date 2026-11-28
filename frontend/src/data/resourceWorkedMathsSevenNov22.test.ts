import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
const quad = readFileSync(new URL('./resourceGuides/1-maths-4.md', import.meta.url), 'utf8')
const section = (n: number) => quad.split(`## ${n}.`)[1].split('\n## ')[0]
describe('seven Maths reasoning topics', () => {
  it('connects a justified diagonal sum to a new variable-angle transfer problem', () => {
    const x = 360 / (2 + 3 + 4 + 6)
    const angles = [2, 3, 4, 6].map(k => k * x)
    expect(x).toBe(24)
    expect(angles).toEqual([48, 72, 96, 144])
    expect(angles.reduce((sum, a) => sum + a, 0)).toBe(360)
    expect(section(1)).toContain('convex quadrilateral')
    expect(section(1)).toContain('self-crossing bow-tie')
    expect(section(1)).toContain('48°, 72°, 96° and 144°')
    expect(section(1)).toContain('not the list of actual angles')
  })
  it('keeps overlapping rectangle, rhombus and square properties independent of orientation', () => {
    const labels = (allSidesEqual: boolean, allAnglesRight: boolean) => [
      ...(allAnglesRight ? ['rectangle'] : []),
      ...(allSidesEqual ? ['rhombus'] : []),
      ...(allSidesEqual && allAnglesRight ? ['square'] : []),
    ]
    expect(labels(false, true)).toEqual(['rectangle'])
    expect(labels(true, false)).toEqual(['rhombus'])
    expect(labels(true, true)).toEqual(['rectangle', 'rhombus', 'square'])
    for (const text of ['A is a rectangle, not a rhombus or square', 'B is a rhombus, not a rectangle or square', 'C is a square and therefore also a rectangle and a rhombus', 'Turning C preserves lengths and angles']) expect(section(3)).toContain(text)
  })
  it('retains the two existing video placements and separate try/check panels', () => {
    const quadKnown=['y9udtA3cPgU','PC9mbAwE3zk','5CeBlu260Rw','-IRXWg2oLY4']
    const quadSelected=quadKnown.filter((id,i)=>![1,3].includes(i)||quad.includes(`[Open the video](https://www.youtube.com/watch?v=${id})`))
    expect([...quad.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(quadSelected.map(id=>`https://www.youtube.com/watch?v=${id}`))
    for(const id of quadSelected)expect(section([1,2,3,4][quadKnown.indexOf(id)])).toContain(`[Open the video](https://www.youtube.com/watch?v=${id})`)
    for (const n of [1, 3]) {
      expect(section(n)).toContain('### Try it yourself')
      expect(section(n)).toContain('**Check:**')
    }
  })
})

const numberPlay = readFileSync(new URL('./resourceGuides/2-maths-5.md', import.meta.url), 'utf8')
const distribute = readFileSync(new URL('./resourceGuides/maths-6.md', import.meta.url), 'utf8')
const block = (text: string, n: number) => text.split(`## ${n}.`)[1].split('\n## ')[0]
describe('multiple, place-value and distribution transfer checks', () => {
  it('proves remainder cancellation for integer pairs without a false converse', () => {
    for (let p = -12; p <= 12; p++) for (let q = -12; q <= 12; q++) {
      const sum = (7 * p + 3) + (7 * q + 4)
      expect(sum).toBe(7 * (p + q + 1))
      expect((sum % 7 + 7) % 7).toBe(0)
    }
    expect(block(numberPlay, 1)).toContain('17 and 25 and the sum is 42')
    expect(block(numberPlay, 1)).toContain('does not force each addend')
  })
  it('enumerates every decimal digit against BOTH divisibility constraints', () => {
    const solutions = Array.from({length: 10}, (_, d) => ({ d, n: 5802 + 10 * d }))
      .filter(({n}) => n % 9 === 0 && n % 4 === 0)
    expect(solutions).toEqual([{d: 3, n: 5832}])
    expect(5832 / 9).toBe(648)
    expect(5832 / 4).toBe(1458)
    expect(block(numberPlay, 2)).toContain('15 + d')
    expect(block(numberPlay, 2)).toContain('ranges from 15 to 24')
    expect(block(numberPlay, 2)).toContain('only the last two digits matter')
  })
  it('finds the LCM by independent positive enumeration rather than multiplying divisors', () => {
    const firstCommon = (a: number, b: number) => Array.from({length: a * b}, (_, k) => k + 1).find(n => n % a === 0 && n % b === 0)
    expect(firstCommon(6, 8)).toBe(24)
    expect(firstCommon(12, 18)).toBe(36)
    expect(36 % 12).toBe(0)
    expect(36 % 18).toBe(0)
    expect(36 % 216).not.toBe(0)
    expect(block(numberPlay, 3)).toContain('2² × 3² = 36')
    expect(block(numberPlay, 3)).toContain('not necessarily a multiple of 216')
  })
  it('checks both near100 products and the reverse-distribution difference', () => {
    expect(28 * 97).toBe(2716)
    expect(46 * 99).toBe(4554)
    expect(46 * 101).toBe(4646)
    expect(46 * 101 - 46 * 99).toBe(46 * (101 - 99))
    expect(46 * (101 - 99)).toBe(92)
    expect(block(distribute, 1)).toContain('4,646 - 4,554 = 92')
    expect(block(distribute, 1)).toContain('each extra group contains 46')
  })
  it('tests signed binomial substitutions and rejects an unsupported all-real ordering', () => {
    for (let k = -20; k <= 20; k++) {
      const x = k / 2
      expect((2 * x + 3) ** 2).toBe(4 * x * x + 12 * x + 9)
      expect((2 * x - 3) * (2 * x + 3)).toBeCloseTo(4 * x * x - 9, 12)
    }
    expect((2 * -2 + 3) ** 2).toBeLessThan((2 * -2 - 3) * (2 * -2 + 3))
    expect(block(distribute, 3)).toContain('At x = 0, the values are 9 and -9; at x = -2, they are 1 and 7')
    expect(block(distribute, 3)).toContain('disprove the claim')
  })
})
