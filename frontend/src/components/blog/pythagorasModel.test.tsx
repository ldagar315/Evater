import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { pythagorasModel, Point } from './pythagorasModel'
import { PythagorasSimulation } from './PythagorasSimulation'
const area = (p: Point[]) => Math.abs(p.reduce((sum, q, i) => { const next = p[(i + 1) % p.length]; return sum + q.x * next.y - next.x * q.y }, 0)) / 2

describe('Pythagoras area model', () => {
  it('constructs true squares with exact area relationship, outside a right triangle, for all144settings', () => {
    for (let a = 1; a <= 12; a++) for (let b = 1; b <= 12; b++) {
      const m = pythagorasModel(a, b)
      const expected = [a * a, b * b, a * a + b * b]
      m.squares.forEach((square, i) => {
        expect(area(square)).toBe(expected[i])
        square.forEach((p, j) => {
          const q = square[(j + 1) % 4], r = square[(j + 2) % 4]
          expect((q.x - p.x) ** 2 + (q.y - p.y) ** 2).toBe(expected[i])
          expect((q.x - p.x) * (r.x - q.x) + (q.y - p.y) * (r.y - q.y)).toBeCloseTo(0, 10)
          const plot = m.plot(p)
          expect(plot.x).toBeGreaterThanOrEqual(39.999)
          expect(plot.x).toBeLessThanOrEqual(440.001)
          expect(plot.y).toBeGreaterThanOrEqual(29.999)
          expect(plot.y).toBeLessThanOrEqual(370.001)
        })
      })
      expect(m.c ** 2).toBeCloseTo(m.cSquared, 10)
      expect(area(m.triangle)).toBe(a * b / 2)
      expect(m.squares.map(area).reduce((s, x) => s + x, 0) + area(m.triangle)).toBe(2 * m.cSquared + a * b / 2)
    }
  })
  it('distinguishes exact integer triples from rounded irrational hypotenuses', () => {
    expect(pythagorasModel(3, 4).c).toBe(5)
    expect(pythagorasModel(5, 12).c).toBe(13)
    expect(pythagorasModel(1, 1).exactHypotenuse).toBe(false)
    expect(pythagorasModel(1, 1).cSquared).toBe(2)
  })
  it('rejects unsupported values', () => { for (const n of [0, 13, -1, NaN, Infinity, 1.5]) { expect(() => pythagorasModel(n, 3)).toThrow(RangeError); expect(() => pythagorasModel(3, n)).toThrow(RangeError) } })
  it('labels controls and states limits instead of claiming proof or3D', () => {
    const html = renderToStaticMarkup(<PythagorasSimulation />)
    expect((html.match(/type="range"/g) || [])).toHaveLength(2)
    for (const text of ['Perpendicular side a: 3 cm', 'Perpendicular side b: 4 cm', '9 + 16 = 25 square cm', 'not a proof', 'drawing rescales', 'right angle', 'Striped orange', 'Dotted purple']) expect(html).toContain(text)
  })
  it('uses independent pattern ids', () => {
    const html = renderToStaticMarkup(<><PythagorasSimulation /><PythagorasSimulation /></>)
    const ids = [...html.matchAll(/<pattern id="([^"]+)"/g)].map(m => m[1])
    expect(ids).toHaveLength(4); expect(new Set(ids).size).toBe(4)
  })
})
