import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { ForceSimulation } from './ForceSimulation'
import { forceModel } from './forceModel'

describe('opposing horizontal forces', () => {
  it('computes every supported pair without inventing motion', () => {
    for (let right = 0; right <= 10; right++) for (let left = 0; left <= 10; left++) {
      const m = forceModel(right, left)
      expect(m.net).toBe(right - left)
      expect(m.magnitude).toBe(Math.abs(right - left))
      expect(m.direction).toBe(right === left ? 'balanced' : right > left ? 'right' : 'left')
      expect(m.rightLength).toBe(right * 8)
      expect(m.leftLength).toBe(left * 8)
      expect(m).not.toHaveProperty('speed')
      expect(m).not.toHaveProperty('position')
    }
  })
  it('treats no push and equal nonzero pushes as balanced', () => {
    expect(forceModel(0, 0).result).toBe('0 N: the horizontal forces are balanced.')
    expect(forceModel(10, 10).net).toBe(0)
  })
  it('rejects unsupported control values', () => {
    for (const value of [-1, 11, 0.5, NaN, Infinity]) {
      expect(() => forceModel(value, 1)).toThrow(RangeError)
      expect(() => forceModel(1, value)).toThrow(RangeError)
    }
  })
  it('labels both keyboard controls and explains model limits', () => {
    const html = renderToStaticMarkup(<ForceSimulation />)
    expect((html.match(/type="range"/g) || []).length).toBe(2)
    expect((html.match(/<label /g) || []).length).toBe(2)
    expect(html).toContain('Push to the right: 6 N')
    expect(html).toContain('Push to the left: 4 N')
    expect(html).toContain('2 N to the right.')
    expect(html).toContain('Friction and turning are ignored')
    expect(html).toContain('not a prediction of speed or distance')
    expect(html).toContain('aria-live="polite"')
    expect(html).not.toContain('setInterval')
  })
})
