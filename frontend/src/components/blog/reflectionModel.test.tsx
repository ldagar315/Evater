import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { reflectionModel } from './reflectionModel'
import { ReflectionSimulation } from './ReflectionSimulation'

describe('plane mirror reflection', () => {
  it('obeys equal angles measured from the normal at every supported value', () => {
    for (let angle = 0; angle <= 75; angle++) {
      const m = reflectionModel(angle)
      const incidence = Math.atan2(210 - m.incoming.x, 250 - m.incoming.y) * 180 / Math.PI
      const reflection = Math.atan2(m.outgoing.x - 210, 250 - m.outgoing.y) * 180 / Math.PI
      expect(incidence).toBeCloseTo(angle, 10)
      expect(reflection).toBeCloseTo(angle, 10)
      expect(m.incoming.y).toBeCloseTo(m.outgoing.y, 10)
      expect(m.incoming.x + m.outgoing.x).toBeCloseTo(420, 10)
      expect(m.surfaceAngle).toBe(90 - angle)
      expect(Math.hypot(m.incoming.x - 210, m.incoming.y - 250)).toBeCloseTo(190, 10)
      expect(m.incoming.x).toBeGreaterThanOrEqual(16)
      expect(m.outgoing.x).toBeLessThanOrEqual(404)
      expect(m.incoming.y).toBeGreaterThanOrEqual(60)
      expect(m.incoming.y).toBeLessThan(250)
    }
  })
  it('retraces the normal at zero with no degenerate arc', () => {
    const m = reflectionModel(0)
    expect(m.incoming).toEqual(m.outgoing)
    expect(m.incoming.x).toBe(210)
    expect(m.incidentPath).toBe('M210.000,60.000 L210,250')
    expect(m.reflectedPath).toBe('M210,250 L210.000,60.000')
    expect(m.incidentArc).toBe('')
    expect(m.reflectedArc).toBe('')
  })
  it('rejects values outside the slider domain', () => {
    for (const angle of [-1, 76, 0.5, NaN, Infinity]) expect(() => reflectionModel(angle)).toThrow(RangeError)
  })
  it('labels its control and distinguishes this from a 3D or lens model', () => {
    const html = renderToStaticMarkup(<ReflectionSimulation />)
    expect(html).toContain('Angle of incidence: 45 degrees')
    expect(html).toContain('Angle of incidence = angle of reflection')
    expect(html).toContain('A 2D diagram')
    expect(html).toContain('not model image formation')
    expect(html).toContain('normal is a reference line, not another ray')
    expect(html).toContain('Solid: incident ray')
    expect(html).toContain('Dashed: reflected ray')
    expect((html.match(/type="range"/g) || []).length).toBe(1)
    expect(html).not.toContain('<canvas')
  })
  it('gives independent SVG marker ids for multiple rendered models', () => {
    const html = renderToStaticMarkup(<><ReflectionSimulation /><ReflectionSimulation /></>)
    const ids = [...html.matchAll(/<marker id="([^"]+)"/g)].map(match => match[1])
    expect(ids).toHaveLength(4)
    expect(new Set(ids).size).toBe(4)
  })
})
