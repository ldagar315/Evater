import {describe, expect, it} from 'vitest'
import {readFileSync} from 'node:fs'
import {compressionDrawing, compressionVolumes, type CompressionVolume, particleCount, particleRadius} from './compressionModel'
describe('finite sealed gas drawings', () => {
  for (const volume of compressionVolumes) {
    it(`keeps count, symbol size, containment and separation at ${volume}`, () => {
      const m = compressionDrawing(volume)
      expect(m.count).toBe(16); expect(m.particles).toHaveLength(particleCount)
      expect(m.width * m.height / (300 * 140)).toBe(volume / 100)
      for (const p of m.particles) {
        expect(p.radius).toBe(particleRadius)
        expect(p.x - p.radius).toBeGreaterThan(m.left)
        expect(p.x + p.radius).toBeLessThan(m.left + m.width)
        expect(p.y - p.radius).toBeGreaterThan(m.top)
        expect(p.y + p.radius).toBeLessThan(m.top + m.height)
      }
      for (let i = 0; i < m.count; i++) for (let j = i + 1; j < m.count; j++) {
        const a = m.particles[i], b = m.particles[j]
        expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(a.radius + b.radius)
      }
    })
  }
  it('reduces horizontal gaps rather than particle dimensions, count or drawing scale', () => {
    const samples = compressionVolumes.map(compressionDrawing)
    const gaps = samples.map(m => m.particles[1].x - m.particles[0].x - 2 * particleRadius)
    expect(gaps[0]).toBeGreaterThan(gaps[1]); expect(gaps[1]).toBeGreaterThan(gaps[2])
    for (const m of samples) {expect(m.height).toBe(140); expect(m.left).toBe(30); expect(m.top).toBe(40)}
    expect(samples.map(m => m.particles.map(p => p.radius))).toEqual(Array(3).fill(Array(16).fill(6)))
  })
  it('rejects unsupplied views rather than inventing physical results', () => {
    for (const v of [0, -1, 60, 200, NaN, Infinity]) expect(() => compressionDrawing(v as CompressionVolume)).toThrow(RangeError)
  })
  it('returns fresh coordinates and supports a deterministic reset', () => {
    const m = compressionDrawing(100); m.particles[0].x = -999
    expect(compressionDrawing(100).particles[0].x).toBe(67.5)
  })
  it('keeps safety, honest model limits and a distinct prediction', () => {
    const source = readFileSync(new URL('./GasCompression.tsx', import.meta.url), 'utf8')
    for (const s of ['24 same-size symbols','120 relative volume units','80 units','no gas escaping','Practically incompressible','not a physical simulation','keep moving randomly','No pressure, force, temperature or rate is calculated','do not pressurise sealed containers','Reset gas view','aria-pressed','aria-live','aria-expanded']) expect(source).toContain(s)
    expect(source).not.toMatch(/setInterval|requestAnimationFrame|Math\.random|type="range"/)
  })
  it('mounts on the matching resource only and retains all original prose', () => {
    const page = readFileSync(new URL('../../pages/BlogPostPage.tsx', import.meta.url), 'utf8')
    expect(page).toContain("post.slug === 'class-8-science-particulate-nature-of-matter-study-resources' && <div")
    expect(page.match(/<GasCompression\s*\/>/g)).toHaveLength(1)
    const lesson = readFileSync(new URL('../../data/resourceGuides/13-science-7.md', import.meta.url), 'utf8')
    for (const s of ['## 1. The particle model','## 2. Solids, liquids and gases','## 3. Diffusion','## 4. Changes of state','## Your five-minute revision','4Q45iIeMdCM','KRLNDTmBFZY','xYU7RSoOZ0U']) expect(lesson).toContain(s)
  })
})
