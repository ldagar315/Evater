import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const guide = (name: string) => readFileSync(new URL(`./resourceGuides/${name}.md`, import.meta.url), 'utf8')
const pythagoras = guide('3-maths-9')
const area = guide('8-maths-14')
const fractions = guide('maths-8')
const section = (text: string, n: number) => text.split(`## ${n}.`)[1].split('\n## ')[0]

describe('ten Pythagoras, area and percentage transfer lessons', () => {
  it('has original teaching and independent check panels in all ten selected sections', () => {
    for (const [text, topics] of [[pythagoras, [1, 2, 3, 4]], [area, [2, 3, 4]], [fractions, [1, 2, 3]]] as const) {
      for (const n of topics) {
        const block = section(text, n)
        expect(block).toContain('### Try it yourself')
        expect(block).toContain('**Check:**')
        expect(block).toMatch(/### (Understand it|Think it through)/)
      }
    }
  })

  it('identifies the hypotenuse from the right-angle vertex, not orientation', () => {
    const block = section(pythagoras, 1)
    expect(block).toContain('right-angled at S')
    expect(block).toContain('RT is opposite the right angle at S')
    expect(Math.hypot(7, 24)).toBe(25)
    expect(block).toContain('RT = 25 cm')
  })

  it('checks both unknown-leg subtraction and unknown-hypotenuse addition', () => {
    expect(Math.sqrt(37 ** 2 - 12 ** 2)).toBe(35)
    expect(Math.hypot(20, 21)).toBe(29)
    expect(section(pythagoras, 2)).toContain('35 cm')
    expect(section(pythagoras, 2)).toContain('29 cm')
  })

  it('distinguishes right, non-right and impossible triples with the triangle inequality', () => {
    const classify = (sides: number[]) => {
      const [a, b, c] = [...sides].sort((x, y) => x - y)
      if (a + b <= c) return 'impossible'
      return a * a + b * b === c * c ? 'right' : 'non-right'
    }
    expect(classify([26, 10, 24])).toBe('right')
    expect(classify([10, 24, 25])).toBe('non-right')
    expect(classify([4, 7, 12])).toBe('impossible')
    expect(section(pythagoras, 3)).toContain('no triangle exists')
  })

  it('independently constructs the central square and checks the general rearrangement', () => {
    for (let a = 1; a <= 12; a++) for (let b = 1; b <= 12; b++) {
      const vertices = [[a, 0], [a + b, a], [b, a + b], [0, b]]
      const vectors = vertices.map((v, i) => [vertices[(i + 1) % 4][0] - v[0], vertices[(i + 1) % 4][1] - v[1]])
      vectors.forEach((v, i) => {
        const w = vectors[(i + 1) % 4]
        expect(v[0] * w[0] + v[1] * w[1]).toBe(0)
        expect(v[0] ** 2 + v[1] ** 2).toBe(a * a + b * b)
      })
      const centralArea = Math.abs(vertices.reduce((sum, v, i) => {
        const w = vertices[(i + 1) % 4]
        return sum + v[0] * w[1] - v[1] * w[0]
      }, 0)) / 2
      expect(centralArea).toBe((a + b) ** 2 - 2 * a * b)
    }
    expect(17 ** 2 - 4 * 0.5 * 5 * 12).toBe(169)
    expect(section(pythagoras, 4)).toContain('side 13 cm')
    expect(section(pythagoras, 4)).toContain('neighbouring acute angles add to 90°')
  })

  it('pairs a new parallelogram base with its corresponding perpendicular height', () => {
    expect(12 * 5).toBe(60)
    expect(13 * (60 / 13)).toBeCloseTo(60, 10)
    expect(12 * 13).not.toBe(60)
    expect(section(area, 2)).toContain('60/13 cm')
    expect(section(area, 2)).toContain('not a base and its perpendicular height')
  })

  it('adds the non-overlapping triangle areas on opposite sides of a quadrilateral diagonal', () => {
    expect(0.5 * 11 * 8).toBe(44)
    expect(0.5 * 16 * 3 + 0.5 * 16 * 5).toBe(64)
    expect(section(area, 3)).toContain('64 cm²')
    expect(section(area, 3)).toContain('stay unchanged')
  })

  it('checks trapezium, rhombus and hole subtraction separately', () => {
    expect(0.5 * (9 + 15) * 4).toBe(48)
    expect(0.5 * 8 * 12).toBe(48)
    expect(15 * 8 - 3 * 4).toBe(108)
    expect(section(area, 4)).toContain('108 cm²')
    expect(section(area, 4)).toContain('does not imply congruence')
  })

  it('converts a new fraction and its complement while preserving the whole', () => {
    expect(11 / 40).toBe(0.275)
    expect(29 / 40).toBe(0.725)
    expect(27.5 + 72.5).toBe(100)
    expect(section(fractions, 1)).toContain('11/40 = 0.275 = 27.5%')
    expect(section(fractions, 1)).toContain('29/40 = 0.725 = 72.5%')
  })

  it('compares amounts rather than percentages of different wholes', () => {
    expect(0.125 * 360).toBe(45)
    expect(0.2 * 200).toBe(40)
    expect(45 / 0.125).toBe(360)
    expect(section(fractions, 2)).toContain('A has 5 g more despite its lower percentage')
  })

  it('uses each successive base and derives the inverse reduction', () => {
    expect(500 * 1.2).toBe(600)
    expect(600 * 0.8).toBe(480)
    expect((500 - 480) / 500 * 100).toBe(4)
    expect(600 * (1 - 1 / 6)).toBe(500)
    expect(section(fractions, 3)).toContain('16⅔%')
    expect(section(fractions, 3)).toContain('different bases')
  })

  it('retains the previously completed extensions and existing video inventory', () => {
    expect(section(pythagoras, 5)).toContain('6√2 cm')
    expect(section(pythagoras, 6)).toContain('Andrew Wiles')
    for (const [text, count] of [[pythagoras, 3], [area, 3], [fractions, 3]] as const) {
      const knownAreaVideo = text === area && text.includes("[Open the video](https://www.youtube.com/watch?v=iNZZIb_-PA8)") ? 1 : 0
      const knownPythagorasVideo0 = text === pythagoras && text.includes("[Open the video](https://www.youtube.com/watch?v=UKaoHO4ClS4)") ? 1 : 0
      const knownPythagorasVideo1 = text === pythagoras && text.includes("[Open the video](https://www.youtube.com/watch?v=Eo04qjLRiSY)") ? 1 : 0
      expect((text.match(/\[Open the video\]/g) || []).length).toBe(count + knownAreaVideo + knownPythagorasVideo0 + knownPythagorasVideo1)
    }
  })
})
