import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
const powers = readFileSync(new URL('./resourceGuides/maths-2.md', import.meta.url), 'utf8')
const ratios = readFileSync(new URL('./resourceGuides/maths-7.md', import.meta.url), 'utf8')
const block = (s: string, n: number) => s.split(`## ${n}.`)[1].split('\n## ')[0]
describe('eight powers and ratio reasoning transfers', () => {
  it('counts independent repeated choices and distinguishes adding a position', () => {
    const labels = (alphabet: string[], places: number): string[] => places === 0 ? [''] : labels(alphabet, places - 1).flatMap(s => alphabet.map(c => s + c))
    expect(new Set(labels(['A', 'B'], 3)).size).toBe(8)
    expect(new Set(labels(['A', 'B', 'C'], 3)).size).toBe(27)
    expect(new Set(labels(['A', 'B'], 4)).size).toBe(16)
    expect(3 * 8).not.toBe(27)
    expect((-2) ** 2).toBe(4)
    expect(-(2 ** 2)).toBe(-4)
    for (const text of ['independently', 'letters may repeat', '27, not 3 × 8 = 24', 'minus is outside the power']) expect(block(powers, 1)).toContain(text)
  })
  it('distinguishes product, quotient, power of a power, and a sum', () => {
    expect((3 ** 2) ** 3 / 3 ** 4).toBe(9)
    expect(3 ** 2 + 3 ** 3).toBe(36)
    expect(3 ** (2 + 3)).toBe(243)
    for (let base = -5; base <= 5; base++) for (let m = 1; m <= 4; m++) for (let n = 1; n <= 4; n++) {
      expect(base ** m * base ** n).toBe(base ** (m + n))
      expect((base ** m) ** n).toBe(base ** (m * n))
      if (base) expect(base ** m / base ** n).toBeCloseTo(base ** (m - n), 10)
    }
    for (const text of ['3² = 9, not 3¹', '36, whereas 3⁵ = 243', 'nonzero base']) expect(block(powers, 2)).toContain(text)
  })
  it('checks reciprocals and their domain rather than treating negative exponents as negative answers', () => {
    for (const base of [-7, -3, -2, 2, 3, 7]) for (let n = 1; n <= 5; n++) expect(base ** -n * base ** n).toBeCloseTo(1, 12)
    expect(2 ** -3).toBe(1 / 8)
    expect((-2) ** -3).toBe(-1 / 8)
    expect(7 ** 0).toBe(1)
    for (const text of ['nonzero base', 'do not define 0⁰ or 0⁻¹', '1/0, which is undefined', 'base and exponent both matter']) expect(block(powers, 3)).toContain(text)
  })
  it('checks scientific normalization, unchanged value, and a ratio with equal exponents', () => {
    expect(8.4 * 10 ** -5).toBeCloseTo(0.000084, 12)
    expect(84 * 10 ** -6).toBeCloseTo(0.000084, 12)
    expect(8.4 / 6.3).toBeCloseTo(4 / 3, 12)
    for (const coefficient of [1, 1.2, 5.5, 8.4, 9.99]) for (let k = -6; k <= 6; k++) {
      const value = coefficient * 10 ** k
      expect((coefficient * 10) * 10 ** (k - 1)).toBeCloseTo(value, 7)
    }
    for (const text of ['1 ≤ c < 10', 'integer k', 'value is correct but', 'A/B = 8.4/6.3 = 4/3', 'fictional measured lengths']) expect(block(powers, 4)).toContain(text)
  })
  it('distinguishes part to part, part to whole, and equal additions', () => {
    expect(18 / 30).toBe(3 / 5)
    expect(18 / (18 + 30)).toBe(3 / 8)
    expect((18 + 6) / (30 + 6)).toBe(2 / 3)
    expect(2 * 5).not.toBe(3 * 3)
    for (const text of ['Green : all = 18 : 48 = 3 : 8', 'equal-sized groups', 'Equal additions do not multiply']) expect(block(ratios, 1)).toContain(text)
  })
  it('checks common-unit invariance and one-sided scaling', () => {
    expect(120 / 80).toBeCloseTo(1.2 / 0.8, 12)
    expect(120 / 80).toBe(3 / 2)
    expect(120 / 160).toBe(3 / 4)
    for (const x of [0.5, 1.2, 2, 3.75]) for (const y of [0.1, 0.8, 2.5]) expect(x / y).toBeCloseTo((100 * x) / (100 * y), 12)
    for (const text of ['120 : 80 = 3 : 2', '120 : 160 = 3 : 4', 'before matching their units']) expect(block(ratios, 2)).toContain(text)
  })
  it('separates constant item price from a once-per-order fixed charge', () => {
    const price = (170 - 20) / 5
    expect(price).toBe(30)
    expect(8 * price + 20).toBe(260)
    expect(170 / 5 * 8).toBe(272)
    expect(10 * price + 20).toBe(320)
    for (const text of ['same single delivery charge', 'constant folder price', '₹320, not ₹340', 'item cost doubles', '₹260']) expect(block(ratios, 3)).toContain(text)
  })
  it('finds a ratio scale from the difference and verifies both constraints', () => {
    const unit = 270 / (7 - 4), a = 4 * unit, b = 7 * unit
    expect([a, b, a + b]).toEqual([360, 630, 990])
    expect(b - a).toBe(270)
    expect(a / b).toBe(4 / 7)
    expect(270 / 11).not.toBe(unit)
    for (const text of ['difference is 7 - 4 = 3', 'total is ₹990', '₹630 - ₹360 = ₹270', 'question did not give that total']) expect(block(ratios, 4)).toContain(text)
  })
  it('preserves the four existing videos, four try/check pairs, and revision on each page', () => {
    for (const text of [powers, ratios]) {
      expect((text.match(/\[Open the video\]/g) || []).length).toBe(4)
      expect((text.match(/### Try it yourself/g) || []).length).toBe(4)
      expect((text.match(/\*\*Check:\*\*/g) || []).length).toBe(4)
      expect(text).toContain('## Your five-minute revision')
    }
  })
})
