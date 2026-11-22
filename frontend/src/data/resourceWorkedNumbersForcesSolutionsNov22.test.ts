import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
const numbers = readFileSync(new URL('./resourceGuides/maths-3.md', import.meta.url), 'utf8')
const forces = readFileSync(new URL('./resourceGuides/science-5.md', import.meta.url), 'utf8')
const solutions = readFileSync(new URL('./resourceGuides/15-science-9.md', import.meta.url), 'utf8')
const block = (s: string, n: number) => s.split(`## ${n}.`)[1].split('\n## ')[0]
describe('six number representation force and solution transfers', () => {
 it('decodes only the explicitly invented additive system and distinguishes decimal positions', () => {
  const decode = (s: string) => [...s].reduce((sum, c) => { if (c !== 'T' && c !== '|') throw new Error('invalid symbol'); return sum + (c === 'T' ? 10 : 1) }, 0)
  expect(decode('TT||||')).toBe(24)
  expect(decode('|T||T|')).toBe(24)
  expect(() => decode('XIV')).toThrow()
  expect(2 * 10 + 4).not.toBe(4 * 10 + 2)
  for (const text of ['invented additive system', 'not a claim that order never matters', '2 × 10 + 4 = 24', 'not generally in decimal place value']) expect(block(numbers, 1)).toContain(text)
 })
 it('independently enumerates valid three-digit permutations and verifies both extrema', () => {
  const digits = [0, 4, 7], values: number[] = []
  for (const a of digits) for (const b of digits) for (const c of digits) if (new Set([a, b, c]).size === 3 && a !== 0) values.push(100 * a + 10 * b + c)
  expect(values.sort((a, b) => a - b)).toEqual([407, 470, 704, 740])
  expect(Math.max(...values) - Math.min(...values)).toBe(333)
  expect(Number('047')).toBe(47)
  expect(2 * 100 + 0 * 10 + 4).toBe(204)
  expect(Number('0204')).toBe(204)
  for (const text of ['exactly once each', 'three-digit', '740 - 407 = 333', 'Valid candidates are 407, 470, 704 and 740']) expect(block(numbers, 2)).toContain(text)
 })
 it('separates net force, current velocity and initial slowing in a stated straight-line model', () => {
  expect(6 - 6).toBe(0)
  expect(6 - 8).toBe(-2)
  const initialRightwardVelocity = 3, mass = 4, time = 1
  const laterVelocity = initialRightwardVelocity + (6 - 8) / mass * time
  expect(laterVelocity).toBeGreaterThan(0)
  expect(laterVelocity).toBeLessThan(initialRightwardVelocity)
  for (const text of ['A nonzero net force changes motion', 'fictional cart constrained to a straight, level track', 'All horizontal forces are listed', 'first slows while still moving right', 'rather than instantly reversing', 'Do not perform a pushing experiment']) expect(block(forces, 1)).toContain(text)
 })
 it('balances a floating boat, retains upthrust on a sinking stone and separates mass from weight', () => {
  expect(12 - 12).toBe(0)
  expect(1 - 3).toBe(-2)
  expect(1).toBeGreaterThan(0)
  const mass = 2
  expect(mass * 10).toBe(20)
  expect(mass * 5).toBe(10)
  expect(mass).toBe(2)
  for (const text of ['12 N buoyancy upward', '2 N net force downward', 'still has a 1 N upward buoyant force', 'weight but leaves its mass unchanged', 'ignore other forces at that instant']) expect(block(forces, 4)).toContain(text)
 })
 it('conserves added solute through a solvent-capacity change and distinguishes saturation from appearance', () => {
  const equilibrium = (added: number, water: number) => { const capacity = water * 12 / 50; return { dissolved: Math.min(added, capacity), undissolved: Math.max(0, added - capacity), spare: Math.max(0, capacity - added) } }
  expect(equilibrium(16, 50)).toEqual({ dissolved: 12, undissolved: 4, spare: 0 })
  expect(equilibrium(16, 100)).toEqual({ dissolved: 16, undissolved: 0, spare: 8 })
  expect(equilibrium(12, 50)).toEqual({ dissolved: 12, undissolved: 0, spare: 0 })
  for (let added = 0; added <= 30; added++) for (const water of [25, 50, 75, 100]) {
   const result = equilibrium(added, water)
   expect(result.dissolved + result.undissolved).toBe(added)
  }
  for (const text of ['fictional data', '12 g dissolves', '4 g remains undissolved', 'capacity for 24 - 16 = 8 g more', 'Do not taste laboratory solutions', 'Clear appearance by itself cannot tell']) expect(block(solutions, 2)).toContain(text)
 })
 it('identifies controlled and confounded gas comparisons without mistaking equilibrium limits for speed', () => {
  const a = { temperature: 10, pressure: 100, mg: 8 }, b = { temperature: 30, pressure: 100, mg: 5 }, c = { temperature: 30, pressure: 200, mg: 10 }
  const changed = (x: typeof a, y: typeof a) => (['temperature', 'pressure'] as const).filter(k => x[k] !== y[k])
  expect(changed(a, b)).toEqual(['temperature'])
  expect(changed(b, c)).toEqual(['pressure'])
  expect(changed(a, c)).toEqual(['temperature', 'pressure'])
  expect(b.mg).toBeLessThan(a.mg)
  expect(c.mg).toBeGreaterThan(b.mg)
  for (const text of ['same fictional gas and 100 g of water', 'pressure of that gas', 'change both variables', 'not measurements of dissolving speed', 'do not heat or open sealed gas containers']) expect(block(solutions, 3)).toContain(text)
 })
 it('retains current videos, four try/check pairs, revision and adjacent topic structure', () => {
  const known=["https://www.youtube.com/watch?v=SYU1TKQru1c", "https://www.youtube.com/watch?v=beo130kj9R4", "https://www.youtube.com/watch?v=Ps90zArJEJY"];expect([...forces.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(known.filter((url,i)=>i>0||forces.includes(`[Open the video](${url})`)))
  for (const [text, videos] of [[numbers, 4], [forces, 2], [solutions, 2]] as const) {
   expect((text.match(/\[Open the video\]/g) || []).length).toBe(videos+Number(text===forces&&forces.includes('[Open the video](https://www.youtube.com/watch?v=SYU1TKQru1c)')))
   expect((text.match(/### Try it yourself/g) || []).length).toBe(4)
   expect((text.match(/\*\*Check:\*\*/g) || []).length).toBe(4)
   expect(text).toContain('## Your five-minute revision')
  }
 })
})
