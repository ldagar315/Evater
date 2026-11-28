import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
const numbers = readFileSync(new URL('./resourceGuides/2-maths-5.md', import.meta.url), 'utf8')
const pyth = readFileSync(new URL('./resourceGuides/3-maths-9.md', import.meta.url), 'utf8')
const block = (s: string, n: number) => s.split(`## ${n}.`)[1].split('\n## ')[0]
describe('six independent number-play and Pythagoras transfers', () => {
 it('enumerates every allowed cryptarithm assignment rather than assuming uniqueness', () => {
  const solutions: number[][] = []
  for (let a = 1; a <= 9; a++) for (let b = 1; b <= 9; b++) if (a !== b && a % 2 === 0 && a < b && (10 * a + b) + (10 * b + a) === 121) solutions.push([a, b])
  expect(solutions).toEqual([[2, 9], [4, 7]])
  for (const phrase of ['AB + BA = 121', 'A is even and A < B', 'every assignment', 'There are two assignments']) expect(block(numbers, 4)).toContain(phrase)
 })
 it('checks all32 sign patterns for51 starting values and rejects an odd sum of zero', () => {
  for (let n = 0; n <= 50; n++) for (let mask = 0; mask < 32; mask++) {
   let sum = 0
   for (let i = 0; i < 5; i++) sum += (mask & (1 << i) ? -1 : 1) * (n + i)
   expect(Math.abs(sum) % 2).toBe(n % 2)
   if (n === 7) expect(sum).not.toBe(0)
  }
  for (const phrase of ['five consecutive whole numbers', '5n + 10', 'same parity as n', 'Zero is even']) expect(block(numbers, 5)).toContain(phrase)
 })
 it('checks the successor and square of101 possible numbers with remainder6', () => {
  for (let k = 0; k <= 100; k++) {
   const n = 7 * k + 6
   expect((n + 1) % 7).toBe(0)
   expect(n ** 2 % 7).toBe(1)
   expect(n % 7).toBe(6)
  }
  for (const phrase of ['remainder of N²', '7(7k² + 12k + 5) + 1', 'five complete groups of 7', '169 = 7 × 24 + 1']) expect(block(numbers, 6)).toContain(phrase)
 })
 it('distinguishes a rejected answer from a root-consistent but still incorrect answer', () => {
  const dr = (n: number): number => { while (n >= 10) n = [...String(n)].reduce((s, d) => s + Number(d), 0); return n }
  expect(dr(37) * dr(24)).toBe(6)
  expect(dr(897)).toBe(6)
  expect(dr(898)).toBe(7)
  expect(37 * 24).toBe(888)
  expect(dr(888)).toBe(6)
  expect(897 - 888).toBe(9)
  for (const phrase of ['897 and 898', '898', 'both proposed answers are wrong', 'Passing is not the same as proving']) expect(block(numbers, 7)).toContain(phrase)
 })
 it('checks the rectangle counterexample and exactly when the doubled-area rule holds', () => {
  expect(9 ** 2 + 12 ** 2).toBe(15 ** 2)
  expect(15 ** 2).toBe(225)
  expect(2 * 9 * 12).toBe(216)
  for (let a = 1; a <= 25; a++) for (let b = 1; b <= 25; b++) expect(a * a + b * b === 2 * a * b).toBe(a === b)
  for (const phrase of ['9 cm by 12 cm rectangle', '225 cm², not 216 cm²', 'unequal perpendicular sides', 'not to every rectangle']) expect(block(pyth, 5)).toContain(phrase)
 })
 it('tests arithmetic and domain separately for the two proposed Fermat counterexamples', () => {
  expect(0 ** 3 + 4 ** 3).toBe(4 ** 3)
  expect(0 > 0).toBe(false)
  expect(1 ** 3 + 2 ** 3).toBe(9)
  expect(2 ** 3).toBeLessThan(9)
  expect(3 ** 3).toBeGreaterThan(9)
  expect(Math.cbrt(9)).toBeGreaterThan(2)
  expect(Math.cbrt(9)).toBeLessThan(3)
  expect(Number.isInteger(Math.cbrt(9))).toBe(false)
  for (const phrase of ['positive integers x, y and z', 'n is an integer greater than 2', 'zero is not positive', 'not an integer', 'every condition']) expect(block(pyth, 6)).toContain(phrase)
 })
 it('retains all topic structures, one check per topic and revision without adding provenance notes', () => {
  for (const [text, count] of [[numbers, 7], [pyth, 6]] as const) {
   expect((text.match(/### Try it yourself/g) || []).length).toBe(count)
   expect((text.match(/\*\*Check:\*\*/g) || []).length).toBe(count)
   expect(text).toContain('## Your five-minute revision')
   expect(text).not.toMatch(/supporting evidence|YouTube transcript/i)
  }
 })
})
