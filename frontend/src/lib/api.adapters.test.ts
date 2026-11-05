import { describe, it, expect } from 'vitest'
import {
  wireDifficultyToStore,
  storeDifficultyToWire,
  type WireDifficulty,
  type UiDifficulty,
} from './api.adapters'

describe('wireDifficultyToStore (preserves CreateTestPage convertedQuestions)', () => {
  it('maps every backend Literal value to lowercase store casing', () => {
    expect(wireDifficultyToStore('Easy')).toBe('easy')
    expect(wireDifficultyToStore('Medium')).toBe('medium')
    expect(wireDifficultyToStore('Hard')).toBe('hard')
  })
  it('passes unknown values through lowercased (blind cast, no fallback)', () => {
    // CreateTestPage currently casts unchecked; the adapter must not change that.
    expect(wireDifficultyToStore('Extreme')).toBe('extreme')
    expect(wireDifficultyToStore('')).toBe('')
  })
})

describe('storeDifficultyToWire (preserves ViewTestPage capitalizeDifficulty)', () => {
  it('maps every store value to backend Literal casing', () => {
    const cases: [UiDifficulty, WireDifficulty][] = [
      ['easy', 'Easy'],
      ['medium', 'Medium'],
      ['hard', 'Hard'],
    ]
    for (const [store, wire] of cases) expect(storeDifficultyToWire(store)).toBe(wire)
  })
  it('is case-insensitive like the original (map keyed on toLowerCase)', () => {
    expect(storeDifficultyToWire('HARD')).toBe('Hard')
    expect(storeDifficultyToWire('Easy')).toBe('Easy')
  })
  it('falls back to Medium for unknown values, as ViewTestPage does today', () => {
    expect(storeDifficultyToWire('extreme')).toBe('Medium')
    expect(storeDifficultyToWire('')).toBe('Medium')
  })
})

describe('round-trip for known values', () => {
  it('store -> wire -> store is identity', () => {
    for (const d of ['easy', 'medium', 'hard'] as UiDifficulty[]) {
      expect(wireDifficultyToStore(storeDifficultyToWire(d))).toBe(d)
    }
  })
})
