import { describe, expect, it } from 'vitest'
import { densityModel } from './densityModel'

describe('bounded density model', () => {
  it('uses whole outer volume and mass in compatible units', () => {
    expect(densityModel(240, 300, 1).density).toBe(0.8)
    expect(densityModel(240, 300, 1).state).toBe('floats')
    expect(densityModel(30, 10, 1).state).toBe('sinks')
  })
  it('handles neutral buoyancy without inventing a partial floating depth', () => {
    const m = densityModel(120, 100, 1.2)
    expect(m.state).toBe('suspended')
    expect(m.submergedFraction).toBe(1)
    expect(m.displacedMass).toBe(120)
  })
  it('independently checks the transfer question', () => {
    expect(densityModel(180, 300, 0.8).submergedFraction).toBe(0.75)
    expect(densityModel(180, 300, 1.2).submergedFraction).toBe(0.5)
    expect(densityModel(180, 300, 0.8).density).toBe(densityModel(180, 300, 1.2).density)
  })
  it('checks every slider combination against integer mass capacity', () => {
    for (let mass = 20; mass <= 600; mass += 10) for (let volume = 50; volume <= 500; volume += 10) for (const liquid of [0.8, 1, 1.2]) {
      const model = densityModel(mass, volume, liquid)
      const delta = mass * 10 - volume * Math.round(liquid * 10)
      expect(model.state).toBe(delta === 0 ? 'suspended' : delta < 0 ? 'floats' : 'sinks')
      expect(model.submergedFraction).toBeGreaterThan(0)
      expect(model.submergedFraction).toBeLessThanOrEqual(1)
      if (model.state === 'floats') expect(model.submergedFraction * volume * liquid).toBeCloseTo(mass, 9)
    }
  })
  it('does not turn invalid inputs into a plausible answer', () => {
    for (const bad of [0, -1, NaN, Infinity, -Infinity]) {
      expect(() => densityModel(bad, 100, 1)).toThrow(RangeError)
      expect(() => densityModel(100, bad, 1)).toThrow(RangeError)
      expect(() => densityModel(100, 100, bad)).toThrow(RangeError)
    }
  })
})
