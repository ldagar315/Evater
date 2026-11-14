import { describe, expect, it } from 'vitest'
import { cubeModel } from './cubeModel'

describe('unit cube volume model', () => {
  it('counts every side and layer exactly', () => {
    for (let side = 1; side <= 6; side++) {
      for (let layers = 1; layers <= side; layers++) {
        const model = cubeModel(side, layers, 45)
        expect(model.baseArea).toBe(side ** 2)
        expect(model.fullVolume).toBe(side ** 3)
        expect(model.shownVolume).toBe(side ** 2 * layers)
        expect(model.units).toHaveLength(side ** 2 * layers)
        expect(new Set(model.units.map((unit) => unit.key)).size).toBe(model.shownVolume)
      }
    }
  })
  it('keeps every rendered vertex within the viewbox across camera extremes', () => {
    for (let side = 1; side <= 6; side++) for (const angle of [20, 45, 70]) {
      const model = cubeModel(side, side, angle)
      for (const unit of model.units) for (const face of [unit.front, unit.right, unit.top]) {
        for (const pair of face.split(' ')) {
          const [x, y] = pair.split(',').map(Number)
          expect(x).toBeGreaterThan(0)
          expect(x).toBeLessThan(340)
          expect(y).toBeGreaterThan(0)
          expect(y).toBeLessThan(310)
        }
      }
    }
  })
  it('clamps layer and side inputs, and rotation never changes the volume', () => {
    expect(cubeModel(99, 99, 45).units).toHaveLength(216)
    expect(cubeModel(-1, -1, 45).units).toHaveLength(1)
    expect(cubeModel(4, 2, 20).shownVolume).toBe(32)
    expect(cubeModel(4, 2, 70).shownVolume).toBe(32)
  })
})
