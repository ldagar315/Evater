import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
const text=readFileSync(new URL('./resourceGuides/15-science-9.md',import.meta.url),'utf8')
describe('density transfer reasoning',()=>{
 it('preserves original check and adds units and average density',()=>{
  const block=text.split('## 4.')[1].split('## Your five-minute revision')[0]
  for(const part of ["Use supplied values for a fictional object: mass 0.24 kg and outer volume 300 cm\u00b3. Convert mass first: 0.24 kg = 240 g. Its average density is 240 \u00f7 300 = 0.8 g/cm\u00b3, not 0.0008 g/cm\u00b3. A separate 30 g solid sample occupying 10 cm\u00b3 has density 3 g/cm\u00b3. The heavier object has the lower average density because its volume is much larger. For a hollow object, use its total mass and whole outer volume, not only the density of its wall material.", "In a simplified paper model, sealed rigid object A has mass 90 g and outer volume 150 cm\u00b3; solid object B has mass 20 g and volume 10 cm\u00b3. Water has density 1 g/cm\u00b3. Ignore surface tension and other support; assume neither object absorbs water, dissolves or changes shape. Calculate their average densities and predict which can float. Does the smaller mass decide?", "A has average density 90 \u00f7 150 = 0.6 g/cm\u00b3, below the stated water density, so it can float partly submerged in this model. B has density 20 \u00f7 10 = 2 g/cm\u00b3 and sinks despite being lighter. Compare mass per volume, not mass alone. A floating at rest has buoyancy balancing its weight, not zero gravity. Keep the stated assumptions; this is not an instruction to test unknown materials in water.", "**Check:** 3 g/cm\u00b3."])expect(block).toContain(part)
 })
 it('retains safe paper modelling and revision',()=>{
  expect(text).toContain('## Your five-minute revision')
  expect(text).toContain('not an instruction to test unknown materials')
  expect(text).not.toMatch(/https?:\/\/(?:ncert|openstax)/)
 })
})
