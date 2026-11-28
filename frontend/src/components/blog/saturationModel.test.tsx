import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { SaturationExplorer } from './SaturationExplorer'
import { saturationModel } from './saturationModel'

describe('fixed-temperature saturation explorer', () => {
  it('conserves every added unit across all 84 control pairs', () => {
    for(let solvent=1;solvent<=4;solvent++) for(let added=0;added<=20;added++) {
      const m=saturationModel(solvent,added)
      expect(m.limit).toBe(solvent*4)
      expect(m.dissolved+m.undissolved).toBe(added)
      expect(m.dissolved).toBeLessThanOrEqual(m.limit)
      expect(m.dissolved).toBeGreaterThanOrEqual(0)
      expect(m.undissolved).toBeGreaterThanOrEqual(0)
      expect(m.saturated).toBe(added>=m.limit)
      expect(m.room).toBe(Math.max(0,m.limit-added))
    }
  })
  it('distinguishes exact saturation without extra solid from excess solid',()=>{
    expect(saturationModel(2,8)).toMatchObject({saturated:true,undissolved:0,dissolved:8})
    expect(saturationModel(2,9)).toMatchObject({saturated:true,undissolved:1,dissolved:8})
    expect(saturationModel(2,7)).toMatchObject({saturated:false,undissolved:0,room:1})
  })
  it('increases total capacity, not solubility per solvent unit, on adding solvent',()=>{
    const first=saturationModel(3,15), second=saturationModel(4,15)
    expect(first.undissolved).toBe(3);expect(second.undissolved).toBe(0)
    expect(second.saturated).toBe(false);expect(second.room).toBe(1)
    expect(first.limit/3).toBe(second.limit/4)
  })
  it('rejects invalid controls rather than inventing or clamping data',()=>{
    for(const s of [0,5,1.5,NaN,Infinity])expect(()=>saturationModel(s,8)).toThrow(RangeError)
    for(const a of [-1,21,0.5,NaN,Infinity])expect(()=>saturationModel(2,a)).toThrow(RangeError)
  })
  it('starts with six conserved markers and labels accessible independent controls',()=>{
    const html=renderToStaticMarkup(<SaturationExplorer />)
    expect((html.match(/data-solute="dissolved"/g)||[]).length).toBe(6)
    expect(html).not.toContain('data-solute="undissolved"')
    expect((html.match(/type="range"/g)||[]).length).toBe(2)
    expect(html).toContain('not measured salt or sugar data')
    expect(html).toContain('not visible real particles')
    expect(html).toContain('not total solution volume')
    expect(html).toContain('aria-live="polite"')
    expect(html).toContain('Predict a different case')
    expect(html).toContain('Reset amounts')
    expect(html).toContain('no tasting or heating')
  })
})
