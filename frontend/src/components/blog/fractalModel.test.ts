import {describe,expect,it} from 'vitest'
import {sierpinskiStage,triangleArea} from './fractalModel'
describe('finite Sierpinski construction',()=>{
 it.each([0,1,2,3,4,5])('stage %i has exact count, scale, area and contained triangles',n=>{
  const m=sierpinskiStage(n),base=triangleArea(sierpinskiStage(0).triangles[0])
  expect(m.count).toBe(3**n);expect(m.triangles).toHaveLength(m.count)
  expect(m.areaNumerator).toBe(3**n);expect(m.areaDenominator).toBe(4**n);expect(m.sideDenominator).toBe(2**n)
  expect(new Set(m.triangles.map(t=>JSON.stringify(t))).size).toBe(m.count)
  for(const t of m.triangles){expect(triangleArea(t)/base).toBeCloseTo(1/4**n,12);for(const [x,y] of t){expect(x).toBeGreaterThanOrEqual(16);expect(x).toBeLessThanOrEqual(304);expect(y).toBeGreaterThanOrEqual(14.58468371008169);expect(y).toBeLessThanOrEqual(264)}}
  expect(m.triangles.reduce((a,t)=>a+triangleArea(t),0)/base).toBeCloseTo(m.areaFraction,12)
  if(n){const prev=sierpinskiStage(n-1);expect(m.count/prev.count).toBe(3);expect(m.areaFraction/prev.areaFraction).toBe(.75)}
 })
 it.each([-1,6,.5,NaN,Infinity])('rejects unsupported stage %s',n=>expect(()=>sierpinskiStage(n)).toThrow(RangeError))
 it('removes every middle triangle and preserves three disjoint corner regions',()=>{
  for(let n=0;n<5;n++){
   const a=sierpinskiStage(n),b=sierpinskiStage(n+1)
   a.triangles.forEach((t,i)=>{const children=b.triangles.slice(i*3,i*3+3);children.forEach((c,j)=>expect(c[j]).toEqual(t[j]));expect(children.flat().every(p=>!p.some(v=>!Number.isFinite(v)))).toBe(true)})
  }
 })
 it('independent 64-square-centimetre transfer separates count and area',()=>{expect(sierpinskiStage(2).areaFraction*64).toBe(36);expect(sierpinskiStage(3).areaFraction*64).toBe(27);expect(sierpinskiStage(3).count).toBe(27)})
 it('does not share mutable state between calls',()=>{const a=sierpinskiStage(2);a.triangles.pop();expect(sierpinskiStage(2).triangles).toHaveLength(9)})
})
