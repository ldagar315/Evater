import { describe, it, expect } from 'vitest'
import { contactPressure } from './pressureModel'
import { readFileSync } from 'node:fs'
describe('perpendicular contact pressure',()=>{
 it('starts with an exact reference',()=>expect(contactPressure(60,3)).toMatchObject({area:.3,numerator:200,denominator:1,pascals:200,baselineRatio:1}))
 it('checks all100supported pairs and exact fraction representations',()=>{for(let f=20;f<=200;f+=20)for(let a=1;a<=10;a++){const m=contactPressure(f,a);expect(m.pascals).toBeCloseTo(f/(a/10),10);expect(m.numerator/m.denominator).toBeCloseTo(m.pascals,10);expect(m.baselineRatio).toBeCloseTo(m.pascals/200,10)}})
 it('halves pressure when area doubles at constant force',()=>{for(let f=20;f<=200;f+=20)for(let a=1;a<=5;a++)expect(contactPressure(f,a*2).pascals).toBe(contactPressure(f,a).pascals/2)})
 it('preserves pressure when both force and area double',()=>{for(let f=20;f<=100;f+=20)for(let a=1;a<=5;a++)expect(contactPressure(f*2,a*2).pascals).toBe(contactPressure(f,a).pascals)})
 it('does not round nonterminating pressures into exact equalities',()=>expect(contactPressure(20,3)).toMatchObject({numerator:200,denominator:3}))
 it('covers extremes and rejects invalid units/steps',()=>{expect(contactPressure(20,10).pascals).toBe(20);expect(contactPressure(200,1).pascals).toBe(2000);for(const n of [0,-20,30,220,Infinity,NaN])expect(()=>contactPressure(n,3)).toThrow(RangeError);for(const n of [0,-1,.3,1.5,11,Infinity,NaN])expect(()=>contactPressure(60,n)).toThrow(RangeError)})
 it('independently checks the transfer, including smaller force at higher pressure',()=>{expect(contactPressure(80,2).pascals).toBe(400);expect(contactPressure(120,6).pascals).toBe(200)})
 it('mounts only the matching lesson and states schematic limits',()=>{const s=readFileSync(new URL('../../pages/BlogPostPage.tsx',import.meta.url),'utf8');expect(s.match(/<ContactPressure \/>/g)).toHaveLength(1);expect(s).toContain("post.slug === 'class-8-science-pressure-winds-storms-and-cyclones-study-resources'");const c=readFileSync(new URL('./ContactPressure.tsx',import.meta.url),'utf8');expect(c).toContain('No material deformation or weather is simulated.');expect(c).toContain('useId()')})
})
