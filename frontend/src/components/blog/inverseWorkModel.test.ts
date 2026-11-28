import {describe,it,expect} from 'vitest'
import {inverseWork} from './inverseWorkModel'
import {readFileSync} from 'node:fs'
describe('fixed-job inverse work model',()=>{
 it('starts with four workers and six days',()=>expect(inverseWork(4)).toMatchObject({days:6,workerDays:24,numerator:6,denominator:1}))
 it('conserves worker-days for every supported team',()=>{for(let w=1;w<=12;w++){const m=inverseWork(w);expect(w*m.days).toBeCloseTo(24,12);expect(m.numerator/m.denominator).toBe(m.days);expect(m.workInSixDays).toBe(w*6)}})
 it('halves time when workers double',()=>{for(let w=1;w<=6;w++)expect(inverseWork(w*2).days).toBe(inverseWork(w).days/2)})
 it('retains exact fractions and extremes',()=>{expect(inverseWork(7)).toMatchObject({numerator:24,denominator:7});expect(inverseWork(1).days).toBe(24);expect(inverseWork(12).days).toBe(2)})
 it('rejects invalid worker counts',()=>{for(const n of [0,-1,13,1.5,NaN,Infinity])expect(()=>inverseWork(n)).toThrow(RangeError)})
 it('independently checks the transfer',()=>{expect(5*8).toBe(40);expect(40/10).toBe(4);expect(10*16/40).toBe(4)})
 it('mounts only matching resource lesson with assumptions',()=>{const s=readFileSync(new URL('../../pages/BlogPostPage.tsx',import.meta.url),'utf8');expect(s.match(/<InverseWork \/>/g)).toHaveLength(1);expect(s).toContain("post.slug === 'class-8-maths-proportional-reasoning-2-study-resources'");const c=readFileSync(new URL('./InverseWork.tsx',import.meta.url),'utf8');expect(c).toContain('no waiting or coordination delays');expect(c).toContain('useId()')})
})
