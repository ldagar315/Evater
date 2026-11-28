import {describe,it,expect} from 'vitest'
import {angleAt,quadrilateralAngles} from './quadrilateralModel'
import {readFileSync} from 'node:fs'
describe('bounded convex quadrilateral angles',()=>{
 it('checks triangle and quadrilateral invariants for all81shapes',()=>{for(let l=40;l<=120;l+=10)for(let r=40;r<=120;r+=10){const m=quadrilateralAngles(l,r);expect(m.total).toBeCloseTo(360,10);expect(m.first.reduce((s,n)=>s+n,0)).toBeCloseTo(180,10);expect(m.second.reduce((s,n)=>s+n,0)).toBeCloseTo(180,10);expect(m.first[0]+m.second[0]).toBeCloseTo(m.angles[0],10);expect(m.first[2]+m.second[1]).toBeCloseTo(m.angles[2],10);for(let i=0;i<4;i++){expect(m.angles[i]).toBeGreaterThan(0);expect(m.angles[i]).toBeLessThan(180);const [a,b,c]=[m.points[i],m.points[(i+1)%4],m.points[(i+2)%4]];expect((b.x-a.x)*(c.y-b.y)-(b.y-a.y)*(c.x-b.x)).toBeLessThan(0)}}})
 it('computes known right and straight angles',()=>{expect(angleAt({x:1,y:0},{x:0,y:0},{x:0,y:1})).toBe(90);expect(angleAt({x:-1,y:0},{x:0,y:0},{x:1,y:0})).toBe(180)})
 it('has equal pairs in a symmetric shape',()=>{const m=quadrilateralAngles(80,80);expect(m.angles[0]).toBeCloseTo(m.angles[1],10);expect(m.angles[2]).toBeCloseTo(m.angles[3],10)})
 it('rejects values outside the guarded domain',()=>{for(const n of [30,130,41,45,NaN,Infinity,-1]){expect(()=>quadrilateralAngles(n,70)).toThrow(RangeError);expect(()=>quadrilateralAngles(70,n)).toThrow(RangeError)}})
 it('independently checks both predictions',()=>{expect(360-75-110-95).toBe(80);expect(360-3*90).toBe(90)})
 it('mounts one matching lesson and explains family/rounding limits',()=>{const s=readFileSync(new URL('../../pages/BlogPostPage.tsx',import.meta.url),'utf8');expect(s.match(/<QuadrilateralAngles \/>/g)).toHaveLength(1);expect(s).toContain("post.slug === 'class-8-maths-quadrilaterals-study-resources'");const c=readFileSync(new URL('./QuadrilateralAngles.tsx',import.meta.url),'utf8');expect(c).toContain('It does not show every quadrilateral');expect(c).toContain('rounded to one decimal place');expect(c).toContain('useId()')})
})
