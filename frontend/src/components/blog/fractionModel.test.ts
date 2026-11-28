import { describe, it, expect } from 'vitest'
import { fractionShare, successiveChanges, PART_COUNTS, fractionNumber } from './fractionModel'
import { readFileSync } from 'node:fs'
describe('equal parts',()=>{
 it('converts independently checked fractions and complements',()=>{
  expect(fractionShare(7,20)).toMatchObject({decimal:0.35,percent:35,remaining:13,complementPercent:65})
  expect(fractionShare(11,40)).toMatchObject({decimal:0.275,percent:27.5,remaining:29,complementDecimal:0.725,complementPercent:72.5})
 })
 it('preserves a whole at every supported count including empty and full',()=>{
  for(const parts of PART_COUNTS)for(let n=0;n<=parts;n++){
   const f=fractionShare(n,parts);expect(f.shaded+f.remaining).toBe(parts);expect(f.percent+f.complementPercent).toBeCloseTo(100);expect(f.decimal+f.complementDecimal).toBeCloseTo(1)
  }
  expect(fractionShare(0,8).percent).toBe(0);expect(fractionShare(8,8).percent).toBe(100)
 })
 it('rejects invalid parts rather than silently changing a share',()=>{
  for(const [n,d] of [[NaN,20],[1,0],[1,3],[-1,4],[5,4],[1.5,4],[1,Infinity]])expect(()=>fractionShare(n,d)).toThrow(RangeError)
 })
})
describe('successive percentage changes',()=>{
 it('uses a new base and does not cancel equal rise and fall',()=>{
  expect(successiveChanges(500,20,-20)).toMatchObject({first:600,final:480,firstChange:100,secondChange:-120,netChange:-20,netPercent:-4,scaleMax:600})
  expect(successiveChanges(400,25,-20)).toMatchObject({first:500,final:400,netPercent:0})
  expect(successiveChanges(100,-20,20)).toMatchObject({first:80,final:96,firstChange:-20,secondChange:16,netPercent:-4})
 })
 it('covers zero, double increases, total decrease and reset boundaries',()=>{
  expect(successiveChanges(100,0,0)).toMatchObject({first:100,final:100,netPercent:0})
  expect(successiveChanges(1000,100,100)).toMatchObject({first:2000,final:4000,netPercent:300,scaleMax:4000})
  expect(successiveChanges(500,-100,100)).toMatchObject({first:0,final:0,netPercent:-100,scaleMax:500})
  expect(successiveChanges(500,100,-100)).toMatchObject({first:1000,final:0,netPercent:-100})
  expect(successiveChanges(150,5,-5)).toMatchObject({first:157.5,final:149.625,netPercent:-0.25})
 })
 it('bars can use a positive common maximum for every allowed rate pair',()=>{
  for(let a=-100;a<=100;a+=5)for(let b=-100;b<=100;b+=5){const m=successiveChanges(500,a,b);expect(m.scaleMax).toBeGreaterThan(0);for(const v of [m.original,m.first,m.final]){expect(v/m.scaleMax).toBeGreaterThanOrEqual(0);expect(v/m.scaleMax).toBeLessThanOrEqual(1)}expect(m.netChange).toBeCloseTo(m.firstChange+m.secondChange)}
 })
 it('rejects malformed parameters',()=>{
  for(const x of [[0,20,20],[1001,0,0],[NaN,0,0],[100,Infinity,0],[100,-101,0],[100,0,101],[100,1.5,0]])expect(()=>successiveChanges(...x as [number,number,number])).toThrow(RangeError)
 })
 it('prints supported exact terminating values without false rounding',()=>{
  expect(fractionNumber(149.625)).toBe('149.625');expect(fractionNumber(0.275)).toBe('0.275');expect(fractionNumber(-4)).toBe('-4')
 })
 it('mounts only on the matching resource lesson with no lesson-source changes',()=>{
  const s=readFileSync(new URL('../../pages/BlogPostPage.tsx',import.meta.url),'utf8');expect(s).toContain("post.slug === 'class-8-maths-fractions-in-disguise-study-resources' && <FractionExplorations />")
 })
})
