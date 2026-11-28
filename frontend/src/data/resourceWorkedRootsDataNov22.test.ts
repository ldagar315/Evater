import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const guide=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
const topic=(id:string,n:number)=>guide(id).split(/^## /m).find(s=>s.startsWith(n+'.'))!
const isSquare=(n:number)=>Number.isInteger(Math.sqrt(n))
const isCube=(n:number)=>Math.round(Math.cbrt(n))**3===n
describe('independent roots and data transfer checks, Nov22',()=>{
 it('preserves four topics, videos and checks in each lesson',()=>{
  for(const id of ['resource-maths-1','resource-maths-12']){
   const p=guide(id)
   expect(p.split(/^## /m).filter(s=>/^\d+\./.test(s))).toHaveLength(4)
   expect(p.match(/\[Open the video\]/g)).toHaveLength(4);expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4)
  }
 })
 it('adds different consecutive borders rather than duplicating one',()=>{
  expect(2*24+1).toBe(49);expect(2*25+1).toBe(51)
  expect(26**2-24**2).toBe(100);expect(2*49).not.toBe(100)
  for(let n=1;n<=200;n++)expect((n+2)**2-n**2).toBe((2*n+1)+(2*(n+1)+1))
  expect(topic('resource-maths-1',1)).toContain('misses the two extra tiles')
 })
 it('checks integer and decimal root bounds and the principal-root convention',()=>{
  expect(9**2).toBeLessThan(92);expect(10**2).toBeGreaterThan(92)
  expect(9.5**2).toBe(90.25);expect(9.6**2).toBe(92.16)
  expect(Math.sqrt(49)).toBe(7);expect((-7)**2).toBe(49)
  const p=topic('resource-maths-1',2);expect(p).toContain('√49 = 7');expect(p).toContain('x = -7')
  expect(p).toContain('11 < √123 < 12, not 144')
 })
 it('compares cube edge scaling with volume scaling and units',()=>{
  expect(9**3).toBe(729);expect(18**3).toBe(8*729);expect(72**3).not.toBe(8*729)
  for(let edge=1;edge<=100;edge++)expect((edge*2)**3).toBe(8*edge**3)
  const p=topic('resource-maths-1',3);expect(p).toContain('second edge is 18 cm');expect(p).toContain('edge length in cm, not cm³')
 })
 it('proves the square and cube multipliers are minimal positive integers',()=>{
  expect(2**3*5**2).toBe(200);expect(isSquare(200*2)).toBe(true);expect(isCube(200*5)).toBe(true)
  for(let k=1;k<2;k++)expect(isSquare(200*k)).toBe(false)
  for(let k=1;k<5;k++)expect(isCube(200*k)).toBe(false)
  expect(topic('resource-maths-1',4)).toContain('smallest multipliers, 2 and 5')
 })
 it('recovers a missing value and updates both total and count for a mean',()=>{
  const known=[8,10,14,18],total=5*12,missing=total-known.reduce((a,b)=>a+b,0)
  expect(missing).toBe(10);expect((total+18)/6).toBe(13);expect((12+18)/2).toBe(15)
  expect(topic('resource-maths-12',1)).toContain('not equal weight per session')
 })
 it('decodes nonzero origins and different numerical scales',()=>{
  const a=[80+5*2,80+8*2],b=[9*10,9.6*10]
  expect(a).toEqual([90,96]);expect(b).toEqual(a);expect(a[1]-a[0]).toBe(6)
  expect(topic('resource-maths-12',2)).toContain('Neither increase is larger')
 })
 it('uses elapsed time for rates and retains sampled-data and flow caveats',()=>{
  expect((50-20)/5).toBe(6);expect((80-50)/2).toBe(15);expect(50-50).toBe(0)
  const p=topic('resource-maths-12',3);expect(p).toContain('do not rule out unrecorded changes')
  expect(p).toContain('equal inflow and outflow could balance');expect(p).toContain('not an empty tank')
 })
 it('marks fabricated data and does not turn association or display choice into causation',()=>{
  const p=topic('resource-maths-12',4)
  expect(p).toContain('made-up dataset');expect(p).toContain('plausible common factor')
  expect(p).toContain('not the evidence for a cause');expect(p).toContain('do not show that ice-cream sales cause hat wearing')
 })
})
