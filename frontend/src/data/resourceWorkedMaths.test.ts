import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const guide=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
const topic=(id:string,n:number)=>guide(id).split(/^## /m).find(s=>s.startsWith(n+'.'))!
describe('six worked Maths topics, November19',()=>{
 it('keeps the six selected topics and independent checks without replacing existing videos',()=>{
  const selected:[string,number][]=[['resource-maths-4',2],['resource-maths-4',4],['resource-maths-6',2],['resource-maths-6',4],['resource-maths-8',4],['resource-maths-14',1]]
  for(const [id,n]of selected){const t=topic(id,n);expect(t).toBeTruthy();expect(t).toContain('### Try it yourself');expect(t).toContain('**Check:**')}
  for(const [id,count]of [['resource-maths-4',2],['resource-maths-6',2],['resource-maths-8',3],['resource-maths-14',3]] as const)expect((guide(id).match(/\[Open the video\]/g)||[]).length).toBe(count)
 })
 it('checks unequal bisected diagonals and exact angle conditions',()=>{
  const x=4,pm=3*x-1,mr=x+7,qm=5
  expect(pm).toBe(mr);expect(2*pm).toBe(22);expect(2*qm).toBe(10);expect(2*pm).not.toBe(2*qm)
  expect(180-68).toBe(112);expect(180-74).toBe(106);expect(180-101).toBe(79)
  expect(74+101+106+79).toBe(360)
  expect(topic('resource-maths-4',4)).toContain('at least one pair of parallel opposite sides')
  expect(topic('resource-maths-4',4)).toContain('does not mean BD also bisects AC')
 })
 it('checks the kite construction with different radii, including a counterexample to both diagonals bisecting',()=>{
  // B=(-4,0),D=(4,0),A=(0,3),C=(0,-sqrt(33)).
  const a=[0,3],b=[-4,0],c=[0,-Math.sqrt(33)],d=[4,0]
  const dist=(p:number[],q:number[])=>Math.hypot(p[0]-q[0],p[1]-q[1])
  expect(dist(a,b)).toBeCloseTo(5);expect(dist(a,d)).toBeCloseTo(5)
  expect(dist(c,b)).toBeCloseTo(7);expect(dist(c,d)).toBeCloseTo(7)
  expect((a[1]+c[1])/2).not.toBe(0)
 })
 it('checks both four-product expansions for integers and fractions',()=>{
  for(const x of [-9,-3,-1,0,0.5,1,2,5,8]){
   expect((2*x+3)*(x-4)).toBeCloseTo(2*x*x-5*x-12)
   expect((3*x-2)*(x+5)).toBeCloseTo(3*x*x+13*x-10)
  }
  expect((2*5+3)*(5-4)).toBe(13);expect((3*-1-2)*(-1+5)).toBe(-20)
  const t=topic('resource-maths-6',2);expect(t).toContain('- 2x × (-4) = -8x');expect(t).toContain('- 3 × (-4) = -12');expect(t).not.toContain('| ---')
 })
 it('checks the diagnostic counterexamples and the repaired expressions',()=>{
  for(let t=-20;t<=20;t++)expect(4*(2*t-3)-2*(t+5)).toBe(6*t-22)
  expect(4*(2*0-3)-2*(0+5)).not.toBe(6*0-2)
  for(let p=-20;p<=20;p++)expect(-3*(2*p-5)+4*p).toBe(-2*p+15)
  expect(-3*(2*2-5)+4*2).toBe(11);expect(-2*2-15).toBe(-19)
  expect((0+2)**2).toBe(0**2+4);expect((1+2)**2).not.toBe(1**2+4)
 })
 it('checks discount and profit using different explicit bases',()=>{
  const selling=1000*(1-0.1),profit=selling-720
  expect(selling).toBe(900);expect(profit).toBe(180);expect(profit/720*100).toBe(25)
  const sale=1000*(1-0.2),gain=sale-640
  expect(sale).toBe(800);expect(gain).toBe(160);expect(gain/640*100).toBe(25)
  expect(topic('resource-maths-8',4)).toContain('Profit margin on sales is a different measure')
 })
 it('checks tile unit areas and equal-perimeter unequal-area examples',()=>{
  expect((6/2)*(4/2)).toBe(6);expect(6*2**2).toBe(24)
  expect(2*(6+4)).toBe(2*(7+3));expect(6*4).not.toBe(7*3)
  expect((9/3)*(6/3)).toBe(6);expect(6*3**2).toBe(54)
  expect(2*(9+6)).toBe(2*(12+3));expect(9*6).not.toBe(12*3)
  expect(topic('resource-maths-14',1)).toContain('unless each tile is a one-centimetre square')
 })
})
