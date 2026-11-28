import {describe,it,expect} from 'vitest'
import {numberPyramid} from './pyramidModel'
describe('four-row adjacent-addition pyramid',()=>{
 it('checks every one of 14641 supported bottom rows and the coefficient rule',()=>{
  for(let a=0;a<=10;a++)for(let b=0;b<=10;b++)for(let c=0;c<=10;c++)for(let d=0;d<=10;d++){
   const m=numberPyramid([a,b,c,d]);expect(m.rows.map(r=>r.length)).toEqual([4,3,2,1]);expect(m.top).toBe(a+3*b+3*c+d);expect(m.contributions.reduce((s,n)=>s+n,0)).toBe(m.top)
   for(let r=1;r<4;r++)m.rows[r].forEach((n,i)=>expect(n).toBe(m.rows[r-1][i]+m.rows[r-1][i+1]))
  }
 })
 it('distinguishes middle influence from end influence',()=>{
  const start=[2,4,6,3],m=numberPyramid(start)
  expect(m.rows).toEqual([[2,4,6,3],[6,10,9],[16,19],[35]])
  start.forEach((_,i)=>expect(numberPyramid(start.map((n,j)=>n+(i===j?1:0))).top-m.top).toBe([1,3,3,1][i]))
 })
 it('does not equate equal bottom totals with equal tops',()=>{
  const a=numberPyramid([2,4,6,3]),b=numberPyramid([4,2,6,3]);expect(a.bottomTotal).toBe(b.bottomTotal);expect(a.top).toBe(35);expect(b.top).toBe(31)
 })
 it('covers zero and maximum bounds, and does not mutate the input',()=>{
  expect(numberPyramid([0,0,0,0]).top).toBe(0);expect(numberPyramid([10,10,10,10]).top).toBe(80)
  const b=Object.freeze([1,2,3,4]);numberPyramid(b);expect(b).toEqual([1,2,3,4])
 })
 it('rejects unsupported values and row sizes',()=>{
  for(const b of [[],[1,2,3],[1,2,3,4,5],[-1,2,3,4],[11,2,3,4],[.5,2,3,4],[NaN,2,3,4],[Infinity,2,3,4]])expect(()=>numberPyramid(b)).toThrow(RangeError)
 })
 it('checks an independent transfer',()=>{
  expect(numberPyramid([1,3,5,7]).top).toBe(32);expect(numberPyramid([1,5,5,7]).top).toBe(38);expect(numberPyramid([3,3,5,7]).top).toBe(34)
 })
})
