import { describe, expect, it } from 'vitest'
import { meanBalance } from './meanModel'
describe('mean as balance and fair share', () => {
 it('keeps four values and preserves the total when shared equally throughout the domain', () => {
  for (let last=0; last<=20; last++) {
   const m=meanBalance(last)
   expect(m.values).toEqual([4,6,8,last])
   expect(m.count).toBe(4)
   expect(m.total).toBe(18+last)
   expect(m.mean*4).toBe(m.total)
   expect(m.below).toBe(m.above)
   expect(m.mean).toBeGreaterThanOrEqual(Math.min(...m.values))
   expect(m.mean).toBeLessThanOrEqual(Math.max(...m.values))
   expect(m.mean-7).toBe(m.change)
   if(last<20) expect(meanBalance(last+1).mean-m.mean).toBe(.25)
  }
 })
 it('has exact fractional means and no added-value denominator change',()=>{
  expect(meanBalance(10)).toMatchObject({total:28,mean:7,below:4,above:4,change:0})
  expect(meanBalance(11)).toMatchObject({total:29,mean:7.25,change:.25})
  expect(meanBalance(0)).toMatchObject({total:18,mean:4.5})
  expect(meanBalance(20)).toMatchObject({total:38,mean:9.5})
 })
 it('does not confuse the mean with the middle of the extremes or an observed value',()=>{
  const m=meanBalance(20)
  expect(m.mean).not.toBe((4+20)/2)
  expect(m.values).not.toContain(m.mean)
 })
 it('rejects unsupported inputs',()=>{
  for(const n of [-1,21,.25,NaN,Infinity,-Infinity]) expect(()=>meanBalance(n)).toThrow(RangeError)
 })
 it('checks the independent prediction with a different collection',()=>{
  expect((3+5+7+9)/4).toBe(6)
  expect((3+5+7+13)/4).toBe(7)
  expect((13-9)/4).toBe(1)
 })
})
