import { describe, expect, it } from 'vitest'
import { resourceGuides } from './resourceGuides'
import { youtubeVideoId } from '../components/blog/resourceVideo'
const text=resourceGuides.find(p=>p.id==='resource-maths-6')!.content
const topics=text.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
describe('short source-fit binomial distribution explainer',()=>{
 it('places the exact Khan Academy clip only in the second topic',()=>{
  const url="https://www.youtube.com/watch?v=Xy8NKUoyy98"
  expect(topics[1]).toContain('### Watch: Multiplying binomials intro')
  expect(topics[1]).toContain('Khan Academy')
  expect(topics[1]).toContain(`[Open the video](${url})`)
  expect(youtubeVideoId(url)).toBe("Xy8NKUoyy98")
  expect([...text.matchAll(/watch\?v=([\w-]{11})/g)].map(m=>m[1])).toEqual([...["ZJA1qz4XovY", "Xy8NKUoyy98", "4GcNzvILqtM"],...(text.includes('[Open the video](https://www.youtube.com/watch?v=3NHSwiv_pSE)')?["3NHSwiv_pSE"]:[])])
  expect(topics[1].match(/\[Open the video\]/g)).toHaveLength(1)
 })
 it('preserves four separate teaching, practice and revision panels',()=>{
  expect(topics).toHaveLength(4)
  for(const block of topics){expect(block).toContain('### Try it yourself');expect(block).toContain('**Check:**');expect(block.match(/\[Open the video\]/g)?.length??0).toBeLessThanOrEqual(1)}
  expect(topics[1]).toContain('### Understand it')
  expect(text).toContain('## Your five-minute revision')
  expect(text).toContain('[Try an Evater sample paper](/papers/)')
  expect(text).not.toMatch(/supporting evidence|transcript|playback verified|puzzicle\.org|kastatic\.org/i)
 })
 it('checks signed distribution independently over positive, negative and fractional inputs',()=>{
  for(let k=-40;k<=40;k++){
   const x=k/2
   expect((x-4)*(x+7)).toBeCloseTo(x*x+3*x-28,12)
   expect((2*x+3)*(x-4)).toBeCloseTo(2*x*x-5*x-12,12)
   expect((3*x-2)*(x+5)).toBeCloseTo(3*x*x+13*x-10,12)
  }
  expect([2*-4,3*-4]).toEqual([-8,-12])
  expect((3*-1-2)*(-1+5)).toBe(-20)
  expect((2*5+3)*(5-4)).toBe(13)
 })
})
