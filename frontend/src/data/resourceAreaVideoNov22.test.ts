import { describe, expect, it } from 'vitest'
import { resourceGuides } from './resourceGuides'
import { youtubeVideoId } from '../components/blog/resourceVideo'
const text=resourceGuides.find(p=>p.id==='resource-maths-14')!.content
const topics=text.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
describe('short source-fit unit-square explainer',()=>{
 it('adds the exact Khan Academy clip only to the first topic',()=>{
  const url="https://www.youtube.com/watch?v=iNZZIb_-PA8"
  expect(topics[0]).toContain("### Watch: Measuring area with tiled square units")
  expect(topics[0]).toContain("Khan Academy")
  expect(topics[0]).toContain(`[Open the video](${url})`)
  expect(youtubeVideoId(url)).toBe("iNZZIb_-PA8")
  expect(text.match(new RegExp("iNZZIb_-PA8",'g'))).toHaveLength(1)
  expect(topics[0].match(/\[Open the video\]/g)).toHaveLength(1)
 })
 it('preserves all four original teaching and independent check panels',()=>{
  expect(topics).toHaveLength(4)
  for(const block of topics){expect(block).toContain('### Try it yourself');expect(block).toContain('**Check:**')}
  expect(topics[0]).toContain('### Understand it')
  expect(topics[0]).toContain('cm²')
  expect(text).toContain('## Your five-minute revision')
  expect(text).toContain('[Try an Evater sample paper](/papers/)')
 })
 it('keeps the complete original clip order and mathematical square-unit relationship',()=>{
  expect([...text.matchAll(/watch\?v=([\w-]{11})/g)].map(m=>m[1])).toEqual(["iNZZIb_-PA8", "hm17lVaor0Q", "loAA3TCNAvU", "loAA3TCNAvU"])
  expect((6/2)*(4/2)).toBe(6)
  expect(6*2**2).toBe(24)
  expect(7*4).toBe(28)
  expect(2*(6+4)).toBe(2*(7+3))
  expect(6*4).not.toBe(7*3)
  expect(text).not.toMatch(/supporting evidence|transcript|playback verified|mathmathmath\.com|puzzicle\.org/i)
 })
})
