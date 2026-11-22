import { describe, expect, it } from 'vitest'
import { resourceGuides } from './resourceGuides'
import { youtubeVideoId } from '../components/blog/resourceVideo'
const text=resourceGuides.find(p=>p.id==='resource-science-8')!.content
const first=text.split('## 1.')[1].split('\n## 2.')[0]
describe('source-fit separation explainer',()=>{
 it('puts the exact Cognito clip only beside the first topic',()=>{
  expect(first).toContain('### Watch: Filtration, Evaporation & Crystallisation')
  expect(first).toContain('Cognito')
  expect(first).toContain('[Open the video](https://www.youtube.com/watch?v=vi_SJBnxmHo)')
  expect(first.match(/\[Open the video\]/g)).toHaveLength(1)
  expect(text.match(/vi_SJBnxmHo/g)).toHaveLength(1)
  expect(youtubeVideoId('https://www.youtube.com/watch?v=vi_SJBnxmHo')).toBe('vi_SJBnxmHo')
 })
 it('retains the independent reasoning and practical safety conditions',()=>{
  for(const value of ['filter traps the insoluble sand','salt stays dissolved','clear filtrate is still a mixture','paper reasoning example',"teacher's approved setup and supervision",'If the only goal is to obtain the sand','filtrate is not pure water','not needed just to obtain sand'])expect(first).toContain(value)
  expect(first).toContain('### Understand it')
  expect(first).toContain('### Try it yourself')
  expect(first).toContain('**Check:**')
 })
 it('keeps four topics, the previous exact clip and the revision, without reading detours',()=>{
  expect(text.match(/^## \d+\./gm)).toHaveLength(4)
  expect(text.match(/### Try it yourself/g)).toHaveLength(4)
  expect(text.match(/\*\*Check:\*\*/g)).toHaveLength(4)
  expect(text.match(/Vt7lN4QPU0k/g)).toHaveLength(1)
  expect(text.match(/\[Open the video\]/g)).toHaveLength(2)
  expect(text).toContain('## Your five-minute revision')
  expect(text).not.toMatch(/supporting evidence|transcript|playback verified|pearson\.com|labxchange\.org/i)
 })
})
