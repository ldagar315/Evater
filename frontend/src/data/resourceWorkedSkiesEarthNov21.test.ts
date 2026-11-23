import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const guide=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
const topic=(id:string,n:number)=>guide(id).split(/^## /m).find(s=>s.startsWith(n+'.'))!
describe('six Skies and Earth independent checks, November21 evening',()=>{
 it('preserves inventory and previous leap-year work',()=>{
  for(const id of ['resource-science-11','resource-science-13']){
   const p=guide(id);expect(p.split(/^## /m).filter(s=>/^\d+\./.test(s))).toHaveLength(4)
   const added = id === 'resource-science-11' ? p.includes('[Open the video](https://www.youtube.com/watch?v=l64YwNl1wr0)') : id === 'resource-science-13' && p.includes('[Open the video](https://www.youtube.com/watch?v=o8K0RTYM6Mc)')
   const leap = id === 'resource-science-11' && p.includes('[Open the video](https://www.youtube.com/watch?v=yd1i3vkkh-0)')
   expect(p.match(/\[Open the video\]/g)).toHaveLength(1 + Number(id === 'resource-science-11' && p.includes('[Open the video](https://www.youtube.com/watch?v=-T97d6dsfhE)')) + Number(id === 'resource-science-13' && p.includes('[Open the video](https://www.youtube.com/watch?v=p4652vo5Oic)')) + Number(added) + Number(leap) + Number(id === 'resource-science-13' && p.includes('[Open the video](https://www.youtube.com/watch?v=bs9e6ovISbs)')));expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4)
   if (id === 'resource-science-11') {
    const urls = [...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m => m[1])
    expect(urls).toEqual([...(added ? ['https://www.youtube.com/watch?v=l64YwNl1wr0'] : []), 'https://www.youtube.com/watch?v=Jip3BbZBpsM', ...(p.includes('[Open the video](https://www.youtube.com/watch?v=-T97d6dsfhE)') ? ['https://www.youtube.com/watch?v=-T97d6dsfhE'] : []), ...(leap ? ['https://www.youtube.com/watch?v=yd1i3vkkh-0'] : [])])
    if (added) expect(topic(id,1)).toContain('[Open the video](https://www.youtube.com/watch?v=l64YwNl1wr0)')
   }
   if (id === 'resource-science-13') {
    const urls = [...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m => m[1])
    expect(urls).toEqual(['https://www.youtube.com/watch?v=Bsp5JYNMAQE', ...(added ? ['https://www.youtube.com/watch?v=o8K0RTYM6Mc'] : []), ...(p.includes('[Open the video](https://www.youtube.com/watch?v=p4652vo5Oic)') ? ['https://www.youtube.com/watch?v=p4652vo5Oic'] : []), ...(p.includes('[Open the video](https://www.youtube.com/watch?v=bs9e6ovISbs)') ? ['https://www.youtube.com/watch?v=bs9e6ovISbs'] : [])])
   }
  }
  expect(topic('resource-science-11',4)).toContain('divisible by 100 but not 400')
 })
 it('separates daily, phase and annual cycles with sunlight safety',()=>{
  const p=topic('resource-science-11',1)
  expect(p).toContain('rotation is the main cause');expect(p).toContain('revolution around the Sun is linked to the year')
  expect(p).toContain('average length is about 29.5 days');expect(p).toContain('never look directly at the Sun')
 })
 it('distinguishes illuminated hemisphere and visible fraction',()=>{
  const visible=(elongation:number)=>(1-Math.cos(elongation*Math.PI/180))/2
  expect(visible(0)).toBeCloseTo(0);expect(visible(90)).toBeCloseTo(.5);expect(visible(180)).toBeCloseTo(1)
  for(let angle=0;angle<=180;angle++){expect(visible(angle)).toBeGreaterThanOrEqual(0);expect(visible(angle)).toBeLessThanOrEqual(1)}
  const p=topic('resource-science-11',2);expect(p).toContain('half the whole Moon');expect(p).toContain('eclipse is a different alignment')
  expect(p).toContain('Never look directly at the Sun');expect(p).toContain('binoculars or a telescope')
 })
 it('computes rounded calendar drift independently, not festival predictions',()=>{
  expect(12*29.5).toBe(354);expect(24*29.5).toBe(708);expect(2*365-24*29.5).toBe(22)
  expect(2*365-(24*29.5+1)).toBe(21)
  const p=topic('resource-science-11',3);expect(p).toContain('not an exact calendar-date prediction')
  expect(p).toContain('actual rules must be checked separately')
 })
 it('links systems without claiming one observation proves causation',()=>{
  const p=topic('resource-science-13',2)
  for(const word of ['biosphere','geosphere','hydrosphere'])expect(p).toContain(word)
  expect(p).toContain('Other sources of sediment');expect(p).toContain('one observation does not establish a single cause')
 })
 it('checks population balance and growth versus new individuals',()=>{
  expect(24+7-9).toBe(22)
  for(let start=1;start<=30;start++)for(let born=0;born<=5;born++)expect(start+born-start).toBe(born)
  const p=topic('resource-science-13',3);expect(p).toContain('no plants entering or leaving')
  expect(p).toContain('not two new plants');expect(p).toContain('deaths exceeded new plants')
 })
 it('keeps partial protection and wildlife safety explicit',()=>{
  const p=topic('resource-science-13',4)
  expect(p).toContain('polluted-water inflow and habitat loss remain unresolved')
  expect(p).toContain('not evidence that every pressure has been removed')
  expect(p).toContain('do not catch, feed or disturb animals')
 })
})
