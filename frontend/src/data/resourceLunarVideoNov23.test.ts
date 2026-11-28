import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
import {youtubeVideoId} from '../components/blog/resourceVideo'
const text=resourceGuides.find(p=>p.id==='resource-science-11')!.content
const topics=text.split(/^## /m).filter(t=>/^\d+\./.test(t)),target=topics[2]
describe('lunar solar overview with preserved rounded drift reasoning',()=>{
 it('places one exact overview in topic3',()=>{expect(topics).toHaveLength(4);expect(target).toContain('[Open the video](https://www.youtube.com/watch?v=-T97d6dsfhE)');expect(youtubeVideoId('https://www.youtube.com/watch?v=-T97d6dsfhE')).toBe('-T97d6dsfhE');expect(target).toContain('McMaster Alumni');expect(target.match(/\[Open the video\]/g)).toHaveLength(1)})
 it('distinguishes calendar types without importing universal festival rules',()=>{for(const w of ['Sun, one based on Moon phases','hybrid luni-solar','does not give the arithmetic','adjustment rules for every calendar','does not by itself tell you','do not use this model to predict a festival date'])expect(target).toContain(w);expect(target).toContain('### Understand it');expect(target).toContain('### Try it yourself');expect(target).toContain('**Check:**')})
 it('checks rounded24month arithmetic without treating it as actual calendar dates',()=>{expect(12*29.5).toBe(354);expect(24*29.5).toBe(708);expect(2*365-24*29.5).toBe(22);expect(2*365-(24*29.5+1)).toBe(21);if(target.includes('24 lunar months')){for(const w of ['not an exact calendar-date prediction','21 days','rules must be checked separately'])expect(target).toContain(w)}})
 it('retains all four questions/checks/revision and prior phases lesson',()=>{expect(text.match(/### Try it yourself/g)).toHaveLength(4);expect(text.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(topics[1]).toContain('Jip3BbZBpsM');expect(text).toContain('## Your five-minute revision');expect(text).toContain('[Try an Evater sample paper](/papers/)');expect(text).not.toMatch(/transcript|playback verified|supporting evidence|aa\.usno/i)})
})
