import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const p=resourceGuides.find(p=>p.id==='resource-science-6')!.content
const topics=p.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
const video='https://www.youtube.com/watch?v=xKubdY2mHXc'
describe('short tropical-cyclone foundation video',()=>{
 it('adds exactly one cyclone embed in the fourth topic',()=>{expect(topics).toHaveLength(4);expect(topics[3]).toContain('### Watch: How a tropical cyclone grows');expect(topics[3]).toContain('SciShow Kids');expect(topics[3]).toContain(`[Open the video](${video})`);expect(topics[3].match(/\[Open the video\]/g)).toHaveLength(1);expect(p.match(/xKubdY2mHXc/g)).toHaveLength(1);expect(topics[3]).toContain('### Understand it')})
 it('preserves the three earlier videos in their original topics and order',()=>{const ids=['sfnsuS0-o58','VDf00z8sMFw','yc2-363MIQs','xKubdY2mHXc'];expect([...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(ids.map(id=>`https://www.youtube.com/watch?v=${id}`));for(let i=0;i<ids.length;i++)expect(topics[i]).toContain(`https://www.youtube.com/watch?v=${ids[i]}`)})
 it('distinguishes regional hurricane language from Indian warning rules',()=>{for(const words of ['regional name for a strong tropical cyclone','North American season and hurricane wind threshold',"not India's warning rules"])expect(topics[3]).toContain(words);expect(topics[3]).not.toMatch(/cyclone season is May|119.*India|go outside to/);expect(topics[3]).toContain('calm eye')})
 it('keeps preparation before danger and adult-led official safety decisions',()=>{for(const words of ['responsible adult before dangerous conditions arrive','Never go outside during a cyclone or an eye lull','local official warnings',"shelter's all-clear"])expect(topics[3]).toContain(words);expect(topics[3]).not.toMatch(/collect.*outside|bring.*outside/)})
 it('keeps four attempts, checks and the revision/paper path without public research detours',()=>{expect(p.match(/### Try it yourself/g)).toHaveLength(4);expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(p).toContain('## Your five-minute revision');expect(p).toContain('[Try an Evater sample paper](/papers/)');expect(p).not.toMatch(/supporting evidence|transcript|playback verified|nerdfighteria|learningonlineblog|dcmp\.org|ndma\.gov/i)})
})
