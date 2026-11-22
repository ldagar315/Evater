import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const text=resourceGuides.find(p=>p.id==='resource-science-12')!.content
const topics=text.split(/^## /m).filter(t=>/^\d+\./.test(t))
describe('soil nutrient-recycling foundation video',()=>{
 it('places one exact publisher video only in the decomposition topic',()=>{const s=topics[2];expect(s).toContain('### Watch: How dirt works');expect(s).toContain('The Nature Conservancy');expect(s).toContain('[Open the video](https://www.youtube.com/watch?v=K7YpPgbmSew)');expect(s.match(/\[Open the video\]/g)).toHaveLength(1);expect(text.match(/K7YpPgbmSew/g)).toHaveLength(1);expect(s).toContain('### Understand it')})
 it('preserves the required first and fourth videos and the independently optional food-web video',()=>{const known=[['P1X-WpfUvm4',1,true],['2lqhJNgn_Wg',2,false],['K7YpPgbmSew',3,true],['GK_vRtHJZu4',4,true]] as const;const selected=known.filter(([id,,required])=>required||text.includes(id));expect([...text.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(selected.map(([id])=>`https://www.youtube.com/watch?v=${id}`));for(const[id,n]of selected)expect(topics[n-1]).toContain(id)})
 it('retains all four original teaching and check sections, revision and sample-paper route',()=>{expect(topics).toHaveLength(4);expect(text.match(/### Try it yourself/g)).toHaveLength(4);expect(text.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(text).toContain('## Your five-minute revision');expect(text).toContain('[Try an Evater sample paper](/papers/)')})
 it('keeps matter cycling distinct from energy flow in both source versions',()=>{expect(topics[2]).toMatch(/[Nn]utrients can/);expect(topics[2]).toMatch(/[Ee]nergy flows/);expect(topics[2]).toMatch(/dissipated|heat/);expect(topics[2]).not.toContain('energy is endlessly recycled')})
 it('does not add research, transcript or off-platform reading instructions',()=>{expect(text).not.toMatch(/supporting evidence|transcript|playback verified|nature\.org|emediava\.org|eduye\.com/i)})
})
