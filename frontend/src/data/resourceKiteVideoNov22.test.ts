import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
import {youtubeVideoId} from '../components/blog/resourceVideo'
const p=resourceGuides.find(p=>p.id==='resource-maths-4')!.content
const topics=p.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
const video='https://www.youtube.com/watch?v=-IRXWg2oLY4'
describe('short compass-and-straightedge kite foundation',()=>{
 it('adds exactly one supported embed in the fourth quadrilateral topic',()=>{expect(topics).toHaveLength(4);expect(topics[3]).toContain('### Watch: Construct a kite with a compass and straightedge');expect(topics[3]).toContain("JoAnn's School");expect(topics[3]).toContain(`[Open the video](${video})`);expect(topics[3].match(/\[Open the video\]/g)).toHaveLength(1);expect(p.match(/-IRXWg2oLY4/g)).toHaveLength(1);expect(youtubeVideoId(video)).toBe('-IRXWg2oLY4');expect(topics[3]).toContain('### Understand it')})
 it('retains the two earlier videos in their original topic order and no diagonal video',()=>{const diagonal=p.includes('[Open the video](https://www.youtube.com/watch?v=PC9mbAwE3zk)');const ids=['y9udtA3cPgU',...(diagonal?['PC9mbAwE3zk']:[]),'5CeBlu260Rw','-IRXWg2oLY4'];expect([...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(ids.map(id=>`https://www.youtube.com/watch?v=${id}`));for(let i=0;i<ids.length;i++)expect(topics[(diagonal?[0,1,2,3]:[0,2,3])[i]]).toContain(`https://www.youtube.com/watch?v=${ids[i]}`);if(!diagonal)expect(topics[1]).not.toContain('[Open the video]')})
 it('limits the video to the kite rather than implying trapezium-angle coverage',()=>{for(const words of ['perpendicular bisector','vertices equally far from the two endpoints','kite construction, not trapezium angles'])expect(topics[3]).toContain(words);expect(topics[3]).not.toMatch(/all quadrilaterals have perpendicular diagonals|both diagonals always bisect|exactly one pair of parallel opposite sides/)})
 it('preserves all four attempts, checks, revision and direct practice',()=>{expect(p.match(/### Try it yourself/g)).toHaveLength(4);expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(p).toContain('## Your five-minute revision');expect(p).toContain('[Try an Evater sample paper](/papers/)');expect(p).not.toMatch(/supporting evidence|transcript|playback verified|amsi\.org|mathopenref|patreon\.com|pilo\.co/i)})
})
