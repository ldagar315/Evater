import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
import {youtubeVideoId} from '../components/blog/resourceVideo'
const p=resourceGuides.find(p=>p.id==='resource-science-2')!.content
const topics=p.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
const video="https://www.youtube.com/watch?v=4HdDlpAB2hU"
const intro=topics[2].split('### Understand it')[0]
describe('short microorganism groups foundation',()=>{
 it('places exactly one supported video in the kinds topic',()=>{expect(topics).toHaveLength(4);expect(topics[2]).toContain('### Watch: Meet fungi, protists and bacteria');expect(topics[2]).toContain("EarthPen");expect(topics[2]).toContain(`[Open the video](${video})`);expect(topics[2].match(/\[Open the video\]/g)).toHaveLength(1);expect(p.match(new RegExp("4HdDlpAB2hU",'g'))).toHaveLength(1);expect(youtubeVideoId(video)).toBe("4HdDlpAB2hU")})
 it('retains the microscope, organisation and useful-food videos in their original topics',()=>{const ids=["tVcEEw6qbBQ", "EtWknf1gzKo", "4HdDlpAB2hU", "eksagPy5tmQ"];expect([...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(ids.map(id=>`https://www.youtube.com/watch?v=${id}`));for(let i=0;i<ids.length;i++)expect(topics[i]).toContain(`https://www.youtube.com/watch?v=${ids[i]}`)})
 it('separates groups, single-celled yeast and acellular viruses without unsafe culture tasks',()=>{for(const words of ['protozoa and some algae','Yeast is a single-celled fungus','do not mean every member is microscopic','Viruses are acellular','only inside living host cells','Do not culture, sniff or taste unknown microbes'])expect(intro).toContain(words);expect(intro).not.toMatch(/all fungi are multicellular|viruses are cells|all microbes cause disease/i);expect(topics[2]).toContain('### Understand it')})
 it('keeps the original teaching, attempts, checks, revision and direct practice',()=>{expect(p.match(/### Try it yourself/g)).toHaveLength(4);expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(p).toMatch(/not all harmful|Some microbes help produce food/);expect(p).toContain('teacher-prepared slides');expect(p).toContain('## Your five-minute revision');expect(p).toContain('[Try an Evater sample paper](/papers/)');expect(p).not.toMatch(/supporting evidence|transcript|playback verified|ncert\.nic|rhs\.org/i)})
})
