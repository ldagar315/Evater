import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
import {youtubeVideoId} from '../components/blog/resourceVideo'
const text=resourceGuides.find(p=>p.id==='resource-maths-5')!.content
const topics=text.split(/^## /m).filter(t=>/^\d+\./.test(t)),target=topics[3]
describe('digits place-value foundation preserves all puzzle reasoning',()=>{
 it('puts the exact primer only in topic4',()=>{expect(topics).toHaveLength(7);expect(target).toContain('[Open the video](https://www.youtube.com/watch?v=_dT97MUYuzs)');expect(youtubeVideoId('https://www.youtube.com/watch?v=_dT97MUYuzs')).toBe('_dT97MUYuzs');expect(target).toContain('Khan Academy India - English');expect(target.match(/\[Open the video\]/g)).toHaveLength(1)})
 it('bounds primer scope and preserves place-value and digit constraints',()=>{for(const w of ['832 = 800 + 30 + 2','704 has zero tens','AB means 10A + B','not A multiplied by B','place-value primer, not a solution','finding one assignment does not prove','same letter keeps the same digit','different letters use different digits','cannot be zero'])expect(target).toContain(w)})
 it('retains nonunique reasoning and exhaustively checks the stated even ordered puzzle',()=>{const answers=[];for(let A=1;A<=9;A++)for(let B=1;B<=9;B++)if(A!==B&&A%2===0&&A<B&&(10*A+B)+(10*B+A)===121)answers.push([A,B]);expect(answers).toEqual([[2,9],[4,7]]);if(target.includes('AB + BA = 121'))expect(target).toContain('There are two assignments');else expect(target).toContain('27 + 72 = 99');expect(target).toContain('### Understand it');expect(target).toContain('### Try it yourself');expect(target).toContain('**Check:**')})
 it('keeps seven questions and revision with no public verification detours',()=>{expect(text.match(/### Try it yourself/g)).toHaveLength(7);expect(text.match(/\*\*Check:\*\*/g)).toHaveLength(7);expect(text).toContain('## Your five-minute revision');expect(text).toContain('[Try an Evater sample paper](/papers/)');expect(text).not.toMatch(/transcript|playback verified|supporting evidence|ncert\.nic/i)})
})
