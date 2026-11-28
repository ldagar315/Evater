import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
import {youtubeVideoId} from '../components/blog/resourceVideo'
const p=resourceGuides.find(p=>p.id==='resource-maths-6')!.content
const topics=p.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
const video="https://www.youtube.com/watch?v=3NHSwiv_pSE"
const intro=topics[3].split('### Understand it')[0]
const slots=[["ZJA1qz4XovY", 1, true], ["Xy8NKUoyy98", 2, false], ["4GcNzvILqtM", 3, true], ["3NHSwiv_pSE", 4, false]] as const
const link=(id:string)=>`[Open the video](https://www.youtube.com/watch?v=${id})`
describe('short signed expansion and error-repair foundation',()=>{
 it('adds one exact video only to the error-repair topic',()=>{expect(topics).toHaveLength(4);expect(topics[3]).toContain('### Watch: Keep signs when expanding brackets');expect(topics[3]).toContain('Khan Academy');expect(topics[3]).toContain(`[Open the video](${video})`);expect(topics[3].match(/\[Open the video\]/g)).toHaveLength(1);expect(p.match(new RegExp("3NHSwiv_pSE",'g'))).toHaveLength(1);expect(youtubeVideoId(video)).toBe("3NHSwiv_pSE")})
 it('retains exact earlier topics and the independently optional binomial placement',()=>{const selected=slots.filter(([id,n,required])=>required||n===4||p.includes(link(id)));expect([...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[0])).toEqual(selected.map(([id])=>link(id)));for(const[id,n]of selected)expect(topics[n-1]).toContain(link(id))})
 it('checks the video example over positive, negative and fractional inputs',()=>{for(let k=-40;k<=40;k++){const y=k/2;expect(7*(3*y-5)-2*(10+4*y)).toBe(13*y-55)}expect([-2*10,-2*4]).toEqual([-20,-8]);for(const words of ['signed multiplier','A negative multiplier acts on every term','before collecting like terms'])expect(intro).toContain(words)})
 it('keeps a numerical counterexample distinct from proof of an identity',()=>{const wrong=(u:number)=>u*u+4;expect((0+2)**2).toBe(wrong(0));expect((1+2)**2).not.toBe(wrong(1));expect(intro).toContain('counterexample can disprove an identity');expect(intro).toContain('agreement at one value does not prove it');expect(topics[3]).toContain('### Understand it')})
 it('preserves four original attempts/checks, revision and direct practice without research detours',()=>{expect(p.match(/### Try it yourself/g)).toHaveLength(4);expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(topics[3]).toContain('distributive');expect(p).toContain('## Your five-minute revision');expect(p).toContain('[Try an Evater sample paper](/papers/)');expect(p).not.toMatch(/supporting evidence|transcript|playback verified|ncert\.nic|edshare\.gcu/i)})
})
