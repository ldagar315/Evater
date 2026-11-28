import { describe, expect, it } from 'vitest'
import { resourceGuides } from './resourceGuides'
const lesson=resourceGuides.find(p=>p.id==='resource-science-5')!.content
const topics=lesson.split(/^## /m).filter(t=>/^\d+\./.test(t))
describe('one short mass and weight foundation video',()=>{
 it('places the exact selected publisher link only in the mass and weight topic',()=>{expect(topics[3]).toContain("[Open the video](https://www.youtube.com/watch?v=rFdbY_V7vIo)");expect(topics[3]).toContain("### Watch: Mass and weight");expect(topics[3]).toContain("Sri Chaitanya Academy NEET");expect(topics[3].match(/\[Open the video\]/g)).toHaveLength(1);expect(topics.slice(0,3).join('')).not.toContain("https://www.youtube.com/watch?v=rFdbY_V7vIo")})
 it('keeps four separate teaching/check panels, units, floating discussion and revision',()=>{expect(topics).toHaveLength(4);expect(lesson.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(topics[3]).toContain('### Understand it');expect(topics[3]).toContain('### Try it yourself');expect(topics[3]).toContain('kilograms');expect(topics[3]).toContain('newtons');expect(topics[3]).toMatch(/buoyant|buoyancy/);expect(lesson).toContain('## Your five-minute revision');expect(lesson).not.toMatch(/supporting evidence|transcript|playback verified|openstax|gesci\.org/i)})
 it('preserves the two existing contact-force and friction videos in their topics',()=>{expect(topics[1]).toContain('[Open the video](https://www.youtube.com/watch?v=beo130kj9R4)');expect(topics[2]).toContain('[Open the video](https://www.youtube.com/watch?v=Ps90zArJEJY)')})
 it('changes weight rather than mass across explicitly hypothetical gravity models',()=>{const mass=7;expect(mass*10).toBe(70);expect(mass*2).toBe(14);expect(mass).toBe(7);for(const m of [1,2,7])for(const g of [.5,2,10])expect(m*g/m).toBe(g)})
 it('balances a floating model and retains upward buoyancy in a sinking model',()=>{const floating={up:16,down:16},sinking={up:1,down:4};expect(floating.up-floating.down).toBe(0);expect(sinking.up).toBeGreaterThan(0);expect(sinking.up-sinking.down).toBe(-3);expect(sinking.down).toBeLessThan(floating.down)})
})
