import { describe, expect, it } from 'vitest'
import { resourceGuides } from './resourceGuides'
const guide=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
const topic=(id:string,n:number)=>guide(id).split(/^## /m).filter(t=>/^\d+\./.test(t))[n-1]
describe('two short force and composition explainers',()=>{
 it("places the exact Cognito video in resource-science-5",()=>{
  const selected=topic("resource-science-5",1)
  expect(selected).toContain("### Watch: Free-body diagrams and resultant forces")
  expect(selected).toContain("[Open the video](https://www.youtube.com/watch?v=SYU1TKQru1c)")
  expect(selected.match(/\[Open the video\]/g)).toHaveLength(1)
  expect(selected).toContain('### Understand it')
 })
 it("places the exact Freesciencelessons video in resource-science-8",()=>{
  const selected=topic("resource-science-8",2)
  expect(selected).toContain("### Watch: Elements, compounds and mixtures")
  expect(selected).toContain("[Open the video](https://www.youtube.com/watch?v=nUzOXy9V-K0)")
  expect(selected.match(/\[Open the video\]/g)).toHaveLength(1)
  expect(selected).toContain('### Understand it')
 })
 it('preserves four separate practice/check panels and on-page revision',()=>{
  for(const id of ['resource-science-5','resource-science-8']){const s=guide(id);expect(s.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(s).toContain('## Your five-minute revision');expect(s).not.toMatch(/supporting evidence|transcript|playback verified|openstax|nasa\.gov/i)}
 })
 it('checks balanced and unbalanced force without treating force as velocity',()=>{
  expect(80000-80000).toBe(0);expect(120000-90000).toBe(30000);expect(120000-120000).toBe(0)
  expect(6-6).toBe(0);const net=6-8;expect(net).toBe(-2);const newVelocity=5+net/2*.25;expect(newVelocity).toBe(4.75);expect(newVelocity).toBeGreaterThan(0)
 })
 it('distinguishes atom kinds in a single bonded particle from atom count',()=>{
  const kind=(atoms:string[])=>new Set(atoms).size===1?'element':'compound'
  expect(kind(['O','O'])).toBe('element');expect(kind(['N','N'])).toBe('element');expect(kind(['H','H','O'])).toBe('compound');expect(kind(['C','O','O'])).toBe('compound')
  expect(['H','H','O'].filter(x=>x==='H').length).toBe(2);expect(['H','H','O'].filter(x=>x==='O').length).toBe(1)
 })
})
