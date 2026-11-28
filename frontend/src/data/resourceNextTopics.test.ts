import { describe, expect, it } from 'vitest'
import { resourceGuides } from './resourceGuides'
const lesson=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
const root=(n:number):number=>n<10?n:root([...String(n)].reduce((a,d)=>a+Number(d),0))
describe('November19 topic teaching and source boundaries',()=>{
 it('covers all seven Number Play teaching/check blocks and keeps its existing two videos',()=>{
  const p=lesson('resource-maths-5')
  const topics=p.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
  expect(topics).toHaveLength(7)
  const known=[["nCq3rxCG2MA",1,false],["WC2puADmICM",2,true],["znmPfDfsir8",3,true],["_dT97MUYuzs",4,false],["D0CkdFSWLOA",5,false],["P1qyjdh_sIw",6,false],["dTXMhOGI9Z8",7,false]] as const
  const selected=known.filter(([id,,required])=>required||p.includes(`[Open the video](https://www.youtube.com/watch?v=${id})`))
  expect([...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(selected.map(([id])=>`https://www.youtube.com/watch?v=${id}`))
  for(const[id,topic]of selected)expect(topics[topic-1]).toContain(`[Open the video](https://www.youtube.com/watch?v=${id})`)
  topics.forEach(t=>{expect(t).toContain('### Try it yourself');expect(t).toContain('**Check:**')})
  for(const text of ['same letter keeps the same digit','first digit of a multi-digit number cannot be zero','same digital root','Passing the check does not prove','negative','whole number']) expect(p.toLowerCase()).toContain(text.toLowerCase())
 })
 it('checks original puzzle assignments exhaustively, including both leading digits',()=>{
  const answers=[]
  for(let a=1;a<=9;a++)for(let b=1;b<=9;b++)if(a!==b&&(10*a+b)+(10*b+a)===99)answers.push([a,b])
  expect(answers).toHaveLength(8)
  expect(answers).toContainEqual([2,7]);expect(answers).toContainEqual([4,5])
  expect(27+72).toBe(99);expect(45+54).toBe(99)
 })
 it('checks sign-choice parity over negative and positive consecutive integers',()=>{
  for(let n=-100;n<=100;n++)for(let mask=0;mask<8;mask++){
   const value=n+[1,2,3].reduce((sum,k)=>sum+(mask&(1<<(k-1))?-1:1)*(n+k),0)
   expect(Math.abs(value%2)).toBe(0)
  }
  expect(23-24+25-26).toBe(-2)
 })
 it('checks the remainder argument for all tested whole-number quotients',()=>{
  for(let a=0;a<30;a++)for(let b=0;b<30;b++){
   expect(((7*a+5)+(7*b+4))%7).toBe(2)
   expect((7*a+5)+(7*b+4)).toBe(7*(a+b+1)+2)
  }
  expect((47+25)%7).toBe(2)
 })
 it('demonstrates a digital-root false positive rather than calling it proof',()=>{
  expect(46*28).toBe(1288);expect(1297).not.toBe(46*28)
  expect(root(root(46)*root(28))).toBe(1)
  expect(root(1288)).toBe(1);expect(root(1297)).toBe(1)
  expect(1297-1288).toBe(9)
  expect(root(18)).toBe(9)
 })
 it('keeps particle-motion, molecular-identity, mass and activity limits explicit',()=>{
  const p=lesson('resource-science-7')
  for(const text of ['KRLNDTmBFZY','xYU7RSoOZ0U','particles keep moving','not the bonds holding each water molecule','closed system','do not handle dry ice','do not sniff'])expect(p.toLowerCase()).toContain(text.toLowerCase())
  const s=lesson('resource-science-9')
  for(const text of ['JQRIlS9lNdg','temperature and amount of solvent','can still look clear','same pressure','Dissolving faster','Do not taste laboratory'])expect(s).toContain(text)
 })
})
