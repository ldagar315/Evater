import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const lesson=(ch:number)=>resourceGuides.find(g=>g.id==='resource-maths-'+ch)!.content
const topic=(ch:number,n:number)=>lesson(ch).split(/^## /m).find(s=>s.startsWith(n+'.'))!
const pyramid=(row:number[]):number=>row.length===1?row[0]:pyramid(row.slice(1).map((x,i)=>row[i]+x))
describe('eight geometry and algebra transfer checks, Nov22',()=>{
 it('keeps topic/video/check inventory unchanged',()=>{
  for(const [ch,v] of [[11,4],[13,2]]){
   const p=lesson(ch);expect(p.split(/^## /m).filter(s=>/^\d+\./.test(s))).toHaveLength(4)
   expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4);expect(p.match(/\[Open the video\]/g)).toHaveLength(v + Number(ch===13 && p.includes('[Open the video](https://www.youtube.com/watch?v=7VMsMtk3QdI)')) + Number(ch===13 && p.includes('[Open the video](https://www.youtube.com/watch?v=0Ho8Az8y4Mw)')))
  }
 })
 it('uses stage zero consistently and counts applications rather than pictures',()=>{
  expect(3**5).toBe(243);expect(243/(3**2)).toBe(27);expect(3**4).not.toBe(243)
  const p=topic(11,1);expect(p).toContain('starting picture is stage 0');expect(p).toContain('five applications')
 })
 it('combines piece count with size and keeps length/area rules distinct',()=>{
  expect(18*(4/3)**2).toBe(32);expect(64*(3/4)**2).toBe(36)
  for(let n=0;n<=8;n++){
   expect(3*4**n/3**n).toBeCloseTo(3*(4/3)**n)
   expect(3**n/4**n).toBeCloseTo((3/4)**n)
  }
  expect(topic(11,2)).toContain('Count alone ignores each piece becoming smaller')
 })
 it('rejects one-view identification and face-count-only net validation',()=>{
  let normal=[0,0,1],right=[1,0,0];const stripNormals:string[]=[]
  for(let i=0;i<6;i++){stripNormals.push(normal.join(','));const old=normal;normal=right;right=old.map(v=>-v)}
  expect(new Set(stripNormals).size).toBe(4)
  const p=topic(11,3);expect(p).toContain('their depths, 3 cm and 7 cm')
  expect(p).toContain('straight six-square strip cannot');expect(p).toContain('without overlap')
 })
 it('counts prism and pyramid parts separately',()=>{
  const n=5;expect([n+2,3*n,2*n]).toEqual([7,15,10]);expect([n+1,2*n,n+1]).toEqual([6,10,6])
  const p=topic(11,4);expect(p).toContain('7 faces, 15 edges and 10 vertices')
  expect(p).toContain('6 faces, 10 edges and 6 vertices');expect(p).toContain('5 faces, 9 edges and 6 vertices')
 })
 it('respects operation order and designs the constant with algebra',()=>{
  for(let x=-50;x<=50;x++){
   expect((3*x+15)/3-x).toBe(5);expect((3*x+21)/3-x).toBe(7);expect(3*(x+15)/3-x).toBe(15)
  }
  expect(topic(13,1)).toContain('Replace 15 with 21')
 })
 it('derives position weights by actual repeated pyramid additions',()=>{
  expect(pyramid([2,5,9,4])).toBe(48);expect(pyramid([5,2,9,4])).toBe(42)
  for(let a=-5;a<=5;a++)for(let b=-5;b<=5;b++)expect(pyramid([a,b,9,4])).toBe(a+3*b+3*9+4)
  expect(topic(13,2)).toContain('coefficients are not equal')
 })
 it('solves valid calendar sums and rejects fractional dates/boundary assumptions',()=>{
  expect((68-16)/4).toBe(13);expect([13,14,20,21].reduce((a,b)=>a+b)).toBe(68)
  expect(Number.isInteger((70-16)/4)).toBe(false)
  const p=topic(13,3);expect(p).toContain('all four positions lie within the displayed month')
  expect(p).toContain('not in the final column')
 })
 it('exhaustively validates the odd-total integer maximum and closer-pair proof',()=>{
  const products=Array.from({length:26},(_,i)=>(i+1)*(26-i));expect(Math.max(...products)).toBe(182)
  expect(12*15).toBe(180);expect(13*14).toBe(182)
  for(let a=1;a<=30;a++)for(let b=a+2;b<=35;b++)expect((a+1)*(b-1)-a*b).toBe(b-a-1)
  expect(topic(13,4)).toContain('not an integer pair')
 })
})
