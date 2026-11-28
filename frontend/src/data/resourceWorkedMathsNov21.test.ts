import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const guide=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
const topic=(id:string,n:number)=>guide(id).split(/^## /m).find(s=>s.startsWith(n+'.'))!
const parse=(digits:string,base:number):number=>{
 if(!Number.isInteger(base)||base<2||![...digits].every(d=>Number(d)<base&&/^[0-9]$/.test(d)))throw Error('invalid digit or base')
 return [...digits].reduce((value,d)=>value*base+Number(d),0)
}
describe('six new Maths reasoning blocks, November21',()=>{
 it('preserves topic, question and embedded-video inventory',()=>{
  for(const [id,videos] of [['resource-maths-3',4],['resource-maths-10',4]] as const){
   const p=guide(id),topics=p.split(/^## /m).filter(t=>/^\d+\./.test(t))
   expect(topics).toHaveLength(4);expect(p.match(/\[Open the video\]/g)).toHaveLength(videos)
   topics.forEach(s=>{expect(s).toContain('### Try it yourself');expect(s).toContain('**Check:**')})
  }
 })
 it('checks binary carries and decimal round trips independently',()=>{
  for(let n=0;n<=255;n++){const digits=n.toString(2);expect(parse(digits,2)).toBe(n);expect(parse((n+1).toString(2),2)).toBe(n+1)}
  expect(parse('1101',2)).toBe(13);expect(parse('1110',2)).toBe(14);expect(parse('111',2)+1).toBe(parse('1000',2))
  expect(()=>parse('1102',2)).toThrow();expect(topic('resource-maths-3',3)).toContain('zero preserves the empty fours place')
 })
 it('distinguishes valid representations from illegal digits',()=>{
  expect(parse('31',5)).toBe(parse('10000',2));expect(parse('23',5)).toBe(13);expect(parse('32',5)).toBe(17)
  expect(()=>parse('31',2)).toThrow();expect(()=>parse('25',5)).toThrow()
  for(let base=2;base<=9;base++)for(let n=0;n<=150;n++)expect(parse(n.toString(base),base)).toBe(n)
  expect(topic('resource-maths-3',4)).toContain('Label the base before comparing')
 })
 it('keeps map units and reverse scale calculations consistent',()=>{
  expect(4.5*80000/100000).toBe(3.6);expect(2*100000/50000).toBe(4)
  for(const scale of [50000,80000,100000])for(let cm=1;cm<=20;cm++)expect((cm*scale/100000)*100000/scale).toBeCloseTo(cm)
  const p=topic('resource-maths-10',1);expect(p).toContain('not a road-distance promise');expect(p).toContain('its printed numerical scale no longer applies')
 })
 it('distinguishes a supplied total from a supplied share',()=>{
  const ratio=[2,3,4],parts=ratio.reduce((a,b)=>a+b,0),shares=ratio.map(x=>180*x/parts)
  expect(shares).toEqual([40,60,80]);expect(shares.reduce((a,b)=>a+b,0)).toBe(180)
  const unit=45/ratio[1];expect(unit*parts).toBe(135);expect(ratio.map(x=>x*unit)).toEqual([30,45,60])
  expect(topic('resource-maths-10',2)).toContain('18 is not the total')
 })
 it('checks pie angles, omitted-category count and inverse reading',()=>{
  const counts=[12,9,30-12-9],angles=counts.map(n=>n/30*360)
  expect(angles).toEqual([144,108,108]);expect(angles.reduce((a,b)=>a+b,0)).toBe(360);expect(72/360*30).toBe(6)
  for(let total=1;total<=50;total++)for(let count=0;count<=total;count++)expect((count/total*360)/360*total).toBeCloseTo(count)
  expect(topic('resource-maths-10',3)).toContain('Every item must be counted once')
 })
 it('uses a constant product rather than subtracting or direction alone',()=>{
  expect(6*8/4).toBe(12);expect(4*6/8).toBe(3)
  const products=[[2,10],[4,8],[6,6]].map(([x,y])=>x*y);expect(new Set(products).size).toBe(3)
  for(const pumps of [1,2,3,4,6,8,12,16,24,48])expect(pumps*(48/pumps)).toBe(48)
  const s=topic('resource-maths-10',4);expect(s).toContain('constant independent flow and no leaks');expect(s).toContain('does not prove inverse proportion')
 })
})
