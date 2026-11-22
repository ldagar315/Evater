import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const topic=(id:string,n:number)=>resourceGuides.find(p=>p.id===id)!.content.split(/^## /m).find(s=>s.startsWith(n+'.'))!
const sci=(ch:number,n:number)=>topic('resource-science-'+ch,n)
describe('twelve independent investigation, cell and electrical checks, Nov22',()=>{
 it('preserves four topics, checks and existing video counts',()=>{
  for(const [ch,count] of [[1,3],[2,3],[4,3]]){
   const p=resourceGuides.find(g=>g.id==='resource-science-'+ch)!.content
   expect(p.split(/^## /m).filter(s=>/^\d+\./.test(s))).toHaveLength(4)
   expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4);const variables=ch===1&&p.includes('[Open the video](https://www.youtube.com/watch?v=BXFU86GNmrg)');expect(p.match(/\[Open the video\]/g)).toHaveLength(count+Number(variables));if(ch===1){const known=[['La0m_ocR7cQ',1],['BXFU86GNmrg',2],['hRAFPdDppzs',3],['La0m_ocR7cQ',4]] as const;const selected=known.filter(([,n])=>n!==2||variables);expect([...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(selected.map(([v])=>`https://www.youtube.com/watch?v=${v}`));for(const[v,n]of selected)expect(sci(1,n)).toContain(`[Open the video](https://www.youtube.com/watch?v=${v})`)}
  }
 })
 it('separates measured height from unrecorded light and starting-height claims',()=>{
  expect(14-11).toBe(3);const p=sci(1,1)
  expect(p).toContain('not a recorded fact about exposure');expect(p).toContain('Check starting heights')
  expect(p).toContain('do not create energy')
 })
 it('identifies simultaneous changes instead of isolating temperature falsely',()=>{
  const p=sci(1,2)
  for(const phrase of ['mass, particle size, stirring and water volume also differ','same defined dissolving endpoint','repeated trials','not an instruction to handle hot water'])expect(p).toContain(phrase)
 })
 it('distinguishes precise repeated readings from accuracy and persistent offset',()=>{
  const values=[104,104,104];expect(Math.max(...values)-Math.min(...values)).toBe(0)
  expect(values.reduce((a,b)=>a+b)/3-100).toBe(4)
  const p=sci(1,3);expect(p).toContain('precise in this record');expect(p).toContain('possible systematic offset')
  expect(p).toContain('rather than assuming averaging fixes the bias')
 })
 it('keeps the contrary trial and limits an always-faster claim',()=>{
  const a=[28,29,30,31],b=[24,25,26,52]
  expect(b.filter((x,i)=>x<a[i])).toHaveLength(3);expect(b[3]).toBeGreaterThan(a[3])
  const p=sci(1,4);expect(p).toContain('Keep all readings, including 52');expect(p).toContain('does not support')
  expect(p).toContain('separately documented measurement or recording error')
 })
 it('does not equate absent chloroplasts or bigger pictures with valid identification/detail',()=>{
  const p=sci(2,1)
  for(const phrase of ['plant cells need not all contain chloroplasts','not necessarily resolution','teacher-prepared slides only','do not prepare body samples','do not have a well-defined nucleus'])expect(p).toContain(phrase)
 })
 it('keeps multicellular hierarchy distinct from unicellular life',()=>{
  const p=sci(2,2)
  expect(p).toContain('nerve cell, nervous tissue, brain, nervous system, human body')
  expect(p).toContain('different tissues working together');expect(p).toContain('within its one cell')
 })
 it('uses a unicellular fungus and useful microbe counterexamples without authorizing unknown cultures',()=>{
  const p=sci(2,3)
  for(const phrase of ['single-celled fungus','useful examples','Do not handle, sniff, taste or culture unknown','Treat viruses separately'])expect(p).toContain(phrase)
 })
 it('computes different dough rises but does not infer gas identity or food safety from height',()=>{
  expect(35-20).toBe(15);expect(21-20).toBe(1);expect((35-20)-(21-20)).toBe(14)
  const p=sci(2,4);expect(p).toContain('fictional teacher record');expect(p).toContain('does not itself chemically identify')
  expect(p).toContain('Do not taste experimental mixtures or culture unknown microbes')
 })
 it('separates a necessary complete circuit from a guarantee of functioning components',()=>{
  const p=sci(4,1)
  for(const phrase of ['lacks a complete return path','break at the open switch','necessary, not a promise','weak cell, poor contact or faulty lamp','never a wall socket'])expect(p).toContain(phrase)
 })
 it('does not erase background fields at switch-off and retains video fire warning',()=>{
  const p=sci(4,2)
  expect(p).toContain("not Earth's field");expect(p).toContain('Watch only: do not copy the paper electromagnet')
  expect(p).toContain('overheating and fire risk')
 })
 it('qualifies current-controlled attraction and release in actual electromagnets',()=>{
  const p=sci(4,3)
  for(const phrase of ['not every metal','can retain some magnetism','not a build instruction','overheat and catch fire'])expect(p).toContain(phrase)
 })
 it('rejects useful-effect equals safe-operation and touch testing',()=>{
  const p=sci(4,4)
  for(const phrase of ['not establish a safe temperature','Do not touch-test, open or repair','alert a responsible adult','do not open appliances'.toLowerCase()])expect(p.toLowerCase()).toContain(phrase.toLowerCase())
 })
})
