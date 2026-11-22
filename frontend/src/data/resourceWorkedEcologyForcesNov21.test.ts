import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const guide=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
const topic=(id:string,n:number)=>guide(id).split(/^## /m).find(s=>s.startsWith(n+'.'))!
const chains=(edges:ReadonlyArray<readonly [string,string]>,end:string):string[][]=>{
 const visit=(node:string,path:string[]):string[][]=>{
  if(path.includes(node))throw Error('cyclic fixture')
  const food=edges.filter(([,eater])=>eater===node).map(([prey])=>prey)
  return food.length?food.flatMap(n=>visit(n,[node,...path])):[[node,...path]]
 }
 return visit(end,[])
}
const friction=(objectVelocity:number,surfaceVelocity:number):number=>-Math.sign(objectVelocity-surfaceVelocity)
describe('six ecology and force transfer checks, November21',()=>{
 it('preserves inventory and video URLs while providing independent checks',()=>{
  for(const [id,count] of [['resource-science-12',2],['resource-science-5',2]] as const){
   const p=guide(id),topics=p.split(/^## /m).filter(t=>/^\d+\./.test(t))
   expect(topics).toHaveLength(4)
   const added = (id === 'resource-science-12' && p.includes('[Open the video](https://www.youtube.com/watch?v=2lqhJNgn_Wg)')) || (id === 'resource-science-5' && p.includes('[Open the video](https://www.youtube.com/watch?v=SYU1TKQru1c)'))
   expect(p.match(/\[Open the video\]/g)).toHaveLength(count + Number(added))
   if (id === 'resource-science-12') {
    const urls = [...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m => m[1])
    expect(urls).toEqual(['https://www.youtube.com/watch?v=P1X-WpfUvm4', ...(added ? ['https://www.youtube.com/watch?v=2lqhJNgn_Wg'] : []), 'https://www.youtube.com/watch?v=GK_vRtHJZu4'])
   }
   if(id==='resource-science-5'){const known=["https://www.youtube.com/watch?v=SYU1TKQru1c", "https://www.youtube.com/watch?v=beo130kj9R4", "https://www.youtube.com/watch?v=Ps90zArJEJY"];expect([...p.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(known.filter((url,i)=>i>0||p.includes(`[Open the video](${url})`)))}
   topics.forEach(t=>{expect(t).toContain('### Try it yourself');expect(t).toContain('**Check:**')})
  }
 })
 it('distinguishes a soil sample from each classified component',()=>{
  const living=['seedling','worm'],nonliving=['water','stone']
  expect(new Set([...living,...nonliving]).size).toBe(4)
  const p=topic('resource-science-12',1)
  expect(p).toContain('living organisms as well as non-living mineral particles and water')
  expect(p).toContain('takes up water through its roots');expect(p).toContain('without entering water')
 })
 it('derives two food-to-eater paths and the surviving alternative',()=>{
  const edges=[['lettuce','snail'],['snail','thrush'],['berries','thrush']] as const
  expect(chains(edges,'thrush')).toEqual([['lettuce','snail','thrush'],['berries','thrush']])
  expect(chains(edges.filter(([a,b])=>a!=='snail'&&b!=='snail'),'thrush')).toEqual([['berries','thrush']])
  const p=topic('resource-science-12',2)
  expect(p).toContain('Lettuce → snail → thrush and berries → thrush')
  expect(p).toContain('does not prove there is enough food')
 })
 it('separates material cycling from energy flow without creating energy',()=>{
  const p=topic('resource-science-12',3)
  expect(p).toContain('Nutrients can be released and taken up again')
  expect(p).toContain('some spreads to the surroundings as heat')
  expect(p).toContain('still needs light for photosynthesis')
  expect(p).toContain('Do not handle mould');expect(p).toContain('not expected to happen at the same rate')
 })
 it('checks indirect paths without making a deterministic population promise',()=>{
  expect(chains([['larvae','fish'],['fish','heron']],'heron')).toEqual([['larvae','fish','heron']])
  const p=topic('resource-science-12',4)
  expect(p).toContain('Larvae may increase');expect(p).toContain('Herons may have less')
  expect(p).toContain('neither exact counts nor a certain heron decline')
  expect(p).toContain('not enough by itself to show which caused which')
 })
 it('classifies the actual interacting pair and permits balanced forces',()=>{
  const interactions=[{source:'thread',touches:true},{source:'Earth',touches:false},{source:'magnet',touches:false}]
  expect(interactions.map(i=>i.touches?'contact':'non-contact')).toEqual(['contact','non-contact','non-contact'])
  const forces=[[3,4],[0,-4],[-3,0]];expect(forces.reduce((v,[x,y])=>[v[0]+x,v[1]+y],[0,0])).toEqual([0,0])
  const p=topic('resource-science-5',2)
  expect(p).toContain('where thread and clip touch');expect(p).toContain('Several forces can act even while the clip is at rest')
 })
 it('sets sliding friction from relative motion rather than the hand push',()=>{
  expect(friction(-2,0)).toBe(1);expect(friction(2,0)).toBe(-1)
  for(let v=-10;v<=10;v++)if(v!==0)expect(friction(v,0)*v).toBeLessThan(0)
  const p=topic('resource-science-5',3)
  expect(p).toContain('same direction as the hand');expect(p).toContain('not automatically the applied push')
 })
 it('keeps fair-comparison and physical safety boundaries',()=>{
  const p=topic('resource-science-5',3)
  expect(p).toContain('same book and a similar starting speed')
  expect(p).toContain('do not run or slide on wet floors or plastic sheets')
  expect(p).toContain('would not isolate surface type')
 })
})
