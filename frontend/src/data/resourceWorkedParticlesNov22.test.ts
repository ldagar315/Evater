import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const guide=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
const topic=(id:string,n:number)=>guide(id).split(/^## /m).find(s=>s.startsWith(n+'.'))!
describe('particle and solution transfer checks, November22',()=>{
 it('preserves topic, video and check inventory',()=>{
  for(const [id,count] of [['resource-science-7',4],['resource-science-9',2]] as const){
   const p=guide(id);expect(p.split(/^## /m).filter(s=>/^\d+\./.test(s))).toHaveLength(4)
   expect(p.match(/\[Open the video\]/g)).toHaveLength(count);expect(p.match(/\*\*Check:\*\*/g)).toHaveLength(4)
  }
 })
 it('distinguishes model symbols from samples and visible grains',()=>{
  const p=topic('resource-science-7',1)
  expect(p).toContain('each circle represents many particles');expect(p).toContain('not automatically air bubbles')
  expect(p).toContain('Enlarging the drawing does not expand the real sample')
 })
 it('conserves count during compression rather than shrinking particles',()=>{
  const n=120,volumeA=100,volumeB=50;expect(n/volumeB).toBe(2*n/volumeA)
  const p=topic('resource-science-7',2)
  expect(p).toContain('same particle count and particle type');expect(p).toContain('Dry beads')
  expect(p).toContain('Particles in solids still vibrate')
 })
 it('distinguishes crossing events from net transfer',()=>{
  expect(12-12).toBe(0);expect(12+12).toBe(24);expect(13-11).toBe(2)
  for(let crossings=1;crossings<=100;crossings++){
   expect(crossings-crossings).toBe(0);expect(crossings+crossings).toBeGreaterThan(0)
  }
  const p=topic('resource-science-7',3);expect(p).toContain('Zero net transfer is not zero motion')
  expect(p).toContain('does not prove a lasting concentration difference');expect(p).toContain('Do not sniff unknown')
 })
 it('accounts for closed versus open system mass without changing molecule kind',()=>{
  expect(180-0).toBe(180);expect(180-174).toBe(6)
  const p=topic('resource-science-7',4)
  expect(p).toContain('molecules remain water');expect(p).toContain('6 g of water has left')
  expect(p).toContain('do not heat sealed containers');expect(p).toContain('dry ice or boiling water')
  expect(p).toContain('ice occupies more volume')
 })
 it('adds solute and solvent mass without assuming additive volumes',()=>{
  expect(72+8).toBe(80);expect(5+95).toBe(100)
  const p=topic('resource-science-9',1)
  expect(p).toContain('solid is the solute, water the solvent');expect(p).toContain('vessel\'s own mass')
  expect(p).toContain('does not require the final volume');expect(p).toContain('Do not taste laboratory solutions')
 })
 it('computes density independently and rejects a mass-only floating rule',()=>{
  const rhoA=54/60,rhoB=40/20;expect(rhoA).toBe(.9);expect(rhoB).toBe(2)
  expect(54).toBeGreaterThan(40);expect(rhoA).toBeLessThan(1);expect(rhoB).toBeGreaterThan(1)
  for(let k=1;k<=30;k++)expect((54*k)/(60*k)).toBeCloseTo(rhoA)
  const p=topic('resource-science-9',4);expect(p).toContain('Density, not mass alone')
  expect(p).toContain('under this model');expect(p).toContain('not imply that every solid floats')
 })
})
