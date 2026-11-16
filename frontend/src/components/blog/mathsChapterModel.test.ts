import { describe, it, expect } from 'vitest'
import { areaShape, powerGrowth } from './mathsChapterModel'
import { mathsChapterGuides } from '../../data/mathsChapterGuides'
import { blogPosts } from '../../data/blogPosts'
describe('Power Play and Area', () => {
 it('keeps addition and multiplication distinct, including zero steps', () => {
  expect(powerGrowth(2,0)).toMatchObject({added:1,multiplied:1})
  expect(powerGrowth(2,4)).toMatchObject({added:9,multiplied:16})
  expect(powerGrowth(5,6).multiplied).toBe(15625)
  expect(powerGrowth(NaN,Infinity)).toMatchObject({base:2,steps:0})
  expect(powerGrowth(10,-3)).toMatchObject({base:5,steps:0})
 })
 it('keeps area invariant under sideways shift and uses perpendicular height', () => {
  for(let s=-3;s<=3;s++) expect(areaShape(8,3,s)).toMatchObject({area:24,triangleArea:12})
  expect(areaShape(8,3,0).slopingSide).toBe(3)
  expect(areaShape(8,3,3).slopingSide).toBeCloseTo(Math.sqrt(18))
  expect(areaShape(NaN,Infinity,NaN)).toMatchObject({base:8,height:3,shift:0})
 })
 it('has original full guides, qualified source and no direct unverified concept targets', () => {
  expect(mathsChapterGuides).toHaveLength(2)
  expect(new Set(blogPosts.map(p=>p.id)).size).toBe(blogPosts.length)
  expect(new Set(blogPosts.map(p=>p.slug)).size).toBe(blogPosts.length)
  mathsChapterGuides.forEach(p=>{
   expect(p.content.split(/\s+/).length).toBeGreaterThan(700)
   expect(p.content).toContain('Answer check:')
   expect(p.content).toContain('](/practice)')
   expect(p.content).not.toContain('/practice?')
  })
  expect(mathsChapterGuides[0].content).toContain('does not define 0⁰')
  expect(mathsChapterGuides[0].content).toContain('nonzero')
  expect(mathsChapterGuides[1].content).toContain('official PDF bytes')
  expect(mathsChapterGuides[1].content).toContain('unreachable')
  expect(mathsChapterGuides[1].content).toContain('Equal area alone does not prove')
 })
})
