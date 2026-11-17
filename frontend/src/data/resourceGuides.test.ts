import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resourceGuides } from './resourceGuides'
import { blogPosts, getPostBySlug } from './blogPosts'
import { youtubeVideoId } from '../components/blog/resourceVideo'
describe('student resource lessons', () => {
 it('keeps all nine live routes and distinct identities', () => {
  expect(resourceGuides).toHaveLength(9)
  expect(new Set(blogPosts.map(p=>p.id)).size).toBe(blogPosts.length)
  expect(new Set(blogPosts.map(p=>p.slug)).size).toBe(blogPosts.length)
  const sitemap=readFileSync('public/sitemap.xml','utf8'),vite=readFileSync('vite.config.ts','utf8')
  resourceGuides.forEach(p=>{expect(p.status).toBe('published');expect(getPostBySlug(p.slug)).toBe(p);expect(sitemap).toContain(`/blog/${p.slug}/`);expect(vite).toContain(`/blog/${p.slug}`)})
 })
 it('teaches each topic on-page with at most one verified YouTube video', () => {
  resourceGuides.forEach((p,i)=>{
   const topics=p.content.split(/^## /m).slice(1,5)
   expect(topics).toHaveLength(4)
   expect(p.content.match(/\[Open the video\]/g)).toHaveLength([4,4,2,4,3,2,4,2,1][i])
   topics.forEach(t=>{expect(t.match(/\[Open the video\]/g)?.length??0).toBeLessThanOrEqual(1);expect(t).toContain('### Try it yourself');expect(t).toContain('**Check:**')})
   const urls=[...p.content.matchAll(/\]\((https?:[^)]+)\)/g)].map(m=>m[1])
   urls.forEach(url=>expect(youtubeVideoId(url)).not.toBeNull())
   expect(p.content).toContain('## Your five-minute revision')
   expect(p.content).toContain('](/papers/)')
   for(const ledger of ['Runtime evidence','supporting source','Supporting reading','transcript','playback check','Evidence and limits','Video coverage remains partial']) expect(p.content.toLowerCase()).not.toContain(ledger.toLowerCase())
   expect(p.excerpt).not.toContain('coverage remains partial')
  })
 })
 it('keeps important mathematical conditions and practical safety', () => {
  expect(resourceGuides[3].content).toContain('11 < √123 < 12')
  expect(resourceGuides[4].content).toContain('original ₹200')
  expect(resourceGuides[5].content).toContain('energy flows through')
  expect(resourceGuides[5].content).not.toContain('body decompose')
  expect(resourceGuides[0].content).toContain('nonzero')
  expect(resourceGuides[0].content).toContain('1 ≤ c < 10')
  expect(resourceGuides[1].content).toContain('75 : 300 = 1 : 4')
  expect(resourceGuides[1].content).toContain('delivery charge')
  expect(resourceGuides[2].content).toContain('wet floors')
  expect(resourceGuides[2].content).toContain('adult or teacher')
  expect(resourceGuides[2].content).toContain('balanced')
  expect(resourceGuides[2].content).toContain('not a rule that every smoother surface always has less friction')
 })
})
