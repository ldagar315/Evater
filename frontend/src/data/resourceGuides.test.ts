import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resourceGuides } from './resourceGuides'
import { blogPosts, getPostBySlug } from './blogPosts'
describe('incremental resource guide release', () => {
 it('publishes only the three selected reviewed guides with distinct identities', () => {
  expect(resourceGuides).toHaveLength(3)
  expect(new Set(blogPosts.map(p=>p.id)).size).toBe(blogPosts.length)
  expect(new Set(blogPosts.map(p=>p.slug)).size).toBe(blogPosts.length)
  resourceGuides.forEach(p=>{expect(p.status).toBe('published');expect(getPostBySlug(p.slug)).toBe(p)})
 })
 it('keeps topic checks, caveats, sources and working site discovery', () => {
  const sitemap=readFileSync('public/sitemap.xml','utf8'), vite=readFileSync('vite.config.ts','utf8')
  resourceGuides.forEach(p=>{
   expect(p.content.match(/### Try without looking/g)).toHaveLength(4)
   expect(p.content.match(/^Answer:/gm)).toHaveLength(4)
   expect(p.content).toContain('Video coverage remains partial')
   expect(p.content).toContain('not complete audiovisual or current Indian playback verification')
   expect(p.content).toContain('[Read the NCERT chapter](https://ncert.nic.in/')
   expect(p.content).toContain('[Browse Evater sample papers](/papers/)')
   expect(p.content).toContain('[Open the video](https://')
   expect(p.content).not.toContain('Private review draft')
   expect(p.content).not.toContain('No public blog post')
   expect(sitemap).toContain(`/blog/${p.slug}/`)
   expect(vite).toContain(`/blog/${p.slug}`)
  })
 })
 it('preserves the material language, fit and safety qualifications', () => {
  expect(resourceGuides[0].content).toContain('power-of-a-power')
  expect(resourceGuides[1].content).toContain('converting metres to centimetres')
  expect(resourceGuides[2].content).toContain('not a verified YouTube')
  expect(resourceGuides[2].content).toContain('wet floors')
  expect(resourceGuides[2].content).toContain('apparent weight')
 })
})
