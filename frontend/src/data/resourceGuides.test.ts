import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resourceGuides } from './resourceGuides'
import { blogPosts, getPostBySlug } from './blogPosts'
import { youtubeVideoId } from '../components/blog/resourceVideo'

type VideoSlot = readonly [videoId: string, topic: number, required: boolean]
const knownVideoInventories: Record<string, readonly VideoSlot[]> = {
 'resource-maths-5': [['WC2puADmICM',2,true],['znmPfDfsir8',3,true],['dTXMhOGI9Z8',7,false]],
 'resource-science-11': [["l64YwNl1wr0",1,false],["Jip3BbZBpsM",2,true]],
 'resource-maths-9': [["JVrkLIcA2qw", 1, true], ["JVrkLIcA2qw", 2, true], ["UKaoHO4ClS4", 3, false], ["YompsDlEdtc", 4, true], ["Eo04qjLRiSY", 5, false]],
 'resource-science-5': [["SYU1TKQru1c", 1, false], ["beo130kj9R4", 2, true], ["Ps90zArJEJY", 3, true], ["rFdbY_V7vIo", 4, false]],
 'resource-maths-6': [["ZJA1qz4XovY", 1, true],["Xy8NKUoyy98", 2, false],["4GcNzvILqtM", 3, true]],
 'resource-maths-14': [['iNZZIb_-PA8',1,false],['hm17lVaor0Q',2,true],['loAA3TCNAvU',3,true],['loAA3TCNAvU',4,true]],
 'resource-science-12': [['P1X-WpfUvm4',1,true],['2lqhJNgn_Wg',2,false],['K7YpPgbmSew',3,false],['GK_vRtHJZu4',4,true]],
 'resource-science-13': [['Bsp5JYNMAQE',1,true],['o8K0RTYM6Mc',2,false]],
 'resource-science-8': [["vi_SJBnxmHo", 1, false], ["nUzOXy9V-K0", 2, false], ["Vt7lN4QPU0k", 3, true], ["KgUmNQD6m5Q", 4, false]],
}
const videoLink = (id: string) => `[Open the video](https://www.youtube.com/watch?v=${id})`
const assertKnownVideoInventory = (id: string, text: string): number | null => {
 const inventory=knownVideoInventories[id]
 if(!inventory)return null
 const selected=inventory.filter(([videoId,,required])=>required||text.includes(videoLink(videoId)))
 const links=[...text.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[0])
 expect(links).toEqual(selected.map(([videoId])=>videoLink(videoId)))
 const topics=text.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
 for(const [videoId,topic] of selected)expect(topics[topic-1]).toContain(videoLink(videoId))
 return selected.length
}

describe('student resource lessons', () => {
 it('keeps all 27 lesson routes and distinct identities', () => {
  expect(resourceGuides).toHaveLength(27)
  expect(new Set(blogPosts.map(p=>p.id)).size).toBe(blogPosts.length)
  expect(new Set(blogPosts.map(p=>p.slug)).size).toBe(blogPosts.length)
  const sitemap=readFileSync('public/sitemap.xml','utf8'),vite=readFileSync('vite.config.ts','utf8')
  resourceGuides.forEach(p=>{expect(p.status).toBe('published');expect(getPostBySlug(p.slug)).toBe(p);expect(sitemap).toContain(`/blog/${p.slug}/`);expect(vite).toContain(`/blog/${p.slug}`)})
 })
 it('teaches each topic on-page with at most one verified YouTube video', () => {
  resourceGuides.forEach((p,i)=>{
   const topics=p.content.split(/^## /m).slice(1).filter(t=>/^\d+\./.test(t))
   expect(topics).toHaveLength(p.id === 'resource-maths-9' ? 6 : p.id === 'resource-maths-5' ? 7 : 4)
   const expectedVideos = [4,4,2,4,3,2,4,2,3,2,2,3,4,4,4,2,3,3,2,3,3,4,1,2,4,1,1]
   expect(p.content.match(/\[Open the video\]/g)??[]).toHaveLength(assertKnownVideoInventory(p.id,p.content)??expectedVideos[i])
   topics.forEach(t=>{expect(t.match(/\[Open the video\]/g)?.length??0).toBeLessThanOrEqual(1);expect(t).toContain('### Try it yourself');expect(t).toContain('**Check:**')})
   const urls=[...p.content.matchAll(/\]\((https?:[^)]+)\)/g)].map(m=>m[1])
   urls.forEach(url=>expect(youtubeVideoId(url)).not.toBeNull())
   expect(p.content).toContain('## Your five-minute revision')
   expect(p.content).toContain('](/papers/)')
   for(const ledger of ['Runtime evidence','supporting source','Supporting reading','transcript','playback check','Evidence and limits','Video coverage remains partial']) expect(p.content.toLowerCase()).not.toContain(ledger.toLowerCase())
   expect(p.excerpt).not.toContain('coverage remains partial')
  })
 })
 it('keeps new topic scope and safety grounded', () => {
  const find=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
  expect(find('resource-maths-11')).not.toContain("Euler")
  expect(find('resource-maths-11')).toContain('5 faces, 9 edges and 6 vertices')
  expect(find('resource-science-6')).toContain('perpendicular force')
  expect(find('resource-science-4')).toContain('chemical reactions')
 })
 it('adds six topic placements and a bounded historical extension without unsafe activities', () => {
  const find=(id:string)=>resourceGuides.find(p=>p.id===id)!.content
  expect(find('resource-science-4')).toContain('can overheat and catch fire')
  expect(find('resource-science-4')).toContain('do not copy')
  expect(find('resource-science-2')).toContain('teacher-prepared slides')
  expect(find('resource-science-10')).toContain('Do not try focusing sunlight')
  const pythagoras=find('resource-maths-9')
  expect(pythagoras).toContain('6√2 cm')
  expect(pythagoras).toContain('Andrew Wiles proved it')
  expect(pythagoras).toContain('One failed cube example')
  expect(6**2+6**2).toBe(72)
  expect(3**3+4**3).toBe(91)
  expect(5**3).toBe(125)
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
 it('keeps exact known inventories strict with or without the selected pending additions', () => {
  const render=(rows: readonly VideoSlot[],topicCount=4)=>Array.from({length:topicCount},(_,i)=>`## ${i+1}. Topic\n`+rows.filter(([,t])=>t===i+1).map(([v])=>videoLink(v)).join('\n')).join('\n')
  for(const [id,inventory] of Object.entries(knownVideoInventories)) {
   const topicCount=id==='resource-maths-9'?6:id==='resource-maths-5'?7:4
   const required=inventory.filter(([, ,r])=>r)
   const optional=inventory.filter(([, ,r])=>!r)
   for(let mask=0;mask<2**optional.length;mask++){const selected=inventory.filter(row=>row[2]||!!(mask&(1<<optional.indexOf(row))));expect(assertKnownVideoInventory(id,render(selected,topicCount))).toBe(selected.length)}
   const present=render(inventory,topicCount),first=videoLink(required[0][0])
   for(const bad of [present+'\n'+videoLink('aaaaaaaaaaa'),present.replace(first,''),present.replace(first,videoLink('bbbbbbbbbbb')),present+'\n'+first,present.replace(first,'')+'\n'+first,present.split(/(?=^## )/m).reverse().join('')])expect(()=>assertKnownVideoInventory(id,bad)).toThrow()
  }
 })

})
