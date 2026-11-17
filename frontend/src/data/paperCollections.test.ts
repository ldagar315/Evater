import { describe, expect, it } from 'vitest'
import { paperCollections } from './paperCollections'
import { samplePapers, filterPapers } from './samplePapers'
import { readFileSync } from 'node:fs'
describe('public Class 8 paper collections', () => {
  it('maps each collection to a unique crawlable canonical route and real inventory', () => {
    expect(new Set(Object.values(paperCollections).map(p => p.path)).size).toBe(3)
    for (const [key, page] of Object.entries(paperCollections)) {
      expect(page.title).toContain('Class 8')
      expect(page.description).toContain('answers')
      expect(page.path.endsWith('/')).toBe(true)
      const papers = samplePapers.filter(p => !page.subject || p.subject === page.subject)
      expect(papers.length).toBe(key === 'all' ? 29 : key === 'maths' ? 15 : 14)
      expect(filterPapers(papers, { query: '', classLevel: '', subject: '', topic: '' })).toHaveLength(papers.length)
      expect(filterPapers(papers, { query: 'no-such-topic', classLevel: '', subject: '', topic: '' })).toHaveLength(0)
    }
  })
  it('prerenders and lists both subject routes in the sitemap without filter URLs', () => {
    const sitemap = readFileSync('public/sitemap.xml', 'utf8')
    const vite = readFileSync('vite.config.ts', 'utf8')
    for (const page of Object.values(paperCollections)) {
      expect(sitemap).toContain(`https://evater.xyz${page.path}`)
      expect(vite).toContain(`'${page.path.replace(/\/$/, '')}'`)
    }
    expect(sitemap).not.toContain('?subject=')
  })
})
