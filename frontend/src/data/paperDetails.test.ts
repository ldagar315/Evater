import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { getPaperDetailBySlug, paperDetailPath, paperDetails, paperForDetail } from './paperDetails'
import { samplePapers } from './samplePapers'
import { paperCollections } from './paperCollections'

const vite = () => readFileSync('vite.config.ts', 'utf8')
const sitemap = () => readFileSync('public/sitemap.xml', 'utf8')

describe('paper detail pages', () => {
  it('covers every catalog paper exactly once with a unique slug and canonical path', () => {
    expect(paperDetails).toHaveLength(samplePapers.length)
    expect(new Set(paperDetails.map(d => d.paperId))).toEqual(new Set(samplePapers.map(p => p.id)))
    expect(new Set(paperDetails.map(d => d.slug)).size).toBe(paperDetails.length)
    for (const d of paperDetails) {
      expect(d.slug).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/)
      expect(d.path).toBe(`/papers/${d.slug}/`)
      expect(getPaperDetailBySlug(d.slug)).toBe(d)
      expect(paperDetailPath(d.paperId)).toBe(d.path)
      expect(paperForDetail(d).id).toBe(d.paperId)
    }
    expect(getPaperDetailBySlug('no-such-paper')).toBeUndefined()
    expect(getPaperDetailBySlug(undefined)).toBeUndefined()
  })

  it('keeps detail slugs clear of the collection routes', () => {
    const collectionSlugs = new Set(Object.values(paperCollections).map(c => c.path.replace(/^\/papers\//, '').replace(/\/$/, '')))
    for (const d of paperDetails) expect(collectionSlugs.has(d.slug)).toBe(false)
  })

  it('prerenders and sitemaps every detail route with the trailing-slash canonical', () => {
    const config = vite()
    const map = sitemap()
    for (const d of paperDetails) {
      expect(config).toContain(`'${d.path.replace(/\/$/, '')}'`)
      expect(map).toContain(`https://evater.xyz${d.path}`)
    }
  })

  it('links to real PDF files that keep their catalog metadata', () => {
    for (const d of paperDetails) {
      const paper = paperForDetail(d)
      expect(existsSync(`public${paper.href}`)).toBe(true)
      expect(d.heading).toContain('Class 8')
      expect(d.seoTitle).toContain('Class 8')
      expect(d.seoDescription.toLowerCase()).toContain('answer')
      expect(d.inside.length).toBeGreaterThanOrEqual(4)
    }
  })

  it('matches the paper format: section marks and question counts agree with the catalog', () => {
    for (const d of paperDetails) {
      const paper = paperForDetail(d)
      const marks = d.sections.reduce((sum, s) => sum + s.marks, 0)
      const questions = d.sections.reduce((sum, s) => sum + s.questions, 0)
      expect(marks).toBe(paper.marks)
      expect(questions).toBe(paper.questions)
      if (paper.format === 'mixed') {
        expect(d.sections).toHaveLength(4)
        const sampled = d.chapterSample?.reduce((sum, c) => sum + c.questions, 0)
        expect(sampled).toBe(paper.questions)
        expect(d.chapterSample?.length).toBe(paper.subject === 'Mathematics' ? 14 : 13)
      } else {
        expect(d.sections).toHaveLength(2)
        expect(d.sections[0].questions).toBe(paper.objectiveQuestions ?? 10)
        expect(d.sections[1].questions).toBe(paper.writtenQuestions ?? 5)
        expect(d.chapterSample).toBeUndefined()
      }
    }
  })

  it('makes no official-paper claim and points study links at live prerendered guides', () => {
    const config = vite()
    for (const d of paperDetails) {
      expect(d.sourceNote).toMatch(/original Evater practice/i)
      expect(d.sourceNote).not.toMatch(/official CBSE|official NCERT paper(?! or marking scheme)/i)
      expect(d.seoTitle + d.seoDescription + d.intro).not.toMatch(/official CBSE paper/i)
      if (d.studyGuide) {
        expect(d.studyGuide.href).toMatch(/^\/blog\/[a-z0-9-]+\/$/)
        expect(config).toContain(`'${d.studyGuide.href.replace(/\/$/, '')}'`)
      }
    }
  })
})
