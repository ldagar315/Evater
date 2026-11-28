import { describe, it, expect } from 'vitest'
import { resourceGuides } from './resourceGuides'
import { resourcePracticeHref, resourcePracticeTarget } from './resourcePracticeTargets'
import { parseChapterPracticeTarget, resolveChapterPracticeTarget } from '../lib/chapterPracticeTarget'
import { readFileSync } from 'node:fs'

describe('all resource lesson practice handoffs', () => {
  it('covers exactly fourteen Maths and thirteen Science chapter lessons once each', () => {
    expect(resourceGuides).toHaveLength(27)
    const targets = resourceGuides.map(guide => resourcePracticeTarget(guide.slug)!)
    for (const [subject, total] of [['mathematics', 14], ['science', 13]] as const) {
      expect(targets.filter(t => t.subject === subject).map(t => t.chapterNumber).sort((a,b) => a-b)).toEqual(Array.from({length:total},(_,i)=>i+1))
    }
    expect(new Set(resourceGuides.map(g=>g.slug)).size).toBe(27)
  })
  it('round-trips all lesson destinations through the existing strict contract', () => {
    for (const guide of resourceGuides) {
      const href = resourcePracticeHref(guide.slug)!
      const target = resourcePracticeTarget(guide.slug)!
      expect(href.startsWith('/practice?')).toBe(true)
      expect(parseChapterPracticeTarget(href.slice('/practice'.length))).toEqual(target)
      expect(new URL(href, 'https://internal.invalid').searchParams.has('concept')).toBe(false)
      expect(resolveChapterPracticeTarget(target, target.subject, [{id:'right',sequence_number:target.chapterNumber}])?.id).toBe('right')
      expect(resolveChapterPracticeTarget(target, target.subject, [])).toBeNull()
      expect(resolveChapterPracticeTarget(target, target.subject, [{id:'a',sequence_number:target.chapterNumber},{id:'b',sequence_number:target.chapterNumber}])).toBeNull()
    }
  })
  it('never guesses from a lookalike or unknown slug', () => {
    for (const slug of ['', 'science-9', 'class-8-science-unknown-study-resources', resourceGuides[0].slug + '-extra']) expect(resourcePracticeHref(slug)).toBeNull()
  })
  it('wires both the closing CTA and in-article practice links without replacing concept navigation', () => {
    const source = readFileSync(new URL('../pages/BlogPostPage.tsx', import.meta.url), 'utf8')
    expect(source).toContain('if (href === \'/practice\' && resourceSetupHref)')
    expect(source).toContain('to={resourceSetupHref || (conceptPracticeReady')
    expect(source).toContain('Set up this chapter’s practice')
    expect(source).toContain("if (href === '/practice' && conceptPracticeReady && blogPracticeTargets[post.slug])")
  })
})
