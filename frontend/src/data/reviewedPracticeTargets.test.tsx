import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { StaticRouter } from 'react-router-dom/server'
import { ChapterConceptLinks } from '../components/practice/ChapterConceptLinks'
import { reviewedPracticeTargets } from './reviewedPracticeTargets'
import { blogPracticeTargets } from './blogPracticeTargets'
import { parsePracticeTarget, practiceHref } from '../lib/practiceTarget'

describe('reviewed release concept discovery', () => {
  it('contains exactly nine unique, valid navigation targets, not question counts', () => {
    expect(reviewedPracticeTargets).toHaveLength(9)
    expect(new Set(reviewedPracticeTargets.map(t => t.concept)).size).toBe(9)
    for (const target of reviewedPracticeTargets) {
      expect(parsePracticeTarget(practiceHref(target).slice('/practice'.length)).target).toEqual({ subject: target.subject, chapter: target.chapter, concept: target.concept })
      expect(Object.keys(target).sort()).toEqual(['chapter', 'concept', 'label', 'subject'])
    }
  })
  it('renders only the matching chapter and subject as links, never start buttons', () => {
    const target = reviewedPracticeTargets.find(t => t.label === 'Estimating roots')!
    const html = renderToStaticMarkup(<StaticRouter location="/practice"><ChapterConceptLinks subject={target.subject} chapter={target.chapter} /></StaticRouter>)
    expect(html).toContain('Estimating roots')
    expect(html).toContain('Unit digits of squares and cubes')
    expect(html).not.toContain('Contact forces')
    expect(html).not.toContain('<button')
    expect(html.match(/<a /g)).toHaveLength(2)
    expect(html).toContain('Your test starts only when you choose Start.')
    expect(renderToStaticMarkup(<StaticRouter location="/practice"><ChapterConceptLinks subject="science" chapter={target.chapter} /></StaticRouter>)).toBe('')
    expect(renderToStaticMarkup(<StaticRouter location="/practice"><ChapterConceptLinks subject="mathematics" chapter="unknown" /></StaticRouter>)).toBe('')
  })
  it('adds the four newly available topics to existing articles without removing their old targets', () => {
    expect(blogPracticeTargets['ncert-class-8-maths-square-and-cube-study-guide'].map(t => t.label)).toEqual(['Perfect squares', 'Perfect cubes', 'Cube roots', 'Unit digits of squares and cubes', 'Estimating roots'])
    expect(blogPracticeTargets['ncert-class-8-science-exploring-forces-study-guide'].map(t => t.label)).toEqual(['Balanced forces', 'Friction', 'Gravity', 'Contact forces', 'Non-contact forces'])
    expect(Object.keys(blogPracticeTargets)).toHaveLength(4)
  })
})
