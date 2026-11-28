import { describe, expect, it } from 'vitest'
import { blogPosts } from './blogPosts'
import { resourceGuides } from './resourceGuides'
import { paperDetails, paperForDetail } from './paperDetails'
import { getStudyGuidePaper } from './studyGuidePapers'
import { readFileSync } from 'node:fs'

describe('direct chapter guide to paper discovery links', () => {
  it('maps every resource guide to exactly its catalog chapter', () => {
    expect(resourceGuides).toHaveLength(27)
    for (const guide of resourceGuides) {
      const paper = getStudyGuidePaper(guide.slug)
      expect(paper, guide.slug).toBeDefined()
      const [, subject, chapter] = guide.id.split('-')
      const catalog = paperForDetail(paper!)
      expect(catalog.subject).toBe(subject === 'maths' ? 'Mathematics' : 'Science')
      const key = { pythagoras: 9, electricity: 4, light: 10 }[paper!.paperId]
      expect(key || Number(paper!.paperId.split('ch')[1])).toBe(Number(chapter))
    }
  })
  it('preserves all reverse paper-guide references and high-impression guides', () => {
    for (const paper of paperDetails.filter(p => p.studyGuide)) {
      const slug = paper.studyGuide!.href.split('/')[2]
      expect(blogPosts.some(p => p.slug === slug)).toBe(true)
      expect(getStudyGuidePaper(slug)?.paperId).toBe(paper.paperId)
    }
    expect(getStudyGuidePaper('ncert-class-8-science-elements-compounds-mixtures-study-guide')?.paperId).toBe('science-ch08')
    expect(getStudyGuidePaper('ncert-class-8-science-keeping-time-skies-study-guide')?.paperId).toBe('science-ch11')
  })
  it('covers all 44 published chapter guides but leaves general study posts alone', () => {
    const guides = blogPosts.filter(post => getStudyGuidePaper(post.slug))
    expect(guides).toHaveLength(44)
  })
  it('does not guess a chapter for general or unknown posts', () => {
    for (const slug of ['active-recall-class-6-10-exams', 'unknown', 'class-8-maths-not-a-chapter']) expect(getStudyGuidePaper(slug)).toBeUndefined()
  })
  it('renders an unconditional crawlable link, not a button or auth-dependent fetch', () => {
    const page = readFileSync(new URL('../pages/BlogPostPage.tsx', import.meta.url), 'utf8')
    expect(page).toContain('to={chapterPaper.path}')
    expect(page).toContain('aria-label="Chapter sample paper"')
    expect(page.indexOf('aria-label="Chapter sample paper"')).toBeLessThan(page.indexOf('<Markdown components='))
  })
})
