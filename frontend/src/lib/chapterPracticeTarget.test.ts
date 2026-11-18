import { describe, expect, it } from 'vitest'
import { chapterPracticeHref, parseChapterPracticeTarget, resolveChapterPracticeTarget } from './chapterPracticeTarget'
import { authHref, authReturn, parsePracticeTarget } from './practiceTarget'
import { samplePapers } from '../data/samplePapers'
import { paperDetails, paperPracticeHref } from '../data/paperDetails'

describe('paper chapter setup intent', () => {
  it('maps all27chapter papers to exact subject/sequence and keeps mixed papers general', () => {
    expect(paperDetails.filter(d => d.studyGuide)).toHaveLength(27)
    for (const paper of samplePapers) {
      const path = paperPracticeHref(paper.id)!
      if (paper.format === 'mixed') { expect(path).toBe('/practice'); continue }
      const parsed = parseChapterPracticeTarget(path.slice(path.indexOf('?')))
      const legacy: Record<string, number> = { electricity: 4, light: 10, pythagoras: 9 }
      const sequence = legacy[paper.id] || Number(paper.id.split('-ch')[1])
      expect(parsed).toEqual({ subject: paper.subject === 'Science' ? 'science' : 'mathematics', chapterNumber: sequence })
      expect(authReturn(new URL(authHref('/auth', path), 'https://local').search)).toBe(path)
    }
    expect(paperPracticeHref('unknown')).toBeUndefined()
  })
  it('roundtrips every allowed chapter and never changes concept-only contract', () => {
    for (const subject of ['science', 'mathematics'] as const) for (let chapterNumber = 1; chapterNumber <= (subject === 'science' ? 13 : 14); chapterNumber++) {
      const target = { subject, chapterNumber }
      const query = chapterPracticeHref(target).split('?')[1]
      expect(parseChapterPracticeTarget(query)).toEqual(target)
      expect(parsePracticeTarget(query)).toEqual({ provided: true, target: null })
    }
  })
  it('rejects missing, duplicate, malformed, unsupported, concept or extra parameters', () => {
    for (const query of ['', '?subject=science', '?chapterNumber=1', '?subject=science&chapterNumber=14', '?subject=mathematics&chapterNumber=15', '?subject=english&chapterNumber=1', '?subject=science&chapterNumber=0', '?subject=science&chapterNumber=01', '?subject=science&chapterNumber=1.5', '?subject=science&chapterNumber=-1', '?subject=science&chapterNumber=1e1', '?subject=science&chapterNumber=1&subject=science', '?subject=science&chapterNumber=1&chapterNumber=1', '?subject=science&chapterNumber=1&concept=bad', '?subject=science&chapter=bad', '?subject=science&chapterNumber=1&foo=x']) expect(parseChapterPracticeTarget(query), query).toBeNull()
  })
  it('resolves only one live subject-catalog chapter, never first-match fallback', () => {
    const target = { subject: 'science', chapterNumber: 4 } as const
    const chapters = [{ id: 'wrong', sequence_number: 1 }, { id: 'right', sequence_number: 4 }]
    expect(resolveChapterPracticeTarget(target, 'science', chapters)?.id).toBe('right')
    expect(resolveChapterPracticeTarget(target, 'mathematics', chapters)).toBeNull()
    expect(resolveChapterPracticeTarget(target, 'science', [])).toBeNull()
    expect(resolveChapterPracticeTarget(target, 'science', chapters.slice(0, 1))).toBeNull()
    expect(resolveChapterPracticeTarget(target, 'science', [...chapters, { id: 'duplicate', sequence_number: 4 }])).toBeNull()
  })
})
