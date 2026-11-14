import { describe, it, expect } from 'vitest'
import { parsePracticeTarget, practiceHref, safePracticeReturn, authReturn, authHref } from './practiceTarget'
import { blogPracticeTargets } from '../data/blogPracticeTargets'
describe('article practice targets', () => {
  it('round trips all metadata without leaking labels into parameters', () => {
    expect(Object.keys(blogPracticeTargets)).toHaveLength(4)
    for (const targets of Object.values(blogPracticeTargets)) for (const t of targets) {
      const path = practiceHref(t)
      expect(parsePracticeTarget(path.slice(path.indexOf('?')))).toEqual({ provided: true, target: { subject: t.subject, chapter: t.chapter, concept: t.concept } })
      expect(authReturn(new URL(authHref('/auth', path), 'https://local').search)).toBe(path)
    }
  })
  it('separates no target from an invalid target without defaulting', () => {
    expect(parsePracticeTarget('')).toEqual({ provided: false, target: null })
    for (const search of ['?chapter=bad','?concept=x','?subject=science','?subject=english&chapter=x&concept=y', '?foo=1']) expect(parsePracticeTarget(search)).toEqual({ provided: true, target: null })
    const path = practiceHref(blogPracticeTargets[Object.keys(blogPracticeTargets)[0]][0])
    expect(parsePracticeTarget(path.slice(path.indexOf('?')) + '&subject=science').target).toBeNull()
  })
  it('never accepts external or unrelated return destinations', () => {
    for (const path of ['https://evil.invalid', '//evil.invalid/practice', '/home', '/practice/other', '/practice#bad', '/practice\\evil', '/practice\n?x=1', 'javascript:alert(1)', '/%70ractice']) expect(safePracticeReturn(path)).toBeNull()
    expect(authReturn('?next=https://evil.invalid')).toBe('/home')
    expect(safePracticeReturn('/practice?concept=bad')).toBe('/practice?concept=bad')
  })
})
