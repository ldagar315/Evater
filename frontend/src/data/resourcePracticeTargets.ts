import { resourceGuides } from './resourceGuides'
import { chapterPracticeHref, type ChapterPracticeTarget } from '../lib/chapterPracticeTarget'

/** Chapter setup only, never a claim that a particular concept or question is available. */
export function resourcePracticeTarget(slug: string): ChapterPracticeTarget | null {
  const guide = resourceGuides.find(guide => guide.slug === slug)
  const match = guide?.id.match(/^resource-(maths|science)-([1-9][0-9]?)$/)
  if (!match) return null
  const subject = match[1] === 'maths' ? 'mathematics' : 'science'
  const chapterNumber = Number(match[2])
  if (chapterNumber > (subject === 'science' ? 13 : 14)) return null
  return { subject, chapterNumber }
}

export function resourcePracticeHref(slug: string): string | null {
  const target = resourcePracticeTarget(slug)
  return target ? chapterPracticeHref(target) : null
}
