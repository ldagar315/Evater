export type ChapterPracticeTarget = { subject: 'science' | 'mathematics'; chapterNumber: number }

/** A separate setup-only intent. Never interpret a broken concept URL as chapter practice. */
export function parseChapterPracticeTarget(search: string): ChapterPracticeTarget | null {
  const params = new URLSearchParams(search)
  if (params.size !== 2 || [...params.keys()].some(k => !['subject', 'chapterNumber'].includes(k))) return null
  const subject = params.get('subject')
  const number = params.get('chapterNumber') || ''
  if ((subject !== 'science' && subject !== 'mathematics') || !/^[1-9][0-9]?$/.test(number)) return null
  if (params.getAll('subject').length !== 1 || params.getAll('chapterNumber').length !== 1) return null
  const chapterNumber = Number(number)
  return chapterNumber <= (subject === 'science' ? 13 : 14) ? { subject, chapterNumber } : null
}

export function chapterPracticeHref(target: ChapterPracticeTarget): string {
  return '/practice?' + new URLSearchParams({ subject: target.subject, chapterNumber: String(target.chapterNumber) }).toString()
}

/** Require one live catalog match. Missing/duplicate chapter numbers must not pick a default. */
export function resolveChapterPracticeTarget<T extends { id: string; sequence_number: number }>(
  target: ChapterPracticeTarget, subject: string, chapters: T[],
): T | null {
  if (subject !== target.subject) return null
  const matches = chapters.filter(chapter => chapter.sequence_number === target.chapterNumber)
  return matches.length === 1 ? matches[0] : null
}
