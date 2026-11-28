import { resourceGuides } from './resourceGuides'
import { paperDetails, paperForStudyGuide, type PaperDetail } from './paperDetails'

export function getStudyGuidePaper(slug: string): PaperDetail | undefined {
  const exact = paperForStudyGuide(slug)
  if (exact) return exact
  const guide = resourceGuides.find(guide => guide.slug === slug)
  const match = guide?.id.match(/^resource-(maths|science)-(\d+)$/)
  if (!match) return undefined
  const [, subject, chapter] = match
  const legacy: Record<string, string> = { 'maths-9': 'pythagoras', 'science-4': 'electricity', 'science-10': 'light' }
  const paperId = legacy[`${subject}-${chapter}`] || `${subject}-ch${chapter.padStart(2, '0')}`
  return paperDetails.find(paper => paper.paperId === paperId)
}
