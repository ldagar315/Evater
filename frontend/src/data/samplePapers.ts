export interface SamplePaper {
  id: string
  title: string
  subtitle: string
  classLevel: number
  subject: 'Science' | 'Mathematics'
  topic: string
  keywords: string[]
  href: string
  questions: number
  marks: number
  minutes: number
}

/** Discovery metadata only. The approved PDF files and their contents are unchanged. */
export const samplePapers: SamplePaper[] = [
  { id: 'light', title: 'Light', subtitle: 'Mirrors and lenses', classLevel: 8, subject: 'Science', topic: 'Light', keywords: ['reflection', 'refraction', 'optics', 'convex', 'concave', 'mirrors', 'lenses'], href: '/1-class-8-light-sample-paper.pdf', questions: 15, marks: 22, minutes: 45 },
  { id: 'electricity', title: 'Electricity', subtitle: 'Magnetic and heating effects', classLevel: 8, subject: 'Science', topic: 'Electricity', keywords: ['magnetism', 'electromagnet', 'current', 'circuits', 'heating', 'cells'], href: '/2-class-8-electricity-sample-paper.pdf', questions: 15, marks: 22, minutes: 45 },
  { id: 'pythagoras', title: 'The Pythagoras theorem', subtitle: 'The Baudhayana-Pythagoras theorem', classLevel: 8, subject: 'Mathematics', topic: 'Pythagoras', keywords: ['maths', 'math', 'right triangle', 'triangles', 'hypotenuse', 'geometry', 'Baudhayana'], href: '/3-class-8-pythagoras-sample-paper.pdf', questions: 15, marks: 22, minutes: 45 },
]

export interface PaperFilters { query: string; classLevel: string; subject: string; topic: string }
export function filterPapers(papers: SamplePaper[], filters: PaperFilters) {
  const words = filters.query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  return papers.filter(paper => {
    const text = [paper.title, paper.subtitle, paper.topic, paper.subject, `class ${paper.classLevel}`, ...paper.keywords].join(' ').toLocaleLowerCase()
    return (!filters.classLevel || String(paper.classLevel) === filters.classLevel)
      && (!filters.subject || paper.subject === filters.subject)
      && (!filters.topic || paper.topic === filters.topic)
      && words.every(word => text.includes(word))
  })
}
