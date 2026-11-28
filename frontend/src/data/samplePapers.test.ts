import { describe, expect, it } from 'vitest'
import { filterPapers, samplePapers } from './samplePapers'
const empty = { query: '', classLevel: '', subject: '', topic: '' }
describe('paper discovery', () => {
  it('lists 27 chapter and 2 mixed-chapter papers', () => expect(filterPapers(samplePapers, empty)).toHaveLength(29))
  it('combines class, subject and topic filters', () => expect(filterPapers(samplePapers, { ...empty, classLevel: '8', subject: 'Science', topic: 'Light' }).map(p => p.id)).toEqual(['light']))
  it('finds topics, synonyms, class and subject with case-insensitive terms', () => {
    for (const query of ['LIGHT mirrors', 'class 8 reflection', 'science lenses']) expect(filterPapers(samplePapers, { ...empty, query }).map(p => p.id)).toEqual(['light'])
    expect(filterPapers(samplePapers, { ...empty, query: 'Baudhayana' }).map(p => p.id)).toEqual(['pythagoras'])
    expect(filterPapers(samplePapers, { ...empty, query: 'magnetism' }).map(p => p.id)).toEqual(['electricity'])
  })
  it('handles whitespace, no match and inconsistent filters honestly', () => {
    expect(filterPapers(samplePapers, { ...empty, query: '   ' })).toHaveLength(29)
    expect(filterPapers(samplePapers, { ...empty, query: 'quantum gravity' })).toHaveLength(0)
    expect(filterPapers(samplePapers, { ...empty, subject: 'Mathematics', topic: 'Light' })).toHaveLength(0)
    expect(filterPapers(samplePapers, { ...empty, classLevel: '9' })).toHaveLength(0)
  })
  it('keeps original static PDF filenames and metadata', () => {
    expect(new Set(samplePapers.map(p => p.href)).size).toBe(29)
    expect(samplePapers.filter(p => p.format !== 'mixed').every(p => p.href.startsWith('/') && p.href.endsWith('.pdf') && p.marks === 22 && p.questions === 15)).toBe(true)
  })
})

describe('mixed-chapter format', () => {
  it('keeps chapter papers unchanged and gives mixed papers their own correct counts', () => {
    const mixed = samplePapers.filter(p => p.format === 'mixed')
    expect(mixed).toHaveLength(2)
    expect(new Set(mixed.map(p => p.subject)).size).toBe(2)
    for (const paper of mixed) {
      expect(paper.questions).toBe(24)
      expect(paper.marks).toBe(60)
      expect(paper.minutes).toBe(150)
      expect(paper.objectiveQuestions).toBe(12)
      expect(paper.writtenQuestions).toBe(12)
      expect(paper.objectiveQuestions! + paper.writtenQuestions!).toBe(paper.questions)
    }
    expect(samplePapers.filter(p => p.format !== 'mixed')).toHaveLength(27)
    expect(filterPapers(samplePapers, { ...empty, query: 'mixed' })).toHaveLength(2)
  })
})
