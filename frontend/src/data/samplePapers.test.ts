import { describe, expect, it } from 'vitest'
import { filterPapers, samplePapers } from './samplePapers'
const empty = { query: '', classLevel: '', subject: '', topic: '' }
describe('paper discovery', () => {
  it('lists all27catalog chapter papers', () => expect(filterPapers(samplePapers, empty)).toHaveLength(27))
  it('combines class, subject and topic filters', () => expect(filterPapers(samplePapers, { ...empty, classLevel: '8', subject: 'Science', topic: 'Light' }).map(p => p.id)).toEqual(['light']))
  it('finds topics, synonyms, class and subject with case-insensitive terms', () => {
    for (const query of ['LIGHT mirrors', 'class 8 reflection', 'science lenses']) expect(filterPapers(samplePapers, { ...empty, query }).map(p => p.id)).toEqual(['light'])
    expect(filterPapers(samplePapers, { ...empty, query: 'Baudhayana' }).map(p => p.id)).toEqual(['pythagoras'])
    expect(filterPapers(samplePapers, { ...empty, query: 'magnetism' }).map(p => p.id)).toEqual(['electricity'])
  })
  it('handles whitespace, no match and inconsistent filters honestly', () => {
    expect(filterPapers(samplePapers, { ...empty, query: '   ' })).toHaveLength(27)
    expect(filterPapers(samplePapers, { ...empty, query: 'quantum gravity' })).toHaveLength(0)
    expect(filterPapers(samplePapers, { ...empty, subject: 'Mathematics', topic: 'Light' })).toHaveLength(0)
    expect(filterPapers(samplePapers, { ...empty, classLevel: '9' })).toHaveLength(0)
  })
  it('keeps original static PDF filenames and metadata', () => {
    expect(new Set(samplePapers.map(p => p.href)).size).toBe(27)
    expect(samplePapers.every(p => p.href.startsWith('/') && p.href.endsWith('.pdf') && p.marks === 22 && p.questions === 15)).toBe(true)
  })
})
