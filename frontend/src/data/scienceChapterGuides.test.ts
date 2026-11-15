import { describe, it, expect } from 'vitest'
import { scienceChapterGuides } from './scienceChapterGuides'
import { blogPosts, getPostBySlug } from './blogPosts'
import { blogPracticeTargets } from './blogPracticeTargets'
describe('first three Curiosity chapter guides', () => {
  it('adds exactly three unique chapters without replacing the two existing Science chapter articles', () => {
    expect(scienceChapterGuides).toHaveLength(3)
    expect(new Set(blogPosts.map(p => p.id)).size).toBe(blogPosts.length)
    expect(new Set(blogPosts.map(p => p.slug)).size).toBe(blogPosts.length)
    for (const slug of ['ncert-class-8-science-exploring-forces-study-guide', 'ncert-class-8-science-light-mirrors-lenses-study-guide']) expect(getPostBySlug(slug)).toBeDefined()
  })
  it('has discoverable articles, original diagnostics, sources, local illustrations and honest practice copy', () => {
    scienceChapterGuides.forEach((post,index) => {
      expect(getPostBySlug(post.slug)).toBe(post)
      expect(post.title).toContain(`Chapter ${index+1}:`)
      expect(post.content).toContain('Answer check:')
      expect(post.content).toContain(`hecu10${index+1}.pdf`)
      expect(post.content).toContain('not verified')
      expect(post.content).toContain('/blog/diagrams/')
      expect(post.featured_image.startsWith('/blog/diagrams/')).toBe(true)
      expect(blogPracticeTargets[post.slug]).toBeUndefined()
      expect(post.content.split(/\s+/).length).toBeGreaterThan(550)
    })
  })
  it('keeps health boundaries and experimental examples explicit', () => {
    expect(scienceChapterGuides[0].content).toContain('made-up results')
    expect(scienceChapterGuides[1].content).toContain('do not open, smell closely or taste mouldy food')
    expect(scienceChapterGuides[2].content).toContain('not a diagnosis or treatment plan')
    expect(scienceChapterGuides[2].content).toContain('bacteria can survive')
  })
})
