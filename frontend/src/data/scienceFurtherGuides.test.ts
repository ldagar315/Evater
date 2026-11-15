import { describe, it, expect } from 'vitest'
import { scienceFurtherGuides } from './scienceFurtherGuides'
import { blogPosts, getPostBySlug } from './blogPosts'
import { blogPracticeTargets } from './blogPracticeTargets'
describe('Curiosity remaining eight chapter guides', () => {
  it('adds eight unique chapter articles without replacing existing Forces or Light', () => {
    expect(scienceFurtherGuides).toHaveLength(8)
    expect(new Set(blogPosts.map(p => p.id)).size).toBe(blogPosts.length)
    expect(new Set(blogPosts.map(p => p.slug)).size).toBe(blogPosts.length)
    for (const slug of ['ncert-class-8-science-exploring-forces-study-guide','ncert-class-8-science-light-mirrors-lenses-study-guide']) expect(getPostBySlug(slug)).toBeDefined()
  })
  it('links full sources, original diagnostics and honest generic practice', () => {
    scienceFurtherGuides.forEach((post,i) => {
      expect(post.title).toContain(`Chapter ${[4,6,7,8,9,11,12,13][i]}:`)
      expect(getPostBySlug(post.slug)).toBe(post)
      expect(post.content).toContain(`hecu1${String([4,6,7,8,9,11,12,13][i]).padStart(2,"0")}.pdf`)
      expect(post.content).toContain('Answer check:')
      expect(post.content).toContain('not verified')
      expect(post.content).toContain('/blog/diagrams/')
      expect(post.content.split(/\s+/).length).toBeGreaterThan(700)
      expect(blogPracticeTargets[post.slug]).toBeUndefined()
    })
  })
  it('keeps final-chapter concepts and safety distinct', () => {
    expect(scienceFurtherGuides[5].content).toContain('Never look directly at the Sun')
    expect(scienceFurtherGuides[5].content).toContain('divisible by 400')
    expect(scienceFurtherGuides[6].content).toContain('food to the organism eating it')
    expect(scienceFurtherGuides[6].content).toContain('not exact population forecasts')
    expect(scienceFurtherGuides[7].content).toContain('an ovule is not simply another name for an egg cell')
    expect(scienceFurtherGuides[7].content).toContain('well below 2 degrees Celsius')
  })
  it('preserves safety and particle-model boundaries', () => {
    expect(scienceFurtherGuides[0].content).toContain('Never connect classroom wires to mains sockets')
    expect(scienceFurtherGuides[0].content).toContain('must not be recharged')
    expect(scienceFurtherGuides[1].content).toContain('Do not open windows')
    expect(scienceFurtherGuides[1].content).toContain('weather.gov/safety/lightning-crouch')
    expect(scienceFurtherGuides[2].content).toContain('ice has a more open arrangement')
    expect(scienceFurtherGuides[2].content).toContain('Never taste an unknown solution')
  })
})
