import { describe, it, expect } from 'vitest'
import { publicPagePath } from './seoConfig'
describe('public page URL convention', () => {
  it('matches directory-index public pages and stays idempotent', () => {
    for (const path of ['/about', '/blog', '/blog/example']) {
      expect(publicPagePath(path)).toBe(path + '/')
      expect(publicPagePath(path + '/')).toBe(path + '/')
    }
  })
  it('preserves category parameters and About section fragments', () => {
    expect(publicPagePath('/blog?category=study-skills')).toBe('/blog/?category=study-skills')
    expect(publicPagePath('/about#privacy')).toBe('/about/#privacy')
  })
  it('does not alter app routes, root, assets or look-alike paths', () => {
    for (const path of ['/', '/home', '/profile', '/my-tests', '/about-us', '/blog-image.svg', '/blog/diagrams/example.svg']) expect(publicPagePath(path)).toBe(path)
  })
})
