import { renderToStaticMarkup } from 'react-dom/server'
import { readFileSync } from 'node:fs'
import { describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({ user: null as null | { id: string }, seo: [] as Record<string, any>[] }))
vi.mock('../contexts/AuthContext', () => ({ useAuthContext: () => ({ user: state.user }) }))
vi.mock('../components/layout/Header', () => ({ Header: () => null }))
vi.mock('../components/layout/Footer', () => ({ Footer: () => null }))
vi.mock('../components/blog/BlogSection', () => ({ BlogSection: () => null }))
vi.mock('../components/seo/Seo', () => ({ Seo: (props: Record<string, any>) => { state.seo.push(props); return null } }))
vi.mock('react-router-dom', () => ({ Link: ({ to, children, ...props }: any) => <a href={to} {...props}>{children}</a> }))
import { LandingPage } from './LandingPage'
import { AboutPage } from './AboutPage'

function render(Page: typeof LandingPage, signedIn = false) {
  state.user = signedIn ? { id: 'synthetic-local-user' } : null
  state.seo = []
  const html = renderToStaticMarkup(<Page />)
  expect(state.seo).toHaveLength(1)
  return { html, seo: state.seo[0] }
}

describe('current chapter-practice coverage, separate from intended audience', () => {
  it('keeps homepage badge, description, keywords and website schema scoped to Class 8', () => {
    const { html, seo } = render(LandingPage)
    expect(html).toContain('Class 8 practice')
    expect(html).not.toContain('Class 6–10 practice')
    expect(seo.description).toBe('Evater helps Class 8 students practise Maths and Science, understand feedback, and decide what to review next.')
    expect(seo.keywords).toContain('Class 8 practice')
    expect(seo.keywords).not.toContain('Class 6–10 practice')
    const website = seo.jsonLd['@graph'].find((node: any) => node['@type'] === 'WebSite')
    expect(website.description).toBe('A focused test, feedback, and retry loop for Class 8 Maths and Science.')
  })

  it('says current coverage in visible About copy and its FAQ structured data', () => {
    const { html, seo } = render(AboutPage)
    const scope = 'Chapter practice currently covers Class 8 Maths and Science.'
    expect(html).toContain(scope)
    expect(html).toContain('Evater is being built for students in Classes 6–10')
    expect(seo.description).toBe('Learn how Evater helps Class 8 students practise Maths and Science, understand mistakes, and improve through better feedback.')
    expect(seo.title).not.toContain('Classes 6–10')
    expect(seo.keywords).not.toContain('Class 6-10 practice')
    const graph = seo.jsonLd['@graph']
    expect(graph.find((node: any) => node['@type'] === 'AboutPage').description).toBe(seo.description)
    expect(graph.find((node: any) => node['@type'] === 'FAQPage').mainEntity[0].acceptedAnswer.text).toContain(scope)
  })

  it.each([false, true])('does not change practice-entry routing (signed in: %s)', (signedIn) => {
    for (const Page of [LandingPage, AboutPage]) {
      const { html } = render(Page, signedIn)
      const expectedPath = signedIn ? '/practice' : Page === LandingPage ? '/auth?mode=signup' : '/auth'
      expect(html).toContain(`href="${expectedPath}"`)
    }
  })

  it('keeps no-JavaScript description, Open Graph and Twitter fallback metadata consistent', () => {
    const html = readFileSync(new URL('../../index.html', import.meta.url), 'utf8')
    const expected = render(LandingPage).seo.description
    for (const selector of ['name="description"', 'property="og:description"', 'name="twitter:description"']) {
      const tag = html.match(new RegExp(`<meta ${selector} content="([^"]+)"`))
      expect(tag?.[1]).toBe(expected)
    }
    expect(html).not.toContain('Classes 6–10 take tests')
  })
})
