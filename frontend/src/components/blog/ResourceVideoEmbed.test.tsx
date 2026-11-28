import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { ResourceVideoEmbed } from './ResourceVideoEmbed'

const render = (videoUrl: string) => renderToStaticMarkup(<ResourceVideoEmbed videoUrl={videoUrl} title="Geometry & reasoning" />)

describe('resource video player and independent fallback', () => {
  it('keeps a privacy-enhanced lazy iframe as the primary lesson player', () => {
    const html = render('https://www.youtube.com/watch?v=abcdefghijk&t=20')
    expect(html).toContain('src="https://www.youtube-nocookie.com/embed/abcdefghijk"')
    expect(html).toContain('loading="lazy"')
    expect(html).toContain('allowfullscreen=""')
    expect(html).toContain('referrerPolicy="strict-origin-when-cross-origin"')
    expect(html).toContain('Geometry &amp; reasoning')
  })

  it('renders the visible fallback after and outside the iframe, without a load-event dependency', () => {
    const html = render('https://www.youtube.com/watch?v=abcdefghijk&t=20')
    expect(html.indexOf('Video not loading?')).toBeGreaterThan(html.indexOf('</iframe>'))
    expect(html).toContain('href="https://www.youtube.com/watch?v=abcdefghijk&amp;t=20"')
    expect(html).toContain('Watch on YouTube')
    expect(html).toContain('target="_blank"')
    expect(html).toContain('rel="noopener noreferrer"')
    expect(html).toContain('(opens in a new tab)')
    expect(html).not.toContain('hidden=')
    expect(html).not.toContain('onload=')
  })

  it('preserves each allowlisted original destination for the fallback', () => {
    for (const url of ['https://youtu.be/abcdefghijk', 'https://youtube.com/embed/abcdefghijk', 'https://m.youtube.com/shorts/abcdefghijk']) {
      const html = render(url)
      expect(html).toContain(`href="${url}"`)
      expect(html).toContain('https://www.youtube-nocookie.com/embed/abcdefghijk')
    }
  })

  it('does not create a player or fallback for arbitrary, malformed or unsafe destinations', () => {
    for (const url of ['https://youtube.com.evil.test/watch?v=abcdefghijk', 'javascript:alert(1)', 'http://youtube.com/watch?v=abcdefghijk', 'https://youtube.com/watch?v=bad', '/practice']) {
      expect(render(url)).toBe('')
    }
  })
})
