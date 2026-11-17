/** Accept video links only, never arbitrary iframe URLs from article content. */
export function youtubeVideoId(value: string): string | null {
  try {
    const url = new URL(value)
    if (url.protocol !== 'https:') return null
    let id: string | null = null
    if (url.hostname === 'youtu.be') id = url.pathname.slice(1)
    else if (['youtube.com', 'www.youtube.com', 'm.youtube.com'].includes(url.hostname)) {
      if (url.pathname === '/watch') id = url.searchParams.get('v')
      else if (/^\/(embed|shorts)\//.test(url.pathname)) id = url.pathname.split('/')[2]
    }
    return id && /^[A-Za-z0-9_-]{11}$/.test(id) ? id : null
  } catch { return null }
}
