export type PracticeTarget = { subject: 'science' | 'mathematics'; chapter: string; concept: string }
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function parsePracticeTarget(search: string): { provided: boolean; target: PracticeTarget | null } {
  const p = new URLSearchParams(search)
  if ([...p.keys()].length === 0) return { provided: false, target: null }
  const subject = p.get('subject'), chapter = p.get('chapter'), concept = p.get('concept')
  const valid = [...p.keys()].every(k => ['subject', 'chapter', 'concept'].includes(k) && p.getAll(k).length === 1)
    && (subject === 'science' || subject === 'mathematics') && uuid.test(chapter || '') && uuid.test(concept || '')
  return { provided: true, target: valid ? { subject, chapter, concept } as PracticeTarget : null }
}

export function practiceHref(target: PracticeTarget): string {
  return '/practice?' + new URLSearchParams({ subject: target.subject, chapter: target.chapter, concept: target.concept }).toString()
}

// No general-purpose redirect. Only the internal practice setup may be resumed.
export function safePracticeReturn(value: string | null): string | null {
  if (!value || value.length > 2000 || /[\\#\r\n]/.test(value) || !/^\/practice(?:\?|$)/.test(value)) return null
  const url = new URL(value, 'https://internal.invalid')
  return url.origin === 'https://internal.invalid' && url.pathname === '/practice' ? url.pathname + url.search : null
}

export function authReturn(search: string): string {
  return safePracticeReturn(new URLSearchParams(search).get('next')) || '/home'
}

export function authHref(path: string, destination: string): string {
  const next = safePracticeReturn(destination)
  return next ? path + '?next=' + encodeURIComponent(next) : path
}
