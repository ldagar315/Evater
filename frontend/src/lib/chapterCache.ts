// Memory only: no cross-account reuse, no persistent copy, five-minute freshness.
export function createChapterCache<T>(ttlMs = 5 * 60_000, now = () => Date.now()) {
  let owner: string | null = null
  const entries = new Map<string, { expires: number; value?: T; pending?: Promise<T> }>()
  return async (userId: string, subject: string, load: () => Promise<T>): Promise<T> => {
    if (owner !== userId) { entries.clear(); owner = userId }
    const existing = entries.get(subject)
    if (existing?.pending) return existing.pending
    if (existing?.value !== undefined && existing.expires > now()) return existing.value
    const entry: { expires: number; value?: T; pending?: Promise<T> } = { expires: 0 }
    entry.pending = load().then((value) => {
      entry.value = value; entry.expires = now() + ttlMs; entry.pending = undefined
      return value
    }).catch((error) => { if (entries.get(subject) === entry) entries.delete(subject); throw error })
    entries.set(subject, entry)
    return entry.pending
  }
}
