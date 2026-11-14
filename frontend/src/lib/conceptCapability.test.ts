import { afterEach, describe, expect, it, vi } from 'vitest'
vi.mock('./supabase', () => ({ supabase: {} }))
import { supportsConceptPractice } from './api'
afterEach(() => vi.unstubAllGlobals())
describe('deployment capability', () => {
  it('only accepts the current explicit contract without caching readiness', async () => {
    const fetch = vi.fn().mockResolvedValueOnce({ ok: true, json: async () => ({ concept_test_contract: 1 }) })
      .mockResolvedValueOnce({ ok: false }).mockResolvedValueOnce({ ok: true, json: async () => ({ concept_test_contract: 2 }) })
    vi.stubGlobal('fetch', fetch)
    expect(await supportsConceptPractice()).toBe(true)
    expect(await supportsConceptPractice()).toBe(false)
    expect(await supportsConceptPractice()).toBe(false)
    expect(fetch).toHaveBeenCalledTimes(3)
    expect(fetch.mock.calls[0][1].cache).toBe('no-store')
  })
  it('fails closed for malformed, absent, rejected and broken responses', async () => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    for (const value of [null, {}, { concept_test_contract: '1' }, { concept_test_contract: true }, []]) {
      fetch.mockResolvedValueOnce({ ok: true, json: async () => value })
      expect(await supportsConceptPractice()).toBe(false)
    }
    fetch.mockRejectedValueOnce(new Error('network unavailable'))
    expect(await supportsConceptPractice()).toBe(false)
    fetch.mockResolvedValueOnce({ ok: true, json: async () => { throw new Error('bad json') } })
    expect(await supportsConceptPractice()).toBe(false)
  })
})
