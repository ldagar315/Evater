import { beforeEach, describe, expect, it, vi } from 'vitest'
const auth = vi.hoisted(() => ({ signOut: vi.fn(), getUser: vi.fn() }))
vi.mock('./supabase', () => ({ supabase: { auth } }))
import { signOutSession, validateSessionUser } from './sessionRecovery'
const missing = { name: 'AuthSessionMissingError', status: 400 }
describe('revoked session recovery', () => {
  beforeEach(() => vi.resetAllMocks())
  it('keeps normal sign-out unchanged', async () => {
    auth.signOut.mockResolvedValue({ error: null })
    expect(await signOutSession()).toEqual({ error: null })
    expect(auth.getUser).not.toHaveBeenCalled()
  })
  it('lets SDK getUser clean up only confirmed missing sessions', async () => {
    auth.signOut.mockResolvedValue({ error: missing })
    auth.getUser.mockResolvedValue({ data: { user: null }, error: missing })
    expect(await signOutSession()).toEqual({ error: null })
    expect(auth.getUser).toHaveBeenCalledOnce()
  })
  it('does not turn a transient sign-out error into success', async () => {
    const error = { name: 'AuthRetryableFetchError', status: 503 }
    auth.signOut.mockResolvedValue({ error })
    expect(await signOutSession()).toEqual({ error })
    expect(auth.getUser).not.toHaveBeenCalled()
  })
  it('does not claim signed out if subsequent validation finds a user', async () => {
    auth.signOut.mockResolvedValue({ error: missing })
    auth.getUser.mockResolvedValue({ data: { user: { id: 'fixture' } }, error: null })
    expect(await signOutSession()).toEqual({ error: missing })
  })
  it('preserves a failed validation result during sign-out', async () => {
    const error = { name: 'AuthRetryableFetchError', status: 503 }
    auth.signOut.mockResolvedValue({ error: missing })
    auth.getUser.mockResolvedValue({ data: { user: null }, error })
    expect(await signOutSession()).toEqual({ error })
  })
  it('validates a real user instead of cached identity', async () => {
    auth.getUser.mockResolvedValue({ data: { user: { id: 'verified' } }, error: null })
    expect(await validateSessionUser()).toEqual({ user: { id: 'verified' }, error: null })
  })
  it('represents a confirmed missing session as signed out', async () => {
    auth.getUser.mockResolvedValue({ data: { user: null }, error: missing })
    expect(await validateSessionUser()).toEqual({ user: null, error: null })
  })
  it('reports transient validation failure without accepting cached user', async () => {
    auth.getUser.mockResolvedValue({ data: { user: null }, error: { status: 503 } })
    const result = await validateSessionUser()
    expect(result.user).toBeNull()
    expect(result.error).toContain('Unable to verify')
  })
})
