import { signOutSession, validateSessionUser } from '../lib/sessionRecovery'
import { authReturn } from '../lib/practiceTarget'
import { useEffect, useState } from 'react'
import { AuthChangeEvent, AuthError, Session, User } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase'
import { supabaseConfigured } from '../lib/supabaseConfig'
import { BYPASS_AUTH, DEV_USER, logAuthBypassWarning } from '../lib/auth/devBypass'

export function useAuth() {
  const [user, setUser] = useState<User | null>(BYPASS_AUTH ? DEV_USER : null)
  const [loading, setLoading] = useState(!BYPASS_AUTH)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true
    let revision = 0

    if (BYPASS_AUTH) {
      logAuthBypassWarning('useAuth bootstrap')
      setUser(DEV_USER)
      setError(null)
      setLoading(false)
      return () => {
        mounted = false
      }
    }

    const applySession = async (hasSession: boolean) => {
      const current = ++revision
      if (!hasSession) {
        if (mounted) { setUser(null); setError(null); setLoading(false) }
        return
      }
      if (mounted) setLoading(true)
      try {
        const result = await validateSessionUser()
        if (mounted && current === revision) {
          setUser(result.user)
          setError(result.error)
        }
      } catch {
        if (mounted && current === revision) {
          setUser(null)
          setError('Authentication service unavailable')
        }
      } finally {
        if (mounted && current === revision) setLoading(false)
      }
    }

    // SDK auth callbacks run under its session lock. Schedule validation outside
    // that callback so getUser cannot deadlock. Revision checks discard late
    // validation after sign-out, account changes, or unmount.
    const timers = new Set<ReturnType<typeof setTimeout>>()
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event: AuthChangeEvent, session: Session | null) => {
        ++revision
        if (!session) {
          if (mounted) { setUser(null); setError(null); setLoading(false) }
          return
        }
        if (mounted) setLoading(true)
        const expected = revision
        const timer = setTimeout(() => {
          timers.delete(timer)
          if (mounted && revision === expected) void applySession(true)
        }, 0)
        timers.add(timer)
      }
    )

    // Missing local config must not keep public routes in a loading state.
    if (!supabaseConfigured) {
      setLoading(false)
    } else {
      const initialRevision = revision
      void supabase.auth.getSession().then(({ data, error: sessionError }: { data: { session: Session | null }; error: AuthError | null }) => {
        if (!mounted || revision !== initialRevision) return
        if (sessionError) {
          setError('Failed to initialize authentication')
          setLoading(false)
        } else {
          void applySession(Boolean(data.session))
        }
      }).catch(() => {
        if (mounted && revision === initialRevision) {
          setError('Authentication service unavailable')
          setLoading(false)
        }
      })
    }

    return () => {
      mounted = false
      ++revision
      timers.forEach(clearTimeout)
      subscription.unsubscribe()
    }
  }, [])

  const signUp = async (email: string, password: string) => {
    if (BYPASS_AUTH) {
      logAuthBypassWarning('signUp noop')
      setUser(DEV_USER)
      return { data: { user: DEV_USER }, error: null }
    }
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        ...(authReturn(window.location.search) !== '/home' ? { options: { emailRedirectTo: window.location.origin + authReturn(window.location.search) } } : {}),
      })
      return { data, error }
    } catch (err) {
      console.error('Sign up error:', err)
      return { data: null, error: { message: 'Sign up failed' } }
    }
  }

  const signIn = async (email: string, password: string) => {
    if (BYPASS_AUTH) {
      logAuthBypassWarning('signIn noop')
      setUser(DEV_USER)
      return { data: { user: DEV_USER }, error: null }
    }
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })
      return { data, error }
    } catch (err) {
      console.error('Sign in error:', err)
      return { data: null, error: { message: 'Sign in failed' } }
    }
  }

  const signInWithGoogle = async () => {
    if (BYPASS_AUTH) {
      logAuthBypassWarning('signInWithGoogle noop')
      setUser(DEV_USER)
      return { data: { user: DEV_USER }, error: null }
    }
    try {
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: window.location.origin + authReturn(window.location.search)
        }
      })
      return { data, error }
    } catch (err) {
      console.error('Google sign in error:', err)
      return { data: null, error: { message: 'Google sign in failed' } }
    }
  }

  const signOut = async () => {
    if (BYPASS_AUTH) {
      logAuthBypassWarning('signOut noop')
      setUser(DEV_USER)
      return { error: null }
    }
    try {
      const { error } = await signOutSession()
      return { error }
    } catch (err) {
      console.error('Sign out error:', err)
      return { error: { message: 'Sign out failed' } }
    }
  }

  return {
    user,
    loading,
    error,
    signUp,
    signIn,
    signInWithGoogle,
    signOut,
  }
}
