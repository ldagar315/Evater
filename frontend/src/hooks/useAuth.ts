import { useEffect, useState } from 'react'
import { User } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase'
import { BYPASS_AUTH, DEV_USER, logAuthBypassWarning } from '../lib/auth/devBypass'

export function useAuth() {
  const [user, setUser] = useState<User | null>(BYPASS_AUTH ? DEV_USER : null)
  const [loading, setLoading] = useState(!BYPASS_AUTH)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true

    if (BYPASS_AUTH) {
      logAuthBypassWarning('useAuth bootstrap')
      setUser(DEV_USER)
      setError(null)
      setLoading(false)
      return () => {
        mounted = false
      }
    }

    const initializeAuth = async () => {
      try {
        // Check if Supabase is properly configured
        const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
        const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

        if (!supabaseUrl || !supabaseAnonKey) {
          console.warn('Supabase environment variables not configured')
          if (mounted) {
            setError('Authentication service not configured. Please check environment variables.')
            setLoading(false)
          }
          return
        }

        // Get initial session
        const { data: { session }, error: sessionError } = await supabase.auth.getSession()
        
        if (sessionError) {
          console.error('Session error:', sessionError)
          if (mounted) {
            setError('Failed to initialize authentication')
          }
        } else if (mounted) {
          setUser(session?.user ?? null)
          setError(null)
        }
      } catch (err) {
        console.error('Auth initialization error:', err)
        if (mounted) {
          setError('Authentication service unavailable')
        }
      } finally {
        if (mounted) {
          setLoading(false)
        }
      }
    }

    initializeAuth()

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, session) => {
        console.log('Auth state change:', event, session?.user?.email)
        if (mounted) {
          setUser(session?.user ?? null)
          setError(null)
          setLoading(false)
        }
      }
    )

    return () => {
      mounted = false
      subscription.unsubscribe()
    }
  }, [])

  const signUp = async (email: string, password: string) => {
    if (BYPASS_AUTH) {
      logAuthBypassWarning('signUp noop')
      setUser(DEV_USER)
      return { data: { user: DEV_USER } as any, error: null }
    }
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
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
      return { data: { user: DEV_USER } as any, error: null }
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
      return { data: { user: DEV_USER } as any, error: null }
    }
    try {
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/home`
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
      const { error } = await supabase.auth.signOut()
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
