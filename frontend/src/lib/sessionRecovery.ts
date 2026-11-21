import { AuthError, User } from '@supabase/supabase-js'
import { supabase } from './supabase'

// auth-js turns session_not_found into AuthSessionMissingError (status 400).
// Its signOut path does not clear this case, while getUser does. Let the SDK
// own storage removal and its SIGNED_OUT event; never wipe browser storage.
export async function signOutSession(): Promise<{ error: AuthError | null }> {
  const result = await supabase.auth.signOut()
  if (result.error?.name !== 'AuthSessionMissingError') return result

  const validation = await supabase.auth.getUser()
  if (validation.error?.name === 'AuthSessionMissingError') return { error: null }
  return { error: validation.error ?? result.error }
}

export async function validateSessionUser(): Promise<{ user: User | null; error: string | null }> {
  const { data, error } = await supabase.auth.getUser()
  if (!error) return { user: data.user, error: null }
  if (error.name === 'AuthSessionMissingError') return { user: null, error: null }
  // A network/server failure is not proof of revocation. Do not clear anything
  // or present a cached identity as server-validated.
  return { user: null, error: 'Unable to verify your session. Please reload and try again.' }
}
