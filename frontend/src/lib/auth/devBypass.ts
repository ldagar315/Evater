import { User } from '@supabase/supabase-js'
import { UserProfile } from '../../types'

export const BYPASS_AUTH = import.meta.env.VITE_BYPASS_AUTH === 'true'

const now = new Date().toISOString()

const baseDevUser: Partial<User> = {
  id: 'dev-bypass-user',
  aud: 'authenticated',
  role: 'authenticated',
  email: 'dev@example.com',
  email_confirmed_at: now,
  phone: '',
  phone_confirmed_at: null,
  confirmation_sent_at: now,
  confirmed_at: now,
  recovery_sent_at: null,
  factors: [],
  app_metadata: {
    provider: 'dev-bypass',
  },
  user_metadata: {
    name: 'Dev User',
    bypass: true,
  },
  identities: [],
  created_at: now,
  updated_at: now,
  last_sign_in_at: now,
}

export const DEV_USER = baseDevUser as User

export const buildDevProfile = (overrides: Partial<UserProfile> = {}): UserProfile => ({
  id: overrides.id ?? 'dev-profile',
  created_by: overrides.created_by ?? DEV_USER.id,
  created_at: overrides.created_at ?? now,
  user_name: overrides.user_name ?? 'Dev User',
  email: overrides.email ?? DEV_USER.email,
  grade: overrides.grade ?? 10,
  credits: overrides.credits ?? 999,
  school: overrides.school ?? 'Local Dev School',
  name: overrides.name ?? 'Dev User',
  profile_picture: overrides.profile_picture ?? null,
})

let hasLoggedWarning = false

export const logAuthBypassWarning = (context: string) => {
  if (!BYPASS_AUTH || hasLoggedWarning) return
  hasLoggedWarning = true
  console.warn(`[DEV ONLY] Auth bypass enabled during ${context}. Remove VITE_BYPASS_AUTH before production.`)
}
