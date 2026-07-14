const RESTORED_SUPABASE_URL = 'https://qfxrfwgipsygoozlnkwb.supabase.co'
const RESTORED_SUPABASE_PUBLISHABLE_KEY = 'sb_publishable_vm-kmCaxSYORUQZEoUP0fw_h2iM5PnA'

// Production must use the restored project even if a stale deployment environment
// is still present in the hosting provider. Keep local development environment-driven.
const configuredSupabaseUrl = import.meta.env.DEV
  ? (import.meta.env.VITE_SUPABASE_URL || '').trim()
  : ''
const configuredSupabaseAnonKey = import.meta.env.DEV
  ? (import.meta.env.VITE_SUPABASE_ANON_KEY || '').trim()
  : ''

export const supabaseUrl = import.meta.env.PROD
  ? RESTORED_SUPABASE_URL
  : configuredSupabaseUrl

export const supabaseAnonKey = import.meta.env.PROD
  ? RESTORED_SUPABASE_PUBLISHABLE_KEY
  : configuredSupabaseAnonKey

export const supabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey)
