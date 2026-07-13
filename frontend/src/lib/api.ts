import { supabase } from './supabase'

const DEFAULT_MODAL_API_URL = 'https://ldagar315--evater-v1-wrapper.modal.run'

export const MODAL_API_URL = (
  import.meta.env.VITE_MODAL_API_URL || DEFAULT_MODAL_API_URL
).replace(/\/+$/, '')

export const MODAL_WS_URL = (
  import.meta.env.VITE_MODAL_WS_URL || `${MODAL_API_URL.replace(/^http/, 'ws')}/ws/viva`
).replace(/\/+$/, '')

export async function getAccessToken(): Promise<string | null> {
  const {
    data: { session },
  } = await supabase.auth.getSession()

  return session?.access_token ?? null
}

export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers)
  const accessToken = await getAccessToken()

  if (accessToken) {
    headers.set('Authorization', `Bearer ${accessToken}`)
  }

  return fetch(input, {
    ...init,
    headers,
  })
}
