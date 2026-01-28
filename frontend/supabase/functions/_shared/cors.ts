function splitOrigins(value?: string | null): string[] {
  if (!value) return []
  return value
    .split(',')
    .map((o) => o.trim().replace(/\/$/, ''))
    .filter(Boolean)
}

function isProduction(): boolean {
  const env = (Deno.env.get('ENV') || Deno.env.get('APP_ENV') || Deno.env.get('NODE_ENV') || '').toLowerCase()
  // Default to production-safe behavior if unset/unknown.
  return env !== 'development' && env !== 'dev' && env !== 'local' && env !== 'test'
}

function isLocalhostOrigin(origin: string): boolean {
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)
}

export function corsHeaders(req: Request): Record<string, string> {
  const origin = req.headers.get('origin')?.replace(/\/$/, '')
  if (!origin) return {}

  const allowed = splitOrigins(Deno.env.get('APP_ORIGINS'))
  const allowLocalhost = !isProduction() && (Deno.env.get('ENV') || Deno.env.get('APP_ENV') || Deno.env.get('NODE_ENV'))

  const isAllowed = allowed.includes(origin) || (allowLocalhost && isLocalhostOrigin(origin))
  if (!isAllowed) return {}

  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, PUT, DELETE',
    Vary: 'Origin',
  }
}
