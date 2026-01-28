function getEnv(name) {
  if (typeof process !== 'undefined' && process?.env) return process.env[name]
  if (typeof Deno !== 'undefined' && Deno?.env) return Deno.env.get(name)
  return undefined
}

function splitOrigins(value) {
  if (!value) return []
  return value
    .split(',')
    .map((o) => o.trim().replace(/\/$/, ''))
    .filter(Boolean)
}

function isExplicitNonProd() {
  const env = (getEnv('ENV') || getEnv('APP_ENV') || getEnv('NODE_ENV') || '').toLowerCase()
  return env === 'development' || env === 'dev' || env === 'local' || env === 'test'
}

function isProduction() {
  // Default to production-safe behavior if unset/unknown.
  return !isExplicitNonProd()
}

function isLocalhostOrigin(origin) {
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)
}

export function corsDecision(request, { allowMethods = 'POST, GET, OPTIONS' } = {}) {
  const origin = request.headers.get('origin')?.replace(/\/$/, '')

  if (!origin) return { origin: null, allowed: true, headers: {} }

  const allowed = splitOrigins(getEnv('APP_ORIGINS'))
  const allowLocalhost = isExplicitNonProd()

  const isAllowed = allowed.includes(origin) || (allowLocalhost && isLocalhostOrigin(origin))
  if (!isAllowed) return { origin, allowed: false, headers: {} }

  return {
    origin,
    allowed: true,
    headers: {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': allowMethods,
      'Access-Control-Allow-Headers': 'authorization, content-type',
      Vary: 'Origin',
    },
  }
}

export function corsHeadersFor(request, opts) {
  return corsDecision(request, opts).headers
}
