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

function isProduction() {
  const env = (getEnv('ENV') || getEnv('APP_ENV') || getEnv('NODE_ENV') || getEnv('CONTEXT') || '').toLowerCase()
  return env === 'production' || env === 'prod'
}

function isLocalhostOrigin(origin) {
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)
}

export function corsHeadersFor(request, { allowMethods = 'POST, GET, OPTIONS' } = {}) {
  const origin = request.headers.get('origin')?.replace(/\/$/, '')

  if (!origin) return {}

  const allowed = splitOrigins(getEnv('APP_ORIGINS'))
  const allowLocalhost = !isProduction()

  const isAllowed = allowed.includes(origin) || (allowLocalhost && isLocalhostOrigin(origin))
  if (!isAllowed) return {}

  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': allowMethods,
    'Access-Control-Allow-Headers': 'authorization, content-type',
    Vary: 'Origin',
  }
}

