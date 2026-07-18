import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import process from 'node:process'

export const repoRoot = new URL('..', import.meta.url).pathname.replace(/\/$/, '')

const npxCommand = process.platform === 'win32' ? 'npx.cmd' : 'npx'

export function stageEnvironment() {
  const env = { ...process.env }
  const dockerDesktopSocket = `${env.HOME || ''}/.docker/run/docker.sock`

  // Docker Desktop on macOS exposes its engine through this per-user socket.
  // Prefer an explicitly supplied DOCKER_HOST, but make the normal local
  // stage command work even when /var/run/docker.sock is not symlinked.
  if (!env.DOCKER_HOST && existsSync(dockerDesktopSocket)) {
    env.DOCKER_HOST = `unix://${dockerDesktopSocket}`
  }

  return env
}

export function runSupabase(args, options = {}) {
  return execFileSync(npxCommand, ['--no-install', 'supabase', ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    stdio: options.stdio ?? 'inherit',
    env: stageEnvironment(),
  })
}

function parseEnvOutput(output) {
  return output.split(/\r?\n/).reduce((values, line) => {
    const match = line.match(/^([A-Z0-9_]+)=(.*)$/)
    if (!match) return values

    let value = match[2].trim()
    if (value.length >= 2 && value.startsWith('"') && value.endsWith('"')) {
      value = value.slice(1, -1).replace(/\\"/g, '"')
    }
    values[match[1]] = value
    return values
  }, {})
}

export function getLocalSupabaseEnv() {
  let output
  try {
    output = runSupabase(['status', '-o', 'env'], { stdio: 'pipe' })
  } catch {
    throw new Error(
      'Local Supabase is not running. Start Docker Desktop, then run `npm run stage` again.',
    )
  }

  const values = parseEnvOutput(output)
  const localEnv = {
    SUPABASE_URL: values.API_URL || 'http://127.0.0.1:54321',
    SUPABASE_API_KEY: values.ANON_KEY,
    SUPABASE_ANON_KEY: values.ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY: values.SERVICE_ROLE_KEY,
  }

  const missing = Object.entries(localEnv)
    .filter(([, value]) => !value)
    .map(([key]) => key)

  if (missing.length > 0) {
    throw new Error(`Supabase status did not return: ${missing.join(', ')}`)
  }

  return localEnv
}

export function pythonCommand() {
  return process.env.PYTHON_BIN || `${repoRoot}/backend/.venv/bin/python`
}
