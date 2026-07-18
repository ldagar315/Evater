import { existsSync } from 'node:fs'
import { spawn, spawnSync } from 'node:child_process'
import process from 'node:process'
import {
  getLocalSupabaseEnv,
  pythonCommand,
  repoRoot,
  runSupabase,
  stageEnvironment,
} from './stage-env.mjs'

const resetDatabase = process.argv.includes('--reset')
const frontendPort = process.env.EVATER_STAGE_FRONTEND_PORT || '5173'
const backendPort = process.env.EVATER_STAGE_BACKEND_PORT || '8000'
const python = pythonCommand()
const pythonExecutable = existsSync(python) ? python : 'python3'

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || repoRoot,
    env: options.env || stageEnvironment(),
    stdio: 'inherit',
  })
  if (result.error) throw result.error
  if (result.status !== 0) process.exit(result.status || 1)
}

console.log('Starting local Supabase...')
run(process.platform === 'win32' ? 'npx.cmd' : 'npx', ['--no-install', 'supabase', 'start'])

if (resetDatabase) {
  console.log('Resetting the local database and applying fixtures...')
  runSupabase(['db', 'reset', '--local', '--yes'])
}

const supabaseEnv = getLocalSupabaseEnv()
const origins = `http://127.0.0.1:${frontendPort},http://localhost:${frontendPort}`
const sharedEnv = {
  ...process.env,
  ...supabaseEnv,
  ENV: 'development',
  APP_ENV: 'development',
  NODE_ENV: 'development',
}

const backend = spawn(
  pythonExecutable,
  ['-m', 'uvicorn', 'app.main:app', '--reload', '--host', '127.0.0.1', '--port', backendPort],
  {
    cwd: `${repoRoot}/backend`,
    env: {
      ...sharedEnv,
      APP_ORIGINS: origins,
      PORT: backendPort,
      FASTAPI_RELOAD: 'true',
    },
    stdio: 'inherit',
  },
)

const frontend = spawn(
  process.platform === 'win32' ? 'npm.cmd' : 'npm',
  ['run', 'dev', '--', '--host', '127.0.0.1', '--port', frontendPort],
  {
    cwd: `${repoRoot}/frontend`,
    env: {
      ...sharedEnv,
      VITE_SUPABASE_URL: supabaseEnv.SUPABASE_URL,
      VITE_SUPABASE_ANON_KEY: supabaseEnv.SUPABASE_ANON_KEY,
      VITE_MODAL_API_URL: `http://127.0.0.1:${backendPort}`,
      VITE_MODAL_WS_URL: `ws://127.0.0.1:${backendPort}/ws/viva`,
      VITE_BYPASS_AUTH: process.env.EVATER_STAGE_BYPASS_AUTH === 'true' ? 'true' : 'false',
    },
    stdio: 'inherit',
  },
)

let shuttingDown = false

function shutdown(exitCode = 0) {
  if (shuttingDown) return
  shuttingDown = true
  for (const child of [backend, frontend]) {
    if (!child.killed) child.kill('SIGTERM')
  }
  console.log('\nLocal services stopped. Supabase is still running for fast restarts.')
  console.log('Run `npm run stage:stop` when you want to shut down the local Supabase stack.')
  process.exit(exitCode)
}

for (const child of [backend, frontend]) {
  child.on('exit', (code, signal) => {
    if (shuttingDown) return
    const status = signal ? `signal ${signal}` : `exit code ${code}`
    console.error(`A local service stopped (${status}). Shutting down the stage runner.`)
    shutdown(code || 1)
  })
}

process.on('SIGINT', () => shutdown(0))
process.on('SIGTERM', () => shutdown(0))

console.log(`\nEvater local stage is ready:`)
console.log(`  Frontend:  http://127.0.0.1:${frontendPort}`)
console.log(`  Backend:   http://127.0.0.1:${backendPort}`)
console.log('  Supabase:  http://127.0.0.1:54321')
console.log('  Studio:    http://127.0.0.1:54323')
console.log('  Mailpit:   http://127.0.0.1:54324')
console.log('\nUse the UI to create a local account; stage runs with real Supabase auth.')
