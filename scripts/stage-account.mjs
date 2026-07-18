import process from 'node:process'
import { getLocalSupabaseEnv, runSupabase } from './stage-env.mjs'

const email = process.env.EVATER_STAGE_ACCOUNT_EMAIL || 'stage-admin@evater.test'
const password = process.env.EVATER_STAGE_ACCOUNT_PASSWORD || 'StageAdmin123!'

async function adminRequest(baseUrl, serviceRoleKey, path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`,
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(`Local Supabase Auth returned ${response.status}: ${payload.msg || payload.message || payload.error_description || 'request failed'}`)
  }
  return payload
}

const supabaseEnv = getLocalSupabaseEnv()
const usersPayload = await adminRequest(
  supabaseEnv.SUPABASE_URL,
  supabaseEnv.SUPABASE_SERVICE_ROLE_KEY,
  '/auth/v1/admin/users?page=1&per_page=1000',
)
const existingUser = (usersPayload.users || []).find((user) => user.email?.toLowerCase() === email.toLowerCase())
const user = existingUser
  ? await adminRequest(supabaseEnv.SUPABASE_URL, supabaseEnv.SUPABASE_SERVICE_ROLE_KEY, `/auth/v1/admin/users/${existingUser.id}`, {
    method: 'PUT',
    body: JSON.stringify({
      email,
      password,
      email_confirm: true,
      user_metadata: { name: 'Stage Admin', grade: 8, school: 'Evater Local Stage' },
      app_metadata: { provider: 'email', role: 'admin' },
    }),
  })
  : await adminRequest(supabaseEnv.SUPABASE_URL, supabaseEnv.SUPABASE_SERVICE_ROLE_KEY, '/auth/v1/admin/users', {
    method: 'POST',
    body: JSON.stringify({
      email,
      password,
      email_confirm: true,
      user_metadata: { name: 'Stage Admin', grade: 8, school: 'Evater Local Stage' },
      app_metadata: { provider: 'email', role: 'admin' },
    }),
  })

const userId = user.id
const sqlEmail = email.replaceAll("'", "''")
runSupabase([
  'db',
  'query',
  '--local',
  `insert into public."Users" (created_by, user_name, email, grade, school, credits, name)
   values ('${userId}', 'Stage Admin', '${sqlEmail}', 8, 'Evater Local Stage', 999, 'Stage Admin')
   on conflict (created_by) do update set
     user_name = excluded.user_name,
     email = excluded.email,
     grade = excluded.grade,
     school = excluded.school,
     credits = excluded.credits,
     name = excluded.name`,
])

runSupabase([
  'db',
  'query',
  '--local',
  `insert into public.student_enrollments (user_id, school_id, classroom_id, role)
   select '${userId}', school.id, classroom.id, 'admin'
   from public.schools school
   join public.classrooms classroom on classroom.school_id = school.id
   where school.slug = 'evater-local-stage' and classroom.name = 'Class 8A'
   on conflict (user_id) do update set
     school_id = excluded.school_id,
     classroom_id = excluded.classroom_id,
     role = excluded.role`,
])

console.log(`Local stage account ready: ${email}`)
console.log(`Password: ${password}`)
console.log(`User id: ${userId}`)
