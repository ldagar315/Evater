import { spawnSync } from 'node:child_process'
import process from 'node:process'
import { getLocalSupabaseEnv, pythonCommand, repoRoot, runSupabase } from './stage-env.mjs'

const supabaseEnv = getLocalSupabaseEnv()
const python = pythonCommand()
const executable = spawnSync(python, ['--version'], { stdio: 'ignore' }).error ? 'python3' : python

const result = spawnSync(
  executable,
  ['-m', 'etl.seed_first_chapter', '--publish', '--acknowledge-source-review'],
  {
    cwd: `${repoRoot}/backend`,
    env: {
      ...process.env,
      ...supabaseEnv,
      ENV: 'development',
      APP_ENV: 'development',
    },
    stdio: 'inherit',
  },
)

if (result.error) throw result.error
if (result.status !== 0) process.exit(result.status || 1)

const fixtureResult = spawnSync(
  executable,
  ['-m', 'etl.seed_deterministic_question_types'],
  {
    cwd: `${repoRoot}/backend`,
    env: {
      ...process.env,
      ...supabaseEnv,
      ENV: 'development',
      APP_ENV: 'development',
    },
    stdio: 'inherit',
  },
)

if (fixtureResult.error) throw fixtureResult.error
if (fixtureResult.status !== 0) process.exit(fixtureResult.status || 1)

runSupabase([
  'db',
  'query',
  '--local',
  `with stage_school as (
    select id from public.schools where slug = 'evater-local-stage'
  ), stage_class as (
    select classrooms.id, classrooms.school_id
    from public.classrooms
    join stage_school on stage_school.id = classrooms.school_id
    where classrooms.name = 'Class 8A'
  ), existing_profiles as (
    select created_by as user_id
    from public."Users"
    where school = 'Evater Local Stage' and grade = 8
  )
  insert into public.student_enrollments (user_id, school_id, classroom_id)
  select existing_profiles.user_id, stage_class.school_id, stage_class.id
  from existing_profiles cross join stage_class
  on conflict (user_id) do update
  set school_id = excluded.school_id, classroom_id = excluded.classroom_id`,
])

process.exit(0)
