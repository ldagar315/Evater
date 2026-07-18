import { spawnSync } from 'node:child_process'
import process from 'node:process'
import { getLocalSupabaseEnv, pythonCommand, repoRoot, runSupabase } from './stage-env.mjs'

const supabaseEnv = getLocalSupabaseEnv()
const python = pythonCommand()
const executable = spawnSync(python, ['--version'], { stdio: 'ignore' }).error ? 'python3' : python
const sharedEnv = {
  ...process.env,
  ...supabaseEnv,
  ENV: 'development',
  APP_ENV: 'development',
}

function runPython(args) {
  const result = spawnSync(executable, args, {
    cwd: `${repoRoot}/backend`,
    env: sharedEnv,
    stdio: 'inherit',
  })
  if (result.error) throw result.error
  if (result.status !== 0) process.exit(result.status || 1)
}

const sourceRunDir = `${repoRoot}/tmp/class8-science-source-run`

console.log('Downloading and extracting all source-backed Class 8 Science question types for Chapters 1–13...')
runPython([
  '-m',
  'etl.scrape_class8_science',
  '--chapters',
  '1-13',
  '--output-dir',
  sourceRunDir,
  '--adapt-minimum-candidates',
  '60',
])

console.log('Validating MCQs and archiving all extracted question types in local Supabase for stage testing...')
runPython([
  '-m',
  'etl.seed_class8_science',
  '--chapters',
  '1-13',
  '--source-packs-dir',
  `${sourceRunDir}/packs`,
  '--source-manifest',
  `${sourceRunDir}/source_manifest.json`,
  '--publish',
  '--replace-existing',
  '--acknowledge-source-review',
])

console.log('Publishing deterministic question-type fixtures for stage renderer testing...')
runPython(['-m', 'etl.seed_deterministic_question_types'])

console.log('Ensuring the stage admin is enrolled in Class 8A...')
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

console.log('Class 8 Science stage seed is ready with source-backed questions; counts are intentionally source-sized, not capped at 100.')
