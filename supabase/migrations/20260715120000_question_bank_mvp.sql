-- Evater question-bank MVP foundation.
-- New tables use lowercase identifiers. Existing quoted legacy tables remain untouched.

create table if not exists public.curriculum_versions (
  id uuid primary key default gen_random_uuid(),
  board text not null,
  grade smallint not null check (grade between 1 and 12),
  subject text not null,
  language text not null default 'en',
  academic_year text not null,
  version_label text not null,
  textbook_name text,
  status text not null default 'draft'
    check (status in ('draft', 'published', 'retired')),
  source_url text,
  license_status text not null default 'unknown',
  created_at timestamptz not null default now(),
  unique (board, grade, subject, language, academic_year, version_label)
);

create table if not exists public.chapters (
  id uuid primary key default gen_random_uuid(),
  curriculum_version_id uuid not null references public.curriculum_versions(id) on delete cascade,
  sequence_number smallint not null check (sequence_number > 0),
  title text not null,
  slug text not null,
  description text,
  status text not null default 'draft'
    check (status in ('draft', 'published', 'retired')),
  created_at timestamptz not null default now(),
  unique (curriculum_version_id, slug),
  unique (curriculum_version_id, sequence_number)
);

create table if not exists public.concepts (
  id uuid primary key default gen_random_uuid(),
  chapter_id uuid not null references public.chapters(id) on delete cascade,
  parent_concept_id uuid references public.concepts(id) on delete set null,
  sequence_number smallint not null check (sequence_number > 0),
  title text not null,
  slug text not null,
  learning_outcome text,
  prerequisite_concept_ids uuid[] not null default '{}',
  status text not null default 'draft'
    check (status in ('draft', 'published', 'retired')),
  created_at timestamptz not null default now(),
  unique (chapter_id, slug),
  unique (chapter_id, sequence_number)
);

create table if not exists public.content_sources (
  id uuid primary key default gen_random_uuid(),
  source_url text not null,
  source_type text not null
    check (source_type in ('official_html', 'official_pdf', 'open_license', 'manual', 'other')),
  publisher text,
  license text not null default 'unknown',
  attribution text,
  content_hash text not null,
  fetched_at timestamptz,
  status text not null default 'discovered'
    check (status in ('discovered', 'fetched', 'parsed', 'failed', 'retired')),
  created_at timestamptz not null default now(),
  unique (source_url, content_hash)
);

create table if not exists public.ingestion_jobs (
  id uuid primary key default gen_random_uuid(),
  source_id uuid references public.content_sources(id) on delete set null,
  curriculum_version_id uuid references public.curriculum_versions(id) on delete set null,
  job_type text not null
    check (job_type in ('discover', 'fetch', 'extract', 'normalize', 'generate', 'validate', 'publish', 'refresh')),
  status text not null default 'queued'
    check (status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
  input_count integer not null default 0 check (input_count >= 0),
  output_count integer not null default 0 check (output_count >= 0),
  error_count integer not null default 0 check (error_count >= 0),
  error_summary text,
  metadata jsonb not null default '{}'::jsonb,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists public.question_bank (
  id uuid primary key default gen_random_uuid(),
  curriculum_version_id uuid not null references public.curriculum_versions(id) on delete restrict,
  chapter_id uuid not null references public.chapters(id) on delete restrict,
  concept_id uuid not null references public.concepts(id) on delete restrict,
  ingestion_job_id uuid references public.ingestion_jobs(id) on delete set null,
  source_id uuid references public.content_sources(id) on delete set null,
  question_type text not null default 'mcq_single'
    check (question_type = 'mcq_single'),
  question_text text not null check (char_length(trim(question_text)) between 10 and 1000),
  options_json jsonb not null check (jsonb_typeof(options_json) = 'array'),
  correct_option_id text not null check (correct_option_id in ('A', 'B', 'C', 'D')),
  explanation text not null check (char_length(trim(explanation)) between 5 and 2000),
  hint text,
  difficulty text not null check (difficulty in ('easy', 'medium', 'hard')),
  cognitive_level text not null check (cognitive_level in ('recall', 'understand', 'apply', 'analyze')),
  skill_tags text[] not null default '{}',
  misconception_tags text[] not null default '{}',
  question_style text not null default 'direct'
    check (question_style in ('direct', 'scenario', 'experiment', 'data', 'diagram')),
  estimated_time_seconds smallint not null default 60
    check (estimated_time_seconds between 10 and 600),
  marks smallint not null default 1 check (marks > 0),
  quality_score numeric(4,3) check (quality_score between 0 and 1),
  content_hash text not null,
  status text not null default 'draft'
    check (status in ('draft', 'validated', 'review', 'published', 'rejected', 'retired')),
  reviewed_by uuid references auth.users(id) on delete set null,
  reviewed_at timestamptz,
  published_at timestamptz,
  created_at timestamptz not null default now(),
  unique (chapter_id, content_hash)
);

create table if not exists public.test_attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  chapter_id uuid not null references public.chapters(id) on delete restrict,
  mode text not null default 'practice'
    check (mode in ('diagnostic', 'practice', 'remedial', 'challenge')),
  question_count smallint not null check (question_count between 1 and 100),
  block_size smallint not null default 5 check (block_size between 1 and 20),
  current_block smallint not null default 1 check (current_block > 0),
  seed bigint not null,
  routing_profile jsonb not null default '{}'::jsonb,
  status text not null default 'in_progress'
    check (status in ('in_progress', 'completed', 'abandoned')),
  score numeric(6,2),
  maximum_marks smallint,
  started_at timestamptz not null default now(),
  completed_at timestamptz
);

create table if not exists public.test_questions (
  id uuid primary key default gen_random_uuid(),
  test_attempt_id uuid not null references public.test_attempts(id) on delete cascade,
  question_id uuid not null references public.question_bank(id) on delete restrict,
  block_number smallint not null check (block_number > 0),
  display_order smallint not null check (display_order > 0),
  selection_reason text,
  created_at timestamptz not null default now(),
  unique (test_attempt_id, display_order),
  unique (test_attempt_id, question_id)
);

create table if not exists public.question_attempts (
  id uuid primary key default gen_random_uuid(),
  test_attempt_id uuid not null references public.test_attempts(id) on delete cascade,
  test_question_id uuid not null references public.test_questions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  chapter_id uuid not null references public.chapters(id) on delete restrict,
  question_id uuid not null references public.question_bank(id) on delete restrict,
  selected_option_id text check (selected_option_id in ('A', 'B', 'C', 'D')),
  is_correct boolean not null,
  marks_awarded smallint not null default 0 check (marks_awarded >= 0),
  response_time_ms integer check (response_time_ms is null or response_time_ms >= 0),
  created_at timestamptz not null default now(),
  unique (test_question_id)
);

create table if not exists public.concept_mastery (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  chapter_id uuid not null references public.chapters(id) on delete cascade,
  concept_id uuid not null references public.concepts(id) on delete cascade,
  mastery_score numeric(5,2) not null default 0 check (mastery_score between 0 and 100),
  attempt_count integer not null default 0 check (attempt_count >= 0),
  correct_count integer not null default 0 check (correct_count >= 0),
  last_difficulty text check (last_difficulty is null or last_difficulty in ('easy', 'medium', 'hard')),
  misconception_signals jsonb not null default '{}'::jsonb,
  last_attempted_at timestamptz,
  next_review_at timestamptz,
  updated_at timestamptz not null default now(),
  unique (user_id, chapter_id, concept_id)
);

create index if not exists chapters_curriculum_status_idx
  on public.chapters (curriculum_version_id, status, sequence_number);
create index if not exists concepts_chapter_status_idx
  on public.concepts (chapter_id, status, sequence_number);
create index if not exists content_sources_status_idx
  on public.content_sources (status, fetched_at);
create index if not exists ingestion_jobs_status_created_idx
  on public.ingestion_jobs (status, created_at);
create index if not exists question_bank_published_selection_idx
  on public.question_bank (chapter_id, difficulty, concept_id)
  where status = 'published';
create index if not exists question_bank_concept_status_idx
  on public.question_bank (concept_id, status);
create index if not exists test_attempts_user_status_idx
  on public.test_attempts (user_id, status, started_at desc);
create index if not exists test_questions_attempt_block_idx
  on public.test_questions (test_attempt_id, block_number, display_order);
create index if not exists question_attempts_user_chapter_created_idx
  on public.question_attempts (user_id, chapter_id, created_at desc);
create index if not exists concept_mastery_user_chapter_idx
  on public.concept_mastery (user_id, chapter_id, mastery_score);

alter table public.curriculum_versions enable row level security;
alter table public.chapters enable row level security;
alter table public.concepts enable row level security;
alter table public.content_sources enable row level security;
alter table public.ingestion_jobs enable row level security;
alter table public.question_bank enable row level security;
alter table public.test_attempts enable row level security;
alter table public.test_questions enable row level security;
alter table public.question_attempts enable row level security;
alter table public.concept_mastery enable row level security;

drop policy if exists "Published curriculum is readable" on public.curriculum_versions;
create policy "Published curriculum is readable"
on public.curriculum_versions for select to authenticated
using (status = 'published');

drop policy if exists "Published chapters are readable" on public.chapters;
create policy "Published chapters are readable"
on public.chapters for select to authenticated
using (
  status = 'published'
  and exists (
    select 1 from public.curriculum_versions cv
    where cv.id = curriculum_version_id
      and cv.status = 'published'
  )
);

drop policy if exists "Published concepts are readable" on public.concepts;
create policy "Published concepts are readable"
on public.concepts for select to authenticated
using (
  status = 'published'
  and exists (
    select 1
    from public.chapters c
    join public.curriculum_versions cv on cv.id = c.curriculum_version_id
    where c.id = chapter_id
      and c.status = 'published'
      and cv.status = 'published'
  )
);

drop policy if exists "Published questions are readable" on public.question_bank;
create policy "Published questions are readable"
on public.question_bank for select to authenticated
using (status = 'published');

drop policy if exists "Users can view their own test attempts" on public.test_attempts;
drop policy if exists "Users can create their own test attempts" on public.test_attempts;
drop policy if exists "Users can update their own test attempts" on public.test_attempts;
create policy "Users can view their own test attempts"
on public.test_attempts for select to authenticated
using ((select auth.uid()) = user_id);
create policy "Users can create their own test attempts"
on public.test_attempts for insert to authenticated
with check ((select auth.uid()) = user_id);
create policy "Users can update their own test attempts"
on public.test_attempts for update to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

drop policy if exists "Users can view their own test questions" on public.test_questions;
drop policy if exists "Users can create their own test questions" on public.test_questions;
create policy "Users can view their own test questions"
on public.test_questions for select to authenticated
using (
  exists (
    select 1 from public.test_attempts t
    where t.id = test_attempt_id
      and t.user_id = (select auth.uid())
  )
);
create policy "Users can create their own test questions"
on public.test_questions for insert to authenticated
with check (
  exists (
    select 1 from public.test_attempts t
    where t.id = test_attempt_id
      and t.user_id = (select auth.uid())
  )
);

drop policy if exists "Users can view their own question attempts" on public.question_attempts;
drop policy if exists "Users can create their own question attempts" on public.question_attempts;
create policy "Users can view their own question attempts"
on public.question_attempts for select to authenticated
using ((select auth.uid()) = user_id);
create policy "Users can create their own question attempts"
on public.question_attempts for insert to authenticated
with check ((select auth.uid()) = user_id);

drop policy if exists "Users can view their own concept mastery" on public.concept_mastery;
drop policy if exists "Users can create their own concept mastery" on public.concept_mastery;
drop policy if exists "Users can update their own concept mastery" on public.concept_mastery;
create policy "Users can view their own concept mastery"
on public.concept_mastery for select to authenticated
using ((select auth.uid()) = user_id);
create policy "Users can create their own concept mastery"
on public.concept_mastery for insert to authenticated
with check ((select auth.uid()) = user_id);
create policy "Users can update their own concept mastery"
on public.concept_mastery for update to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

revoke all on table public.curriculum_versions from anon;
revoke all on table public.chapters from anon;
revoke all on table public.concepts from anon;
revoke all on table public.content_sources from anon, authenticated;
revoke all on table public.ingestion_jobs from anon, authenticated;
revoke all on table public.question_bank from anon, authenticated;
revoke all on table public.test_attempts from anon;
revoke all on table public.test_questions from anon;
revoke all on table public.question_attempts from anon;
revoke all on table public.concept_mastery from anon;

grant select on table public.curriculum_versions to authenticated;
grant select on table public.chapters to authenticated;
grant select on table public.concepts to authenticated;
grant select on table public.test_attempts to authenticated;
grant insert, update on table public.test_attempts to authenticated;
grant select, insert on table public.test_questions to authenticated;
grant select, insert on table public.question_attempts to authenticated;
grant select, insert, update on table public.concept_mastery to authenticated;

-- The backend's trusted service client needs to publish content and read the
-- answer key while the browser only receives the column-limited grants below.
grant all privileges on table public.curriculum_versions to service_role;
grant all privileges on table public.chapters to service_role;
grant all privileges on table public.concepts to service_role;
grant all privileges on table public.content_sources to service_role;
grant all privileges on table public.ingestion_jobs to service_role;
grant all privileges on table public.question_bank to service_role;
grant all privileges on table public.test_attempts to service_role;
grant all privileges on table public.test_questions to service_role;
grant all privileges on table public.question_attempts to service_role;
grant all privileges on table public.concept_mastery to service_role;

-- Column-level access deliberately excludes question_bank.correct_option_id.
grant select (
  id,
  curriculum_version_id,
  chapter_id,
  concept_id,
  question_type,
  question_text,
  options_json,
  explanation,
  hint,
  difficulty,
  cognitive_level,
  skill_tags,
  misconception_tags,
  question_style,
  estimated_time_seconds,
  marks,
  source_id,
  quality_score,
  status,
  published_at,
  created_at
) on table public.question_bank to authenticated;
