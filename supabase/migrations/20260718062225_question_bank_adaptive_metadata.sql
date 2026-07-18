-- Future-ready question metadata and per-student item performance.
-- All new question fields are nullable/defaulted so existing MCQ content keeps
-- working while ingestion evolves to support media, variants, and numerical items.

alter table public.question_bank
  add column if not exists question_family_key text,
  add column if not exists variant_key text,
  add column if not exists media_json jsonb not null default '[]'::jsonb,
  add column if not exists option_media_json jsonb not null default '{}'::jsonb,
  add column if not exists answer_spec jsonb not null default '{}'::jsonb,
  add column if not exists generation_spec jsonb not null default '{}'::jsonb,
  add column if not exists source_question_id text,
  add column if not exists source_locator text;

alter table public.question_bank
  drop constraint if exists question_bank_media_json_array_check,
  drop constraint if exists question_bank_option_media_json_object_check,
  drop constraint if exists question_bank_answer_spec_object_check,
  drop constraint if exists question_bank_generation_spec_object_check;

alter table public.question_bank
  add constraint question_bank_media_json_array_check
    check (jsonb_typeof(media_json) = 'array'),
  add constraint question_bank_option_media_json_object_check
    check (jsonb_typeof(option_media_json) = 'object'),
  add constraint question_bank_answer_spec_object_check
    check (jsonb_typeof(answer_spec) = 'object'),
  add constraint question_bank_generation_spec_object_check
    check (jsonb_typeof(generation_spec) = 'object');

create index if not exists question_bank_family_idx
  on public.question_bank (chapter_id, question_family_key)
  where question_family_key is not null;

create index if not exists question_bank_variant_idx
  on public.question_bank (question_family_key, variant_key)
  where question_family_key is not null and variant_key is not null;

-- Materialized student x question signals make repeated-miss selection cheap.
-- Raw history remains in question_attempts; this table is an adaptive-selection cache.
create table if not exists public.student_question_performance (
  user_id uuid not null references auth.users(id) on delete cascade,
  question_id uuid not null references public.question_bank(id) on delete cascade,
  chapter_id uuid not null references public.chapters(id) on delete cascade,
  attempt_count integer not null default 0 check (attempt_count >= 0),
  correct_count integer not null default 0 check (correct_count >= 0),
  wrong_count integer not null default 0 check (wrong_count >= 0),
  consecutive_wrong_count integer not null default 0 check (consecutive_wrong_count >= 0),
  last_answer_correct boolean,
  last_selected_option_id text check (last_selected_option_id in ('A', 'B', 'C', 'D')),
  last_response_time_ms integer check (last_response_time_ms is null or last_response_time_ms >= 0),
  first_attempted_at timestamptz,
  last_attempted_at timestamptz,
  last_correct_at timestamptz,
  last_wrong_at timestamptz,
  next_review_at timestamptz,
  updated_at timestamptz not null default now(),
  primary key (user_id, question_id)
);

create index if not exists student_question_performance_selection_idx
  on public.student_question_performance (
    user_id,
    chapter_id,
    consecutive_wrong_count desc,
    wrong_count desc,
    last_attempted_at desc
  );

alter table public.student_question_performance enable row level security;

drop policy if exists "Students can view their own question performance"
  on public.student_question_performance;
drop policy if exists "Students can create their own question performance"
  on public.student_question_performance;
drop policy if exists "Students can update their own question performance"
  on public.student_question_performance;

create policy "Students can view their own question performance"
on public.student_question_performance for select to authenticated
using ((select auth.uid()) = user_id);

create policy "Students can create their own question performance"
on public.student_question_performance for insert to authenticated
with check ((select auth.uid()) = user_id);

create policy "Students can update their own question performance"
on public.student_question_performance for update to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

revoke all on table public.student_question_performance from anon;
grant select, insert, update on table public.student_question_performance to authenticated;
grant all privileges on table public.student_question_performance to service_role;

-- Initialize the cache from existing attempt history. New submissions maintain it
-- in the backend so no historical attempt data needs to be re-scraped or lost.
with ordered_attempts as (
  select
    qa.user_id,
    qa.question_id,
    qa.chapter_id,
    qa.is_correct,
    qa.selected_option_id,
    qa.response_time_ms,
    qa.created_at,
    row_number() over (
      partition by qa.user_id, qa.question_id
      order by qa.created_at desc, qa.id desc
    ) as newest_first,
    sum(case when qa.is_correct then 1 else 0 end) over (
      partition by qa.user_id, qa.question_id
      order by qa.created_at desc, qa.id desc
      rows between unbounded preceding and current row
    ) as correct_seen_from_latest
  from public.question_attempts qa
),
latest_attempts as (
  select *
  from ordered_attempts
  where newest_first = 1
),
streaks as (
  select user_id, question_id, count(*)::integer as consecutive_wrong_count
  from ordered_attempts
  where not is_correct and correct_seen_from_latest = 0
  group by user_id, question_id
),
aggregates as (
  select
    user_id,
    question_id,
    (array_agg(chapter_id order by created_at desc))[1] as chapter_id,
    count(*)::integer as attempt_count,
    count(*) filter (where is_correct)::integer as correct_count,
    count(*) filter (where not is_correct)::integer as wrong_count,
    min(created_at) as first_attempted_at,
    max(created_at) as last_attempted_at,
    max(created_at) filter (where is_correct) as last_correct_at,
    max(created_at) filter (where not is_correct) as last_wrong_at
  from ordered_attempts
  group by user_id, question_id
)
insert into public.student_question_performance (
  user_id,
  question_id,
  chapter_id,
  attempt_count,
  correct_count,
  wrong_count,
  consecutive_wrong_count,
  last_answer_correct,
  last_selected_option_id,
  last_response_time_ms,
  first_attempted_at,
  last_attempted_at,
  last_correct_at,
  last_wrong_at
)
select
  aggregate.user_id,
  aggregate.question_id,
  aggregate.chapter_id,
  aggregate.attempt_count,
  aggregate.correct_count,
  aggregate.wrong_count,
  coalesce(streak.consecutive_wrong_count, 0),
  latest.is_correct,
  latest.selected_option_id,
  latest.response_time_ms,
  aggregate.first_attempted_at,
  aggregate.last_attempted_at,
  aggregate.last_correct_at,
  aggregate.last_wrong_at
from aggregates aggregate
join latest_attempts latest
  on latest.user_id = aggregate.user_id
  and latest.question_id = aggregate.question_id
left join streaks streak
  on streak.user_id = aggregate.user_id
  and streak.question_id = aggregate.question_id
on conflict (user_id, question_id) do update set
  chapter_id = excluded.chapter_id,
  attempt_count = excluded.attempt_count,
  correct_count = excluded.correct_count,
  wrong_count = excluded.wrong_count,
  consecutive_wrong_count = excluded.consecutive_wrong_count,
  last_answer_correct = excluded.last_answer_correct,
  last_selected_option_id = excluded.last_selected_option_id,
  last_response_time_ms = excluded.last_response_time_ms,
  first_attempted_at = excluded.first_attempted_at,
  last_attempted_at = excluded.last_attempted_at,
  last_correct_at = excluded.last_correct_at,
  last_wrong_at = excluded.last_wrong_at,
  updated_at = now();
