-- Source archive for every extracted question shape. The learner-facing
-- question_bank remains MCQ-only for now; this table prevents non-MCQ source
-- material from being discarded while renderers and evaluators are built.

create table if not exists public.question_bank_items (
  id uuid primary key default gen_random_uuid(),
  curriculum_version_id uuid not null references public.curriculum_versions(id) on delete restrict,
  chapter_id uuid not null references public.chapters(id) on delete restrict,
  concept_id uuid not null references public.concepts(id) on delete restrict,
  ingestion_job_id uuid references public.ingestion_jobs(id) on delete set null,
  source_id uuid references public.content_sources(id) on delete set null,
  question_type text not null
    check (question_type in (
      'mcq_single', 'assertion_reason', 'true_false', 'fill_blank',
      'short_answer', 'long_answer', 'numerical', 'case_study',
      'diagram_based', 'matching', 'other'
    )),
  question_text text not null check (char_length(trim(question_text)) between 10 and 5000),
  options_json jsonb not null default '[]'::jsonb
    check (jsonb_typeof(options_json) = 'array'),
  explanation text,
  difficulty text not null default 'medium'
    check (difficulty in ('easy', 'medium', 'hard')),
  cognitive_level text not null default 'understand'
    check (cognitive_level in ('recall', 'understand', 'apply', 'analyze')),
  skill_tags text[] not null default '{}',
  question_style text not null default 'direct'
    check (question_style in ('direct', 'scenario', 'experiment', 'data', 'diagram')),
  estimated_time_seconds smallint not null default 60
    check (estimated_time_seconds between 10 and 1200),
  marks smallint not null default 1 check (marks > 0 and marks <= 20),
  media_json jsonb not null default '[]'::jsonb
    check (jsonb_typeof(media_json) = 'array'),
  option_media_json jsonb not null default '{}'::jsonb
    check (jsonb_typeof(option_media_json) = 'object'),
  answer_spec jsonb not null default '{}'::jsonb
    check (jsonb_typeof(answer_spec) = 'object'),
  generation_spec jsonb not null default '{}'::jsonb
    check (jsonb_typeof(generation_spec) = 'object'),
  source_url text not null,
  source_question_id text,
  source_locator text,
  content_hash text not null,
  license_status text not null default 'review_required',
  review_status text not null default 'review'
    check (review_status in ('draft', 'review', 'approved', 'rejected')),
  quality_score numeric(4,3) check (quality_score between 0 and 1),
  status text not null default 'review'
    check (status in ('draft', 'validated', 'review', 'published', 'rejected', 'retired')),
  created_at timestamptz not null default now(),
  unique (chapter_id, content_hash)
);

create index if not exists question_bank_items_chapter_type_status_idx
  on public.question_bank_items (chapter_id, question_type, status);
create index if not exists question_bank_items_source_idx
  on public.question_bank_items (source_id, source_question_id);

alter table public.question_bank_items enable row level security;

-- This is an editorial/archive table. It is intentionally service-role-only
-- until a review UI and non-MCQ learner renderer exist.
revoke all on table public.question_bank_items from anon, authenticated;
grant all privileges on table public.question_bank_items to service_role;
