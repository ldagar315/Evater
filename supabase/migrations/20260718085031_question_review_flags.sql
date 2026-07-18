-- A learner can flag a published question for internal review without choosing
-- a reason. The unique pair makes the action idempotent if they click twice or
-- retry after a transient network failure.
create table if not exists public.question_review_flags (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  question_id uuid not null references public.question_bank(id) on delete cascade,
  created_at timestamptz not null default now(),
  unique (user_id, question_id)
);

create index if not exists question_review_flags_question_idx
  on public.question_review_flags (question_id, created_at desc);

alter table public.question_review_flags enable row level security;

drop policy if exists "Users can create their own question flags" on public.question_review_flags;
create policy "Users can create their own question flags"
  on public.question_review_flags
  for insert
  to authenticated
  with check ((select auth.uid()) = user_id);

drop policy if exists "Users can view their own question flags" on public.question_review_flags;
create policy "Users can view their own question flags"
  on public.question_review_flags
  for select
  to authenticated
  using ((select auth.uid()) = user_id);

revoke all on table public.question_review_flags from anon;
grant select, insert on table public.question_review_flags to authenticated;
grant all privileges on table public.question_review_flags to service_role;
