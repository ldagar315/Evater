-- Server-only answer-key store for generated tests.
-- Keys are written by the backend at generation time and read back only by
-- the backend (service role) when grading feedback. Browsers never see them.

create table public.generated_test_keys (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  questions_created_id bigint null references public."Questions_Created"(id) on delete set null,
  questions jsonb not null,
  created_at timestamptz not null default now()
);

-- Legacy data migration links one key row per Questions_Created row.
create unique index generated_test_keys_questions_created_id_key
  on public.generated_test_keys (questions_created_id)
  where questions_created_id is not null;

alter table public.generated_test_keys enable row level security;
-- Deliberately no policies: anon and authenticated roles get NO access.
-- Only the service role (backend) can read or write answer keys.

-- Opaque, harmless reference stored on the client-readable test row.
alter table public."Questions_Created" add column if not exists key_id uuid null;
