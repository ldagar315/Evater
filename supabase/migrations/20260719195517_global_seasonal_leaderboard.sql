-- Global, server-owned seasonal league state.
-- Students are considered verified for leaderboard eligibility when they have
-- a student_enrollments row with role = 'student'. The backend is the only
-- writer; browsers never receive direct access to these tables.

create table if not exists public.leaderboard_seasons (
  id uuid primary key default gen_random_uuid(),
  season_key text not null unique,
  season_number bigint not null unique,
  starts_at timestamptz not null,
  ends_at timestamptz not null,
  status text not null default 'active'
    check (status in ('active', 'completed')),
  created_at timestamptz not null default now(),
  completed_at timestamptz,
  check (ends_at > starts_at)
);

create table if not exists public.leaderboard_players (
  season_id uuid not null references public.leaderboard_seasons(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  league_tier text not null default 'bronze_3'
    check (league_tier in (
      'bronze_3', 'bronze_2', 'bronze_1',
      'silver_3', 'silver_2', 'silver_1',
      'gold_3', 'gold_2', 'gold_1',
      'platinum_3', 'platinum_2', 'platinum_1',
      'diamond_3', 'diamond_2', 'diamond_1'
    )),
  final_tier text
    check (final_tier is null or final_tier in (
      'bronze_3', 'bronze_2', 'bronze_1',
      'silver_3', 'silver_2', 'silver_1',
      'gold_3', 'gold_2', 'gold_1',
      'platinum_3', 'platinum_2', 'platinum_1',
      'diamond_3', 'diamond_2', 'diamond_1'
    )),
  movement text
    check (movement is null or movement in ('promoted', 'demoted', 'held')),
  points integer not null default 0 check (points >= 0),
  completed_practices integer not null default 0 check (completed_practices >= 0),
  correct_answers integer not null default 0 check (correct_answers >= 0),
  last_activity_at timestamptz,
  last_decay_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (season_id, user_id)
);

create table if not exists public.leaderboard_question_scores (
  id uuid primary key default gen_random_uuid(),
  season_id uuid not null references public.leaderboard_seasons(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  question_id uuid not null references public.question_bank(id) on delete restrict,
  test_attempt_id uuid not null references public.test_attempts(id) on delete cascade,
  points_awarded integer not null,
  created_at timestamptz not null default now(),
  unique (season_id, user_id, question_id)
);

create table if not exists public.leaderboard_practice_scores (
  test_attempt_id uuid primary key references public.test_attempts(id) on delete cascade,
  season_id uuid not null references public.leaderboard_seasons(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  raw_score integer not null,
  points_awarded integer not null check (points_awarded >= 0),
  scored_question_count integer not null default 0 check (scored_question_count >= 0),
  correct_answers integer not null default 0 check (correct_answers >= 0),
  completed_at timestamptz not null,
  created_at timestamptz not null default now()
);

create index if not exists leaderboard_players_season_rank_idx
  on public.leaderboard_players (season_id, league_tier, points desc, correct_answers desc, completed_practices desc);
create index if not exists leaderboard_question_scores_user_season_idx
  on public.leaderboard_question_scores (user_id, season_id, created_at desc);
create index if not exists leaderboard_practice_scores_season_idx
  on public.leaderboard_practice_scores (season_id, completed_at desc);

alter table public.leaderboard_seasons enable row level security;
alter table public.leaderboard_players enable row level security;
alter table public.leaderboard_question_scores enable row level security;
alter table public.leaderboard_practice_scores enable row level security;

revoke all on table public.leaderboard_seasons from anon, authenticated;
revoke all on table public.leaderboard_players from anon, authenticated;
revoke all on table public.leaderboard_question_scores from anon, authenticated;
revoke all on table public.leaderboard_practice_scores from anon, authenticated;

grant all privileges on table public.leaderboard_seasons to service_role;
grant all privileges on table public.leaderboard_players to service_role;
grant all privileges on table public.leaderboard_question_scores to service_role;
grant all privileges on table public.leaderboard_practice_scores to service_role;
