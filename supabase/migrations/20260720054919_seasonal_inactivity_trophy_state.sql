-- Trophy balances carry across seasons. These fields track the small amount
-- of state needed for seasonal inactivity protection and its 50% floor.

alter table public.leaderboard_players
  add column if not exists inactive_seasons integer not null default 0,
  add column if not exists inactivity_baseline_points integer,
  add column if not exists inactivity_floor_reached boolean not null default false;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'leaderboard_players_inactive_seasons_check'
  ) then
    alter table public.leaderboard_players
      add constraint leaderboard_players_inactive_seasons_check
      check (inactive_seasons >= 0);
  end if;

  if not exists (
    select 1
    from pg_constraint
    where conname = 'leaderboard_players_inactivity_baseline_points_check'
  ) then
    alter table public.leaderboard_players
      add constraint leaderboard_players_inactivity_baseline_points_check
      check (inactivity_baseline_points is null or inactivity_baseline_points >= 0);
  end if;
end
$$;
