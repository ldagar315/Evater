-- Classroom and school membership foundation for scoped student leaderboards.
-- Membership is intentionally separate from the legacy free-form profile fields.

create table if not exists public.schools (
  id uuid primary key default gen_random_uuid(),
  name text not null check (char_length(trim(name)) between 2 and 160),
  slug text not null unique check (slug ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$'),
  created_at timestamptz not null default now()
);

create table if not exists public.classrooms (
  id uuid primary key default gen_random_uuid(),
  school_id uuid not null references public.schools(id) on delete cascade,
  name text not null check (char_length(trim(name)) between 1 and 80),
  grade smallint not null check (grade between 1 and 12),
  created_at timestamptz not null default now(),
  unique (school_id, name),
  unique (id, school_id)
);

create table if not exists public.student_enrollments (
  user_id uuid primary key references auth.users(id) on delete cascade,
  school_id uuid not null references public.schools(id) on delete restrict,
  classroom_id uuid not null references public.classrooms(id) on delete restrict,
  role text not null default 'student' check (role in ('student', 'teacher', 'admin')),
  joined_at timestamptz not null default now()
);

create index if not exists student_enrollments_school_idx
  on public.student_enrollments (school_id, joined_at);
create index if not exists student_enrollments_classroom_idx
  on public.student_enrollments (classroom_id, joined_at);

alter table public.schools enable row level security;
alter table public.classrooms enable row level security;
alter table public.student_enrollments enable row level security;

drop policy if exists "Students can view their school" on public.schools;
create policy "Students can view their school"
on public.schools for select to authenticated
using (
  exists (
    select 1
    from public.student_enrollments enrollment
    where enrollment.school_id = schools.id
      and enrollment.user_id = (select auth.uid())
  )
);

drop policy if exists "Students can view their classroom" on public.classrooms;
create policy "Students can view their classroom"
on public.classrooms for select to authenticated
using (
  exists (
    select 1
    from public.student_enrollments enrollment
    where enrollment.classroom_id = classrooms.id
      and enrollment.user_id = (select auth.uid())
  )
);

drop policy if exists "Students can view their enrollment" on public.student_enrollments;
create policy "Students can view their enrollment"
on public.student_enrollments for select to authenticated
using ((select auth.uid()) = user_id);

grant select on table public.schools to authenticated;
grant select on table public.classrooms to authenticated;
grant select on table public.student_enrollments to authenticated;

grant all privileges on table public.schools to service_role;
grant all privileges on table public.classrooms to service_role;
grant all privileges on table public.student_enrollments to service_role;

insert into public.schools (name, slug)
values
  ('Evater Local Stage', 'evater-local-stage'),
  ('North Star Academy', 'north-star-academy')
on conflict (slug) do update
set name = excluded.name;

insert into public.classrooms (school_id, name, grade)
select school.id, fixture.name, fixture.grade
from public.schools school
join (
  values
    ('evater-local-stage', 'Class 8A', 8),
    ('evater-local-stage', 'Class 8B', 8),
    ('north-star-academy', 'Class 8A', 8),
    ('north-star-academy', 'Class 9A', 9)
) as fixture(slug, name, grade) on fixture.slug = school.slug
on conflict (school_id, name) do update
set grade = excluded.grade;
