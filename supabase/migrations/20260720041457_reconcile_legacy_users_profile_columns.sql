-- The hosted legacy Users table predates the canonical profile schema.
-- CREATE TABLE IF NOT EXISTS does not add columns to an existing table, so
-- reconcile the columns used by the current profile UI explicitly.
alter table public."Users"
  add column if not exists name text;

alter table public."Users"
  add column if not exists class_level integer;

update public."Users"
set name = nullif(trim(user_name), '')
where name is null
  and user_name is not null;

update public."Users"
set class_level = grade
where class_level is null
  and grade is not null;

notify pgrst, 'reload schema';
