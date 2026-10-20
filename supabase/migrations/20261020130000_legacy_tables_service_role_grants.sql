-- Data API grants for the six legacy tables created in
-- 20260713184029_rls_and_storage_policies.sql, which granted privileges to
-- authenticated only. From 2026-10-30 Supabase stops auto-granting Data API
-- access to new tables, and the same applies when migrations replay on
-- preview branches, new projects, or `supabase db reset`: without these
-- grants the service-role backend would get "permission denied" on
-- public."Users" and public."Chapter_contents" (both read by the backend).
--
-- Production is NOT affected on 2026-10-30 - existing tables keep their
-- current grants. This migration makes the replayed-schema path match.
-- Idempotent: GRANT is a no-op when the privilege already exists.
--
-- Pattern note: `all privileges` to service_role matches every later
-- migration in this repo (service_role is the trusted backend role; the
-- service key stays server-side). anon receives nothing, authenticated
-- keeps its existing scoped grants from the original migration.

grant all privileges on table public."Users" to service_role;
grant all privileges on table public."Chapter_contents" to service_role;
grant all privileges on table public."Questions_Created" to service_role;
grant all privileges on table public."AnswerSheetImages" to service_role;
grant all privileges on table public."FeedbackTest" to service_role;
grant all privileges on table public."Feedback_table_2" to service_role;
