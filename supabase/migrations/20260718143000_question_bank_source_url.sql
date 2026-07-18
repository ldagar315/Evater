-- Preserve the exact source URL for each ingested question. The curriculum
-- anchor in content_sources is not sufficient when a pack combines multiple
-- public question-paper and answer-key sources.

alter table public.question_bank
  add column if not exists source_url text;

-- This is provenance for trusted services/editorial tools, not learner UI
-- content. The existing column-level authenticated grant intentionally does
-- not expose it to students.
revoke select (source_url) on table public.question_bank from anon, authenticated;
grant select (source_url) on table public.question_bank to service_role;
