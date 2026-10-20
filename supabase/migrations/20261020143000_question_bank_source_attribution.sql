-- Source attribution display for close-to-source question-bank items.
-- Holds the public badge string (e.g. the NCERT attribution line). Private
-- provenance (source_url, locator) stays in existing columns / content_sources.
alter table public.question_bank
  add column if not exists source_tier text,
  add column if not exists source_attribution_display text;

comment on column public.question_bank.source_attribution_display is
  'Public attribution badge text for close-to-source items (e.g. NCERT); null for Evater-original items.';
