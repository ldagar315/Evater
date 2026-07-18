alter table public.question_bank_items
  add column if not exists generation_spec jsonb not null default '{}'::jsonb;

alter table public.question_bank_items
  drop constraint if exists question_bank_items_generation_spec_object_check;

alter table public.question_bank_items
  add constraint question_bank_items_generation_spec_object_check
    check (jsonb_typeof(generation_spec) = 'object');
