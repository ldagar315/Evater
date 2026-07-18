-- The learner bank can now serve every deterministic question type. Short and
-- long answers remain in question_bank_items until a semantic evaluator exists.
alter table public.question_bank
  drop constraint if exists question_bank_question_type_check,
  drop constraint if exists question_bank_correct_option_id_check,
  drop constraint if exists question_bank_explanation_check;

alter table public.question_bank
  alter column correct_option_id drop not null,
  alter column explanation drop not null,
  add column if not exists render_config jsonb not null default '{}'::jsonb,
  add constraint question_bank_question_type_check
    check (question_type in (
      'mcq_single', 'assertion_reason', 'true_false', 'fill_blank',
      'numerical', 'case_study', 'diagram_based', 'matching'
    )),
  add constraint question_bank_correct_option_id_check
    check (correct_option_id is null or correct_option_id in ('A', 'B', 'C', 'D')),
  add constraint question_bank_explanation_check
    check (explanation is null or char_length(trim(explanation)) between 5 and 2000),
  add constraint question_bank_render_config_object_check
    check (jsonb_typeof(render_config) = 'object');

alter table public.question_bank_items
  add column if not exists render_config jsonb not null default '{}'::jsonb;

alter table public.question_attempts
  add column if not exists answer_json jsonb not null default '{}'::jsonb;

alter table public.student_question_performance
  add column if not exists last_answer_json jsonb not null default '{}'::jsonb;

drop policy if exists "Users can create their own question attempts" on public.question_attempts;
create policy "Users can create their own question attempts"
on public.question_attempts for insert to authenticated
with check ((select auth.uid()) = user_id and jsonb_typeof(answer_json) = 'object');

revoke all on table public.question_bank_items from anon, authenticated;
grant all privileges on table public.question_bank_items to service_role;

grant select (render_config) on table public.question_bank to authenticated;
grant all privileges on table public.question_bank to service_role;
