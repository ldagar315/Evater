-- Multiple-select MCQs use answer_spec.correct_option_ids. They remain fully
-- deterministic and are distinct from free-form short/long responses.
alter table public.question_bank
  drop constraint if exists question_bank_question_type_check;

alter table public.question_bank
  add constraint question_bank_question_type_check
    check (question_type in (
      'mcq_single', 'mcq_multi', 'assertion_reason', 'true_false', 'fill_blank',
      'numerical', 'case_study', 'diagram_based', 'matching'
    ));
