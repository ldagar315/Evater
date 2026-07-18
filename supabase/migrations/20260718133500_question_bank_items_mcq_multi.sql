-- Preserve multiple-select MCQs in the source archive as a first-class type.
alter table public.question_bank_items
  drop constraint if exists question_bank_items_question_type_check;

alter table public.question_bank_items
  add constraint question_bank_items_question_type_check
    check (question_type in (
      'mcq_single', 'mcq_multi', 'assertion_reason', 'true_false', 'fill_blank',
      'short_answer', 'long_answer', 'numerical', 'case_study', 'diagram_based',
      'matching', 'other'
    ));
