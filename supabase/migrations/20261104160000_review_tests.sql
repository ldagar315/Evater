-- Review tests: cross-chapter "weak spots" practice.
--
-- A review test draws questions from the student's weakest concepts across
-- chapters, so a review test_attempts row belongs to no single chapter.
-- chapter_id becomes nullable exactly for mode = 'review'; per-question rows
-- (question_attempts, concept_mastery, student_question_performance) keep
-- their own NOT NULL chapter ids resolved from each question.

alter table public.test_attempts
  alter column chapter_id drop not null;

alter table public.test_attempts
  drop constraint if exists test_attempts_mode_check;

alter table public.test_attempts
  add constraint test_attempts_mode_check
  check (mode in ('diagnostic', 'practice', 'remedial', 'challenge', 'review'));

alter table public.test_attempts
  add constraint test_attempts_review_chapter_check
  check ((mode = 'review') = (chapter_id is null));
