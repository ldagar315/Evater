-- Small, deterministic fixtures for local stage resets only.
-- The question-bank pack is published separately with `npm run stage:seed`.

insert into storage.buckets (id, name, public, file_size_limit)
values
  ('question-papers-test', 'question-papers-test', true, 52428800),
  ('answer-sheet-test', 'answer-sheet-test', true, 52428800)
on conflict (id) do update
set public = excluded.public,
    file_size_limit = excluded.file_size_limit;

insert into public."Chapter_contents" (grade, subject, board, chapter, summary)
select fixture.grade, fixture.subject, fixture.board, fixture.chapter, fixture.summary
from (
  values (
    '8',
    'Science',
    'NCERT/CBSE',
    'Exploring the Investigative World of Science',
    'A local stage fixture for questions, variables, observations, measurement, and evidence. Run npm run stage:seed to publish the full deterministic practice bank.'
  )
) as fixture(grade, subject, board, chapter, summary)
where not exists (
  select 1
  from public."Chapter_contents" existing
  where existing.grade = fixture.grade
    and existing.subject = fixture.subject
    and existing.chapter = fixture.chapter
);
