alter table public."Users" enable row level security;
alter table public."Chapter_contents" enable row level security;
alter table public."Questions_Created" enable row level security;
alter table public."AnswerSheetImages" enable row level security;
alter table public."FeedbackTest" enable row level security;
alter table public."Feedback_table_2" enable row level security;

drop policy if exists "Users can insert their own profile" on public."Users";
drop policy if exists "Users can update their own profile" on public."Users";
drop policy if exists "Users can view their own profile" on public."Users";

create policy "Users can view their own profile"
on public."Users" for select to authenticated
using ((select auth.uid()) = created_by);

create policy "Users can insert their own profile"
on public."Users" for insert to authenticated
with check ((select auth.uid()) = created_by);

create policy "Users can update their own profile"
on public."Users" for update to authenticated
using ((select auth.uid()) = created_by)
with check ((select auth.uid()) = created_by);

drop policy if exists "Enable read access for all users" on public."Chapter_contents";
drop policy if exists "Authenticated users can read chapter contents" on public."Chapter_contents";

create policy "Authenticated users can read chapter contents"
on public."Chapter_contents" for select to authenticated
using (true);

drop policy if exists "Users can view their own tests" on public."Questions_Created";
drop policy if exists "Users can create their own tests" on public."Questions_Created";

create policy "Users can view their own tests"
on public."Questions_Created" for select to authenticated
using ((select auth.uid()) = created_by);

create policy "Users can create their own tests"
on public."Questions_Created" for insert to authenticated
with check ((select auth.uid()) = created_by);

drop policy if exists "Users can view their own answer sheet records" on public."AnswerSheetImages";
drop policy if exists "Users can create their own answer sheet records" on public."AnswerSheetImages";

create policy "Users can view their own answer sheet records"
on public."AnswerSheetImages" for select to authenticated
using ((select auth.uid()) = uploaded_by);

create policy "Users can create their own answer sheet records"
on public."AnswerSheetImages" for insert to authenticated
with check ((select auth.uid()) = uploaded_by);

drop policy if exists "Users can view their own feedback" on public."FeedbackTest";
drop policy if exists "Users can create their own feedback" on public."FeedbackTest";

create policy "Users can view their own feedback"
on public."FeedbackTest" for select to authenticated
using ((select auth.uid()) = given_by);

create policy "Users can create their own feedback"
on public."FeedbackTest" for insert to authenticated
with check ((select auth.uid()) = given_by);

drop policy if exists "Users can submit general feedback" on public."Feedback_table_2";

create policy "Users can submit general feedback"
on public."Feedback_table_2" for insert to authenticated
with check ((select auth.uid()) = feedback_by);

grant select, insert, update on table public."Users" to authenticated;
grant select on table public."Chapter_contents" to authenticated;
grant select, insert on table public."Questions_Created" to authenticated;
grant select, insert on table public."AnswerSheetImages" to authenticated;
grant select, insert on table public."FeedbackTest" to authenticated;
grant insert on table public."Feedback_table_2" to authenticated;

revoke all on table public."Users" from anon;
revoke all on table public."Chapter_contents" from anon;
revoke all on table public."Questions_Created" from anon;
revoke all on table public."AnswerSheetImages" from anon;
revoke all on table public."FeedbackTest" from anon;
revoke all on table public."Feedback_table_2" from anon;

drop policy if exists "full-access 1hkhm8h_0" on storage.objects;
drop policy if exists "full-access 1hkhm8h_1" on storage.objects;
drop policy if exists "full-access 1hkhm8h_2" on storage.objects;
drop policy if exists "full-access 1hkhm8h_3" on storage.objects;
drop policy if exists "full-policy-test b1ln73_0" on storage.objects;
drop policy if exists "full-policy-test b1ln73_1" on storage.objects;
drop policy if exists "full-policy-test b1ln73_2" on storage.objects;
drop policy if exists "full-policy-test b1ln73_3" on storage.objects;
drop policy if exists "Question papers authenticated uploads" on storage.objects;
drop policy if exists "Question papers authenticated updates" on storage.objects;
drop policy if exists "Question papers authenticated deletes" on storage.objects;
drop policy if exists "Answer sheets authenticated uploads" on storage.objects;
drop policy if exists "Answer sheets authenticated updates" on storage.objects;
drop policy if exists "Answer sheets authenticated deletes" on storage.objects;

create policy "Question papers authenticated uploads"
on storage.objects for insert to authenticated
with check (
  bucket_id = 'question-papers-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

create policy "Question papers authenticated updates"
on storage.objects for update to authenticated
using (
  bucket_id = 'question-papers-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
)
with check (
  bucket_id = 'question-papers-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

create policy "Question papers authenticated deletes"
on storage.objects for delete to authenticated
using (
  bucket_id = 'question-papers-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

create policy "Answer sheets authenticated uploads"
on storage.objects for insert to authenticated
with check (
  bucket_id = 'answer-sheet-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

create policy "Answer sheets authenticated updates"
on storage.objects for update to authenticated
using (
  bucket_id = 'answer-sheet-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
)
with check (
  bucket_id = 'answer-sheet-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

create policy "Answer sheets authenticated deletes"
on storage.objects for delete to authenticated
using (
  bucket_id = 'answer-sheet-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);
