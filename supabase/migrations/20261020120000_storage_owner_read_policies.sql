-- Owner-scoped reads for the user-content storage buckets.
-- Apply AFTER the signed-URL frontend PR is merged and ONLY when flipping
-- answer-sheet-test and question-papers-test to private.
--
-- Caveat: objects uploaded before user-id-prefixed paths existed
-- (top-level folders like "Ayuh-v2/" and "question-test-v1/") do not match
-- the owner-folder rule and will no longer be readable through policies.

create policy "Question papers owner read"
on storage.objects for select to authenticated
using (
  bucket_id = 'question-papers-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

create policy "Answer sheets owner read"
on storage.objects for select to authenticated
using (
  bucket_id = 'answer-sheet-test'
  and (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);
