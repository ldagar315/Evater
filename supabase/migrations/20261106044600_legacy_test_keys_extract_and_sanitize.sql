-- Legacy data migration: move answer keys out of the client-readable
-- Questions_Created.test column into the server-only key store, link the
-- rows, then rewrite every interpretable row to the v2 wrapper shape
-- ({"v":2,"questions":[...]}) with keys stripped from the client copy.
--
-- Why the wrapper: every legacy client guards question reads with
-- Array.isArray(test). An object shape is fail-VISIBLE on old clients
-- (error / empty state); a bare sanitized array is fail-SILENT (the old
-- TakeTestPage first-option fallback would mark option 1 correct).
--
-- Recognized shapes:
--   A) test is a jsonb array of question objects (common case)
--   B) test is a jsonb object with a "questions" array (older format)
--   C) test is already v2 (object with "v") - skipped
-- Anything else (null, scalar) is left untouched.
--
-- Idempotent: key extraction is guarded by key_id null + NOT EXISTS in the
-- key store; the rewrite skips v2 rows and only strips keys when they are
-- already safe in the store (or never existed).

-- 1) Copy keys into the key store (skips rows already migrated).
with legacy as (
  select qc.id, qc.created_by,
         case jsonb_typeof(qc.test)
           when 'array' then qc.test
           when 'object' then qc.test -> 'questions'
           else null
         end as questions
  from public."Questions_Created" qc
  where qc.key_id is null
    and not (jsonb_typeof(qc.test) = 'object' and qc.test ? 'v')
),
keyed_rows as (
  select l.id, l.created_by, l.questions
  from legacy l
  where l.questions is not null
    and jsonb_typeof(l.questions) = 'array'
    and exists (
      select 1
      from jsonb_array_elements(l.questions) as q,
           jsonb_array_elements(
             case
               when jsonb_typeof(q) = 'object' and jsonb_typeof(q -> 'options') = 'array'
               then q -> 'options'
               else '[]'::jsonb
             end
           ) as o
      where jsonb_typeof(o) = 'object'
        and (o ? 'is_correct' or o ? 'isCorrect')
    )
    and not exists (
      select 1 from public.generated_test_keys k
      where k.questions_created_id = l.id
    )
), inserted as (
  insert into public.generated_test_keys (user_id, questions_created_id, questions)
  select created_by, id, questions from keyed_rows
  returning id, questions_created_id
)
update public."Questions_Created" qc
set key_id = i.id
from inserted i
where i.questions_created_id = qc.id;

-- 2) Rewrite interpretable rows to the v2 wrapper, keys stripped from the
--    client-readable copy. A row is rewritten only when its keys are safe in
--    the store (key_id set) or it never had keys.
with legacy as (
  select qc.id, qc.key_id,
         case jsonb_typeof(qc.test)
           when 'array' then qc.test
           when 'object' then qc.test -> 'questions'
           else null
         end as questions
  from public."Questions_Created" qc
  where not (jsonb_typeof(qc.test) = 'object' and qc.test ? 'v')
),
flagged as (
  select l.id, l.key_id, l.questions,
         exists (
           select 1
           from jsonb_array_elements(l.questions) as q,
                jsonb_array_elements(
                  case
                    when jsonb_typeof(q) = 'object' and jsonb_typeof(q -> 'options') = 'array'
                    then q -> 'options'
                    else '[]'::jsonb
                  end
                ) as o
           where jsonb_typeof(o) = 'object'
             and (o ? 'is_correct' or o ? 'isCorrect')
         ) as has_keys
  from legacy l
  where l.questions is not null
    and jsonb_typeof(l.questions) = 'array'
),
sanitized as (
  select f.id,
         jsonb_build_object(
           'v', 2,
           'questions', (
             select jsonb_agg(
               case
                 when jsonb_typeof(q) = 'object'
                      and jsonb_typeof(q -> 'options') = 'array'
                 then jsonb_set(q, '{options}', coalesce((
                        select jsonb_agg(
                          case
                            when jsonb_typeof(o) = 'object'
                            then o - 'is_correct' - 'isCorrect'
                            else o
                          end
                        )
                        from jsonb_array_elements(q -> 'options') o
                      ), '[]'::jsonb))
                 else q
               end
             )
             from jsonb_array_elements(f.questions) q
           )
         ) as wrapped
  from flagged f
  where f.key_id is not null  -- keys safe in the store
     or not f.has_keys        -- nothing to strip
)
update public."Questions_Created" qc
set test = s.wrapped
from sanitized s
where s.id = qc.id;

-- ROLLBACK (run manually, never as part of db push):
--   update public."Questions_Created" qc
--   set test = k.questions, key_id = null
--   from public.generated_test_keys k
--   where k.questions_created_id = qc.id;
-- Restores the original bare-array shape WITH keys. The key store remains
-- the canonical copy; dropping it is NOT part of rollback.
