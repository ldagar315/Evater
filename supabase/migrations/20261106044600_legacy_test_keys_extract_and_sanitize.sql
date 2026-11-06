-- Legacy data migration: move answer keys out of the client-readable
-- Questions_Created.test column into the server-only key store, link the
-- rows, then strip the keys from the client-readable copy. Idempotent.

-- 1) Copy keys into the key store (skips rows already migrated).
with keyed_rows as (
  select qc.id, qc.created_by, qc.test
  from public."Questions_Created" qc
  where exists (
    select 1
    from jsonb_array_elements(qc.test) as q,
         jsonb_array_elements(q -> 'options') as o
    where o ? 'is_correct'
  )
  and not exists (
    select 1 from public.generated_test_keys k
    where k.questions_created_id = qc.id
  )
), inserted as (
  insert into public.generated_test_keys (user_id, questions_created_id, questions)
  select created_by, id, test from keyed_rows
  returning id, questions_created_id
)
update public."Questions_Created" qc
set key_id = i.id
from inserted i
where i.questions_created_id = qc.id;

-- 2) Strip is_correct from the client-readable copy once a key-store row
--    exists for the test.
update public."Questions_Created" qc
set test = (
  select jsonb_agg(
    case
      when q ? 'options' then jsonb_set(
        q, '{options}',
        coalesce(
          (select jsonb_agg(o - 'is_correct') from jsonb_array_elements(q -> 'options') o),
          '[]'::jsonb
        )
      )
      else q
    end
  )
  from jsonb_array_elements(qc.test) as q
)
where qc.key_id is not null
  and exists (
    select 1
    from jsonb_array_elements(qc.test) as q,
         jsonb_array_elements(q -> 'options') as o
    where o ? 'is_correct'
  );

-- ROLLBACK (run manually, never as part of db push):
--   update public."Questions_Created" qc
--   set test = k.questions, key_id = null
--   from public.generated_test_keys k
--   where k.questions_created_id = qc.id;
-- The key store remains the canonical copy; dropping it is NOT part of rollback.
