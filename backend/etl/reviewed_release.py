"""Offline exact-release SQL builder. Does not connect, publish, or establish authority.

The generic fixed-100 publishing validator is deliberately not used or changed.
This path only accepts the separately reviewed, fixed 80-item release.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5

from .reviewed_batch import ReviewedBatchError, prepare_review_rows

BATCH = 'class8-original-reviewed-2026-11-14-v1'
DIGEST = '1fd41b5c0001dfe16b5368b541b263b2908eff99c0fa2cfc2f3071597dd111be'
NAMESPACE = UUID('a38169dc-6c56-4e21-b59a-50c1d5e93b29')


def publication_rows(directory: str | Path) -> list[dict]:
    directory = Path(directory)
    if hashlib.sha256((directory / 'questions.json').read_bytes()).hexdigest() != DIGEST:
        raise ReviewedBatchError('Not the exact reviewed 80-item release.')
    rows = prepare_review_rows(directory)
    if len(rows) != 80:
        raise ReviewedBatchError('Release must contain exactly 80 items.')
    for row in rows:
        row['id'] = str(uuid5(NAMESPACE, BATCH + '/' + row['source_question_id']))
        row['status'] = 'published'
        # Publication is not measured quality or an independent teacher review.
        row['quality_score'] = None
        row['reviewed_by'] = None
        row['reviewed_at'] = None
        row['generation_spec'] = {
            'release_id': BATCH,
            'reviewed_payload_sha256': DIGEST,
            'provenance': 'Original Evater practice; source_locator is curriculum/fact reference, not copied-question source.',
            'review_limits': 'Subject/editorial review, not independent teacher certification or learner-calibrated difficulty.',
        }
    return rows


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_transaction(directory: str | Path, *, commit: bool = False) -> str:
    """Build a fail-closed SQL transaction; default is a production-safe ROLLBACK.

    Operator must separately verify current user authority and deployment readiness.
    Existing rows are never updated, deleted, replaced or republished. Replay is
    accepted only when all 80 rows match every controlled column exactly.
    """
    rows = publication_rows(directory)
    payload = _literal(json.dumps(rows, separators=(',', ':'), ensure_ascii=False))
    columns = sorted(rows[0])
    quoted = ','.join('"' + c + '"' for c in columns)
    # Ignore only DB-managed creation/publication times when comparing a replay.
    expected = 'jsonb_build_object(' + ','.join(_literal(c) + ',to_jsonb(q)->' + _literal(c) for c in columns) + ')'
    schema = json.loads(Path(__file__).with_name('reviewed_release_schema.json').read_text())
    schema_columns = _literal(json.dumps(schema['columns']))
    schema_constraints = _literal(json.dumps(schema['constraints']))
    ending = 'COMMIT;' if commit else 'ROLLBACK;'
    return f'''-- Exact reviewed release: {BATCH}
-- This file cannot establish owner authority. No general publication gates change.
-- Default output ends ROLLBACK. Confirm current authority and deployed UI first.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '30s';
LOCK TABLE public.question_bank IN SHARE ROW EXCLUSIVE MODE;
CREATE TEMP TABLE exact80_expected ON COMMIT DROP AS
SELECT * FROM jsonb_populate_recordset(NULL::public.question_bank, {payload}::jsonb);
DO $release$
DECLARE matching integer; overlap_count integer;
BEGIN
  IF (SELECT jsonb_agg(to_jsonb(c) ORDER BY column_name) FROM
      (SELECT column_name,data_type,udt_name,is_nullable,column_default,numeric_precision,numeric_scale,character_maximum_length
       FROM information_schema.columns WHERE table_schema='public' AND table_name='question_bank') c)
       IS DISTINCT FROM {schema_columns}::jsonb THEN
    RAISE EXCEPTION 'Question bank column schema changed: repeat preflight';
  END IF;
  IF (SELECT jsonb_agg(d ORDER BY d) FROM
      (SELECT pg_get_constraintdef(oid) d FROM pg_constraint WHERE conrelid='public.question_bank'::regclass) c)
       IS DISTINCT FROM {schema_constraints}::jsonb THEN
    RAISE EXCEPTION 'Question bank constraints changed: repeat preflight';
  END IF;
  IF (SELECT count(*) FROM exact80_expected) <> 80
     OR (SELECT count(DISTINCT id) FROM exact80_expected) <> 80
     OR (SELECT count(DISTINCT content_hash) FROM exact80_expected) <> 80 THEN
    RAISE EXCEPTION 'Exact80 release identity failed';
  END IF;
  IF EXISTS (
    SELECT 1 FROM exact80_expected e
    LEFT JOIN public.concepts c ON c.id=e.concept_id
    LEFT JOIN public.chapters h ON h.id=e.chapter_id
    LEFT JOIN public.curriculum_versions v ON v.id=e.curriculum_version_id
    WHERE c.id IS NULL OR h.id IS NULL OR v.id IS NULL
      OR c.chapter_id<>e.chapter_id OR h.curriculum_version_id<>e.curriculum_version_id
      OR c.status<>'published' OR h.status<>'published' OR v.status<>'published'
  ) THEN RAISE EXCEPTION 'Live curriculum mapping/status mismatch'; END IF;
  IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='public.question_bank'::regclass AND NOT tgisinternal) THEN
    RAISE EXCEPTION 'Unexpected production trigger: repeat preflight';
  END IF;
  SELECT count(*) INTO overlap_count FROM public.question_bank q
  WHERE EXISTS (SELECT 1 FROM exact80_expected e WHERE q.id=e.id
     OR (q.chapter_id=e.chapter_id AND q.content_hash=e.content_hash)
     OR q.source_question_id=e.source_question_id);
  SELECT count(*) INTO matching FROM public.question_bank q
  JOIN exact80_expected e ON q.id=e.id
  WHERE {expected} = (SELECT jsonb_object_agg(k,to_jsonb(e)->k) FROM unnest(ARRAY[{','.join(_literal(c) for c in columns)}]) k)
    AND q.published_at IS NOT NULL;
  IF overlap_count NOT IN (0,80) OR matching<>overlap_count THEN
    RAISE EXCEPTION 'Partial release or conflicting existing record: no writes';
  END IF;
  IF overlap_count=0 THEN
    INSERT INTO public.question_bank ({quoted},published_at)
    SELECT {quoted},transaction_timestamp() FROM exact80_expected;
  END IF;
  SELECT count(*) INTO matching FROM public.question_bank q
  JOIN exact80_expected e ON q.id=e.id
  WHERE {expected} = (SELECT jsonb_object_agg(k,to_jsonb(e)->k) FROM unnest(ARRAY[{','.join(_literal(c) for c in columns)}]) k)
    AND q.published_at IS NOT NULL;
  IF matching<>80 THEN RAISE EXCEPTION 'Post-insert exact row verification failed'; END IF;
END $release$;
SELECT e.concept_id,count(*) AS exact_release_published
FROM exact80_expected e JOIN public.question_bank q ON q.id=e.id
WHERE q.status='published' GROUP BY e.concept_id ORDER BY e.concept_id;
{ending}
'''


def build_retirement_transaction(directory: str | Path, *, commit: bool = False) -> str:
    """Prepare rollback by retirement, never deletion, preserving existing attempts.

    Separate operator authority and current-row review are required to execute.
    Every controlled field except status must still match the release.
    """
    rows = publication_rows(directory)
    payload = _literal(json.dumps(rows, separators=(',', ':'), ensure_ascii=False))
    keys = sorted(k for k in rows[0] if k != 'status')
    keys_sql = ','.join(_literal(k) for k in keys)
    ending = 'COMMIT;' if commit else 'ROLLBACK;'
    return f'''-- Retirement rollback of exact release {BATCH}. No DELETE or attempt edits.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '30s';
LOCK TABLE public.question_bank IN SHARE ROW EXCLUSIVE MODE;
CREATE TEMP TABLE exact80_expected ON COMMIT DROP AS
SELECT * FROM jsonb_populate_recordset(NULL::public.question_bank, {payload}::jsonb);
DO $retire$
DECLARE matching integer;
BEGIN
  SELECT count(*) INTO matching FROM public.question_bank q
  JOIN exact80_expected e ON e.id=q.id
  WHERE q.status IN ('published','retired') AND q.published_at IS NOT NULL
    AND (SELECT jsonb_object_agg(k,to_jsonb(q)->k) FROM unnest(ARRAY[{keys_sql}]) k)
      = (SELECT jsonb_object_agg(k,to_jsonb(e)->k) FROM unnest(ARRAY[{keys_sql}]) k);
  IF matching<>80 THEN RAISE EXCEPTION 'Release rows changed: retirement blocked'; END IF;
  IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='public.question_bank'::regclass AND NOT tgisinternal) THEN
    RAISE EXCEPTION 'Unexpected production trigger: repeat preflight';
  END IF;
  UPDATE public.question_bank q SET status='retired'
  FROM exact80_expected e WHERE q.id=e.id AND q.status='published';
  IF (SELECT count(*) FROM public.question_bank q JOIN exact80_expected e ON e.id=q.id WHERE q.status='retired')<>80 THEN
    RAISE EXCEPTION 'Retirement postcheck failed';
  END IF;
END $retire$;
{ending}
'''


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Build offline exact80 SQL. Does not connect or grant execution authority.')
    parser.add_argument('directory', type=Path)
    parser.add_argument('--commit', action='store_true', help='Generate COMMIT instead of safe default ROLLBACK; operator must verify authority/readiness separately.')
    parser.add_argument('--retire', action='store_true', help='Generate exact-row retirement rollback, never deletion.')
    args = parser.parse_args()
    builder = build_retirement_transaction if args.retire else build_transaction
    print(builder(args.directory, commit=args.commit), end='')


if __name__ == '__main__':
    main()
