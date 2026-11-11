# Rough-surface answer correction proposal

Draft only. This file does not run a migration. Live-data changes require separate approval.

The existing published question asks what happens when light hits a rough surface. Its options are B: Diffused reflection and C: Real image, but the stored key is C. The linked source Q30 explanation supports B: https://www.tiwariacademy.in/ncert-solutions-class-8-science-curiosity-chapter-10/

The proposed data-only SQL changes one exact published row, its source-answer letter, and its content hash. It preserves question/options, all review and rights holds, and all historical scores. It fails unless exactly one row matches the old hash/key/prompt/options and expected answer-spec shape. Default final statement is ROLLBACK. This is deliberately outside supabase/migrations.

Offline tests reproduce the wrong key with the existing scorer and test the proposed B key, unchanged metadata, both ETL-convention hashes, and narrow rollback safeguards. Passing these tests is not PostgreSQL or production-schema rehearsal. No live fix or historical repair is claimed.

Run from repository root: python -m pytest --import-mode=importlib docs/drafts/rough-surface-key/5-test_light_key_review.py -q

The uploaded fixture names have numeric prefixes: 1-correct-one-light-answer.DRAFT.sql, 2-published-row.json and 3-corrected-row-proposal.json. The regression test references those exact filenames. These are review artifacts outside the migration path.
