# Class 8 Science MVP runbook

The local implementation is ready for the restored Supabase project `qfxrfwgipsygoozlnkwb`.
The old project `gmedkgtydknqfgpxwxpz` is no longer a usable endpoint.

## One-time remote setup

Run these commands from the repository root:

```bash
npx supabase login
npx supabase link --project-ref qfxrfwgipsygoozlnkwb
npx supabase db lint --linked --schema public
npx supabase db push --linked --yes
```

The CLI is installed as a project-local dependency, so `npx supabase` is the supported command in this repository.

## Current Class 8 catalog

This pipeline targets the current NCERT/CBSE Grade 8 *Curiosity* Science book
(2026–27). The official contents list 13 chapters, starting with [Exploring the
Investigative World of Science](https://ncert.nic.in/textbook/pdf/hecu101.pdf),
so Chapter 1 is not a made-up chapter. The local pack contains original practice
questions derived from chapter concepts; it does not copy textbook prose.

The canonical manifest is in `backend/etl/class8_science_catalog.py`. It assigns
stable chapter and concept IDs and keeps one official PDF URL per chapter.

## Generate and seed all chapters

The generator creates deterministic, inspectable packs for Chapters 2–13:

```bash
PYTHONPATH=backend python3 -m etl.generate_class8_science_packs
```

The generated packs are temporary ETL inputs under `tmp/` and are not the
runtime question store. Publishing writes the validated rows into Supabase.

Validate all 13 packs without writing to a database:

```bash
PYTHONPATH=backend python3 -m etl.seed_class8_science --chapters 1-13
```

After checking the source/rights decision, publish to the configured Supabase
project:

```bash
PYTHONPATH=backend python3 -m etl.seed_class8_science \
  --chapters 1-13 --publish --acknowledge-source-review
```

The command is idempotent. It upserts the curriculum version, 13 chapters,
concepts, source provenance, ingestion jobs, and the selected pack. The
100-question gate is for the original synthetic MVP packs; source-backed packs
are validated with a minimum of 10 candidates per chapter. Student practice is
not blocked on an in-app review workflow: a learner can flag a question, and
the review team can inspect the question ID afterward. The source/license
acknowledgement remains a publish-time safeguard.

For local stage, the equivalent one-command workflow is:

```bash
npm run stage:seed:science
```

This downloads the current NCERT chapter PDFs, scrapes the chapter MCQ pages,
downloads the available 2025–26 KV practice papers and answer keys, extracts
and deduplicates the MCQs, and retains the official end-of-chapter/opening
prompts in the `question_bank_items` archive. The archive includes MCQs,
assertion/reason, true/false, fill-in-the-blank, short/long answer, numerical,
case-study, matching, and diagram-referenced items. Each item keeps the exact
source URL, source locator, answer-key text when available, and a review-required
media reference for diagrams. The learner-facing `question_bank` and
`/api/v1/tests` support deterministic single- and multiple-select MCQ,
assertion/reason, true/false, fill-in-the-blank, numerical, case-study,
matching, and diagram-based questions. Short and long answers remain
archive-only until a semantic evaluator is introduced.
If a chapter has fewer than 60 direct MCQ candidates, the stage crawl adapts
previous-chapter question structures to that chapter's concept manifest until
the 60-question floor is reached. This is a floor, not a cap: the source count
and adapted count are both reported, and adapted MCQs are available in local /
stage practice while their provenance remains attached for internal review.
The crawl writes its files and hashes under `tmp/class8-science-source-run/`.
For local/stage publishing, scraped MCQs are available immediately after the
publish command; the archive still retains source and review metadata for
internal inspection, and no reason needs to be collected from the learner.

## Seed the first chapter

The seed command is dry-run by default. It validates exactly 100 questions before any write:

```bash
PYTHONPATH=backend python3 -m etl.seed_first_chapter
```

After checking the source/rights decision, set `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and the other backend variables in a local uncommitted environment, then run:

```bash
PYTHONPATH=backend python3 -m etl.seed_first_chapter \
  --publish --acknowledge-source-review
```

The command is idempotent. It upserts the curriculum version, chapter, concepts, source provenance, ingestion job, and 100 published MCQs.

## Modal deployment

The backend answer key is intentionally unavailable to browser clients. The Modal secret named `evater-supabase-config` must contain:

```text
SUPABASE_URL=https://qfxrfwgipsygoozlnkwb.supabase.co
SUPABASE_API_KEY=<restored-project-publishable-or-anon-key>
SUPABASE_SERVICE_ROLE_KEY=<server-only-service-role-key>
```

Update that existing secret through the Modal dashboard or CLI without committing the values, then deploy:

```bash
modal deploy backend/application.py
```

The frontend practice flow is available at `/practice` after deployment. It uses `/api/v1/tests` and `/api/v1/tests/{test_id}/blocks/{block_number}/submit` and requires a signed-in Supabase user.

## Verification

Local checks:

```bash
pytest -q
cd frontend && npm run build
```

Remote smoke checks after migration, seed, and deployment:

1. Open Evater and sign in.
2. Open `/practice`.
3. Start the test and answer the five-question block.
4. Submit and confirm the score, mastery update, and routed next block.

All packs contain original question text with an NCERT source reference. They are
not copies of textbook prose; commercial use still requires the source/rights
review recorded in the seed metadata. Generated packs are suitable for local
stage testing, but should receive subject-matter review before production use.
