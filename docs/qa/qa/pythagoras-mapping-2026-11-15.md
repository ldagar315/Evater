# Pythagoras practice mapping defect, 15 November 2026

## Observed failure

The production article's theorem concept link selected chapter
`c9106af8-b1a5-595a-9574-930eafe06b44` and concept
`bb523d0c-18fc-52cc-863c-32c45b76a9f8` correctly. A real practice Start
returned four exponent exercises under the theorem title. Read-only inspection
found the same chapter/concept IDs on the question-bank records themselves:

| Question ID | Stem |
| --- | --- |
| b225c4e4-ffc2-4e62-88b7-7e35b89d0907 | What is 2^5? |
| 7e624a7c-1816-4506-af57-501f3a7147ef | What is 4^4? |
| b3cec108-1f69-47cc-bde6-1fb44947d3e7 | What is 5^3? |
| 7f994656-136b-4d4e-9981-21050ca366a9 | What is 10^3? |

No answers or submission were used to conceal this failure. Readiness count,
selection, Start, content accuracy and scoring are separate checks. This test
established a content failure, not an end-to-end practice pass.

## Cause and code correction

`author_class8_maths.py` scheduled `gen_power` four times in chapter 9 and
explicitly remapped its `exponents` topic to `pythagoras-statement`. Independent
answer verification checked exponent arithmetic, not curriculum alignment.
A real valid chapter and concept UUID therefore carried unrelated content.

The correction removes this remap and replaces the four exponent candidates
with original theorem-statement MCQs. Their stems explicitly name a
right-angled triangle, hypotenuse and legs; their separate verifier reparses
those labels to derive the correct equation. Existing side, ladder, triples,
converse, true/false and numerical families retain their destinations.

An explicit chapter-9 family/concept allowlist rejects unrelated families and
wrong concept IDs at candidate generation and at the Maths seeder boundary.
All seed packs are checked before creating a remote service client, so a stale
exponent pack cannot retire rows before failing. The CLI dry-run checks the same
boundary. These checks are intentionally not a claim of full semantic validation
of arbitrary external packs or of other chapters.

## Production repair is NOT included or performed

Merging this PR does not rewrite the current bank, retire existing questions,
change the API, or fix today's stored rows. There is no automatic migration,
network publisher invocation, service credential, or live data write in this PR.
No deployed content is declared repaired by these offline tests.

The next data operation needs its own reviewed scope and approval. Prepare it
against a fresh read-only export first:

1. Inventory all chapter-9 published rows, their generator families, concept IDs,
   sources, stems and references. Confirm the four IDs above are still exactly
   the affected rows, and report any additional mismatches instead of broadening
   a repair silently. Inspect the other article-linked concept sets without
   creating new attempts.
2. Stage and review the four replacement theorem questions and answer options.
   The four variants use different variable labels but the same theorem skill;
   this fixes alignment, not question diversity or pedagogy coverage.
3. Prepare a single guarded transaction that verifies the exact old payloads and
   current status, inserts the approved replacement payloads and retires only
   the confirmed bad rows. Preserve every existing row ID and all historical
   test/answer references. Do not overwrite a historic question's text or move it
   to a different concept. Do not use chapter-wide `--replace-existing` for this
   narrow repair.
4. Rehearse the exact transaction and rollback on real PostgreSQL, checking
   schema constraints, references, partial-run safety, idempotence and concurrent
   reads. Obtain scoped approval before any live commit. None of that rehearsal
   or live mutation is represented as done here.
5. After approved publication, query the resulting published set, compare its
   full content and answers with the approved payload, verify all three existing
   article links, and complete one authorized correct-topic UI test through
   submission, result and history. Recheck account credit effects. Preserve
   existing unfinished QA history rather than deleting it.

## Validation scope

Regression tests reproduce the former correctly answered but wrongly mapped
exponent item, reject it even after forcing a valid theorem slug, test the
stale-pack publication boundary before any remote client exists, check the
correct theorem equations and distractor uniqueness across multiple seeds,
retain Power Play exponent generation, and check all fourteen current generator
catalogues for identity guard errors. They do not certify every bank row or run
against production.

### Commands and measured results

- `python -m pytest -q tests/test_authored_maths_alignment.py tests/test_class8_maths_catalog.py tests/test_etl.py`: 51 passed.
- `python -m pytest -q --ignore=tests/test_blog_hosting_rules.py --ignore=tests/test_cors.py`: 189 passed.
- Unfiltered `python -m pytest -q`: 193 passed, one CORS assertion failed,
  six hosting tests errored because this clean worktree has no prerender build.
- A separate untouched worktree at main `a0a8ad2482568f55fa299739568b8745018c0f0d`
  reproduced the same CORS failure and all six missing-build setup errors:
  157 passed, one failed, six errors. This is not a clean full-suite claim.
- `git diff --check`: passed.

No frontend or runtime API code is changed, and no frontend/hosted build or UI
repair is claimed for this authoring/seeding-only change.
