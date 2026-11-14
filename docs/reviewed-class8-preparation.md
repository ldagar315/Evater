# Class 8 reviewed content and controlled release preparation

This PR contains the exact 80 reviewed questions and three held guide previews. Ten reserve variants are excluded. It now includes an offline SQL transaction builder for the exact 80-question release. Merging the PR does not run SQL, publish questions, publish guides, or change production settings.

## Unchanged application behavior

- Existing blog catalog, sitemap and ordinary tests stay unchanged.
- The three guide previews exist only in Vite development mode at `/reviewed-content/lenses`, `/reviewed-content/power-play` and `/reviewed-content/invisible-living-world`.
- Production builds exclude their preview route, component and pending text. Production direct navigation returns the existing not-found page.
- The review adapter still produces review-status rows only. Its behavior is unchanged.
- The general fixed-100 publishing validator stays unchanged. No question padding, false difficulty labels or fake quality scores are used.

## Exact 80-question release path

`backend/etl/reviewed_release.py` is an offline SQL builder, not a database client or startup migration. It pins the reviewed question payload SHA-256 in code, independently of the editable manifest. Its deterministic IDs identify only this release. Per-question source fields distinguish original Evater practice from the NCERT curriculum/fact references. Quality score and named reviewer remain null: publication is not a measured quality score or independent teacher certification.

From the repository root, generate the default rehearsal transaction:

```sh
PYTHONPATH=backend python -m etl.reviewed_release content/reviewed/class8-2026-11-14 > exact80-rehearsal.sql
```

The default transaction ends in `ROLLBACK`. `--commit` changes the ending to `COMMIT`, but does not connect or execute it. Operator authority and deployment readiness must be checked separately before execution.

The transaction locks question_bank against concurrent writers for the short release, checks all 38 production column definitions and 25 constraints, rejects unexpected triggers, and checks published concept/chapter/version mappings. It permits either no existing release rows or an exact, complete 80-row replay. A partial release or any conflicting ID, chapter/hash, source-question ID, explanation, answer, provenance or other controlled field aborts the transaction. It does not overwrite or republish existing rows. Publication timestamps are assigned by PostgreSQL; replay preserves them and all other existing fields.

`--retire` generates a separate exact-row retirement rollback, also ending in `ROLLBACK` by default. It checks every controlled field except status before retiring the 80 rows. It never deletes questions or changes test attempts, so existing attempt references survive. A changed release row blocks retirement for inspection. A retired release cannot be silently republished by rerunning the publication transaction.

## Evidence

- 31 tests pass across the unchanged general ETL tests, review adapter and new release builder. Mutating question content and updating its manifest digest still fails the pinned release check.
- PostgreSQL 14.24 rehearsal used captured live 38-column/25-constraint definitions and real curriculum IDs/status/mappings. Parent tables were minimal, and execution used the local database owner. This does not prove production authentication/role parity.
- Rehearsal checks: default insert/rollback, commit 80, stable full-row replay, partial 79 rejection, changed explanation conflict, same hash with a different ID conflict, retired/mis-mapped concept, schema/constraint changes, post-insert failure rollback, default retirement rollback, retirement commit/replay, refusal to republish retired rows, and unchanged unrelated existing data. Report: `qa/reviewed-release-report.json`. Reproduce with `python qa/rehearse_reviewed_release.py --evidence PATH_TO_CAPTURED_PREFLIGHT`, against a disposable local PostgreSQL socket on port 55441. The script creates and removes its own uniquely named database; it has no remote database connection path.
- Earlier responsive guide previews and production exclusion checks passed. Full app TypeScript remained nonzero with the same 54 normalized diagnostics on base and proposal, not a clean typecheck.

## Still required for the real UI outcome

1. Owner merge of PR100 and this PR. No automated merge is included.
2. Verify actual deployed main files, Netlify production and Modal backend exact commits, public practice capabilities and live inventory. A ready preview is not production.
3. Re-read current production schema, mappings, counts, conflicts and source evidence immediately before executing the exact authorized 80-question transaction. Stop on differences; do not use the generic publishing path to work around them.
4. Verify the 80 published rows, then open the blog through the actual Evater UI and complete the exact-concept setup/start/answer/submit/results journey on the authorized QA account. Database presence alone is not acceptance.

Three guides remain held. Neither this release nor its selective guide previews claim full Class 8 coverage. Real existing-account Google-button login and unchanged-profile save/home redirect/reload were checked separately in production. Those checks do not establish first-time signup, email-confirmation delivery or new-profile insertion.
