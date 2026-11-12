# Local mixed-paper HTTP review slice

Review-only patch against draft PR87 head `ce02ddd33794791a3912940f49f21fff098fea44`.

## Boundary

This adds a separate loopback-only review app. It imports the unchanged actual `review_router`, `MixedSamplePaper` response contract, and actual `MixedSamplePaperReview` React component. The browser now fetches JSON over a real local HTTP socket, rather than receiving a direct render fixture. Neither `app.main`, production routes, public hosting rules nor the database is changed. The harness has no DB client, credentials, generation call, payment path, submission path or score writer. It is not a production integration or a student release.

`backend/review_tools/mixed_http_review.py` compares supplied bank IDs, hashes, chapter IDs, types, prompts, options, published state and answer representations to explicit reviewed inputs. It checks live-verification metadata and a maximum 15-minute age for non-synthetic snapshots. This metadata is supplied by the reviewer: the patch does not itself perform a live bank fetch. The caller must earn that claim with a current source-of-truth read. It does not make a snapshot live by naming it so.

Source-answer prose can differ in representation from an option letter. Only source answers that are explicit option labels are compared as keys. Other prose is not independently verified by this validator. No answer normalization or grading behavior is changed.

Readiness holds and strict contract errors produce HTTP503 with paths/reasons, not paper content. Missing/current-unverified data never falls back to fixtures. Unknown ID returns404. Loader failures return sanitized503. The frontend clears the old successful paper before every selection and handles transport failures without leaving stale content visible. Cache is disabled and robots are excluded.

## Reproduce

Use an environment with repository backend dependencies plus `uvicorn`, `pytest`, `httpx`; use the frontend's installed dependencies and Chrome. Do not put secrets in the input file.

```sh
cd frontend
./node_modules/.bin/esbuild review-tools/MixedHttpReview.tsx --bundle --jsx=automatic --outfile=review-tools/review.js --format=esm
./node_modules/.bin/tsc -p review-tools/tsconfig.json
cd ..
PYTHONPATH=backend python -m pytest tests/test_mixed_http_review.py tests/test_mixed_sample_paper.py tests/test_sample_paper_responses.py -q
REVIEW_EVIDENCE=/absolute/path/evidence REVIEW_INPUTS=/absolute/path/private-inputs.json REVIEW_PYTHON=/absolute/path/python node frontend/review-tools/check-http.mjs
```

Input file shape is `{"papers": {"id": draft}, "snapshots": {"id": snapshot}}`. An empty `{"papers": {}, "snapshots": {}}` runs the synthetic success, unknown-ID and loader-failure manual controls; the full browser suite expects the private held `light`, `electricity`, `maths` inputs. The server always binds127.0.0.1. It terminates at the end of the browser suite. To review manually:

```sh
PYTHONPATH=backend python -m review_tools.serve_mixed_review --inputs /absolute/path/private-inputs.json
```

## Evidence on Nov12

- 37 new review tests plus28 prior contract/response tests:65 PASS. Optional private held-input test was included. One upstream TestClient deprecation warning.
- Focused harness/component TypeScript check PASS. Not a clean full-frontend typecheck claim.
- 24 browser cases PASS at320/390/1280: synthetic10+5 layout and disclosures, all three real held-input refusals, unknown404, injected loader503, reload and transport-failure stale-content removal. 21 actual response traces saved. No browser runtime errors on final run.
- Initial browser run caught a real harness build error, `React is not defined`, caused by classic JSX output for the unchanged component. Build was corrected to `--jsx=automatic` and the complete browser suite rerun. Initial failure evidence is preserved separately.
- Actual Light Q3 bank snapshot retains C while held/source answer is B. It returns503 and does not correct either input. All three held snapshots are from Nov11 22:10:55IST, marked not verified live, and refused as stale/unverified on Nov12.
- The held-to-contract adaptation intentionally leaves unverified subjective source pages/URLs and separately validated marking criteria unset. Contract errors concern this incomplete adaptation, not a claim that the original held review File contains no source references or rubric prose. No invented page, citation or rubric is inserted.

Private held question-bank snapshots and held prose are excluded from the patch. Only synthetic study content is committed. They are local evidence for failure tests, not a public curriculum artifact.

## Remaining release gates

A current live read must feed this boundary; current objective matches and teacher/source/rights reviews must be settled; subjective citations/pages/answers/criteria need verification; the real authenticated product integration, submission and scoring journey is not implemented here. Synthetic success establishes wiring only. The earlier default router503 and absence of production registration remain intentional. No merge or deploy is requested by this patch.

## Current-bank follow-up

Nov12 07:33:54IST:30 requested bank identities were read from the live authenticated Table Editor, count and exact ID set reconciled, then fed into the same local HTTP harness before the15-minute freshness window expired. 25 exact objective matches and5 conflicts:LightQ3answer;ElectricityQ2/Q3prompt wording,Q5prompt/options,Q9option punctuation. Punctuation differences are strict-identity conflicts, not a claim of mathematical error. All3heldpapers still return503 for actual unresolved source/teacher/rights/contract fields; current-bank stale/unverified errors are absent. This is a bounded manual live export, not a persistent product database reader. No data writes.

The fresh-feed run exposed Python3.10's rejection of ISO timestamps ending in `Z`; the parser now accepts standard UTC `Z`, with a regression test. 66 tests pass after the fix. The browser fresh-feed mode additionally asserts no invalid-time/stale/unverified error, so a generic503 is not enough to pass freshness validation. Evidence retains the error history; final screenshots show the corrected result.
