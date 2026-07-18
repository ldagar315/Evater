# Local stage environment

The local stage runs the same frontend, FastAPI routes, Supabase Auth, database, storage, and question-bank policies without deploying anything.

## Prerequisites

- Docker Desktop must be running.
- Node dependencies must be installed at the repository root and in `frontend/`.
- Python dependencies must be installed. A project virtualenv is recommended:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
npm install
npm --prefix frontend install
```

## Daily workflow

From the repository root:

```bash
npm run stage
```

This starts local Supabase, then the backend at `http://127.0.0.1:8000` and Vite at `http://127.0.0.1:5173`. Supabase credentials are read from the local CLI at runtime and passed only to the child processes; no stage `.env` file is generated.

AI-backed generation and evaluation routes still need the provider keys your backend uses. Export those in your shell or keep them in the existing local backend environment; the stage runner inherits them and does not commit or print them.

Use the app to create a local email/password account. Emails stay in Mailpit at `http://127.0.0.1:54324`. Supabase Studio is available at `http://127.0.0.1:54323`.

For a repeatable account with access to the seeded Class 8A fixtures, run:

```bash
npm run stage:account
```

This creates or refreshes `stage-admin@evater.test` with the local-only password `StageAdmin123!` and enrolls it as an admin in Class 8A. Override `EVATER_STAGE_ACCOUNT_EMAIL` and `EVATER_STAGE_ACCOUNT_PASSWORD` when needed.

The stage runner stops the frontend and backend when you press Ctrl-C, but leaves Supabase running so the next start is fast. Stop the containers when finished:

```bash
npm run stage:stop
```

## Reset and fixtures

To recreate the local database, apply every migration, create storage buckets, and insert the chapter-picker fixture:

```bash
npm run stage:reset
```

This is destructive to the local database only. To publish the deterministic Class 8 Science practice pack into the local question bank:

```bash
npm run stage:seed
```

To download, extract, validate, and publish the current Grade 8 *Curiosity*
Science source candidates across all 13 chapters:

```bash
npm run stage:seed:science
```

The full command is idempotent and also refreshes the stage admin's Class 8A
enrollment. It downloads the official NCERT curriculum PDFs, current public
chapter MCQ pages, and the available KV 2025–26 question-paper/answer-key
pairs. It stores source URLs, locators, hashes, and a review-required state on
every candidate. It also fills the service-role-only `question_bank_items`
archive with non-MCQ source items such as numericals, case studies, matching,
and diagram-referenced prompts. The learner-facing bank supports deterministic
single-/multiple-select MCQ, assertion/reason, true/false, fill-in-the-blank,
numerical, case-study, matching, and diagram-based questions. Short and long
answers remain archive-only until a semantic evaluator exists. The stage seed
also publishes one fixture of each deterministic non-MCQ type so the full UI
can be tested immediately; scraped items still require their own answer-key
and subject/rights review before learner publication. For chapters with thin direct coverage, the same run adapts
previous-chapter assessment structures to current-chapter concepts until a
configurable 60-question floor. Adapted items carry their source question ID,
source URL, target chapter, and adaptation mode for review; this is a floor,
not a cap.

The seed command requires the explicit source-review acknowledgement already required by the ETL workflow. It uses the local Supabase service-role key and never sends that key to the browser.

Useful commands:

```bash
npm run stage:status
npm run stage:stop
```

The workflow uses Supabase's local CLI stack and its standard local ports. See the [Supabase local development guide](https://supabase.com/docs/guides/local-development/cli/getting-started) for Docker and CLI troubleshooting.
