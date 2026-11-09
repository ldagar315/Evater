# My tests discovery, 9 November 2026

User request: make the previous-test flow easy to find. The existing header, home and footer had no link to `/previous-tests`. That old route reads generated papers only, while chapter-practice attempts are stored separately. Its empty state must not be used to claim a student has never taken a practice test.

## Change

- New `/my-tests` entry in signed-in desktop/mobile learning navigation, home CTA and footer.
- Hub links to existing Test papers and Paper feedback pages, without changing those flows or enabling paid features.
- Practice history reads the signed-in user's `test_attempts` with `user_id` filter and existing RLS. It requests only summary columns, chapter title and subject. No answer keys, answer sheets or other users' records.
- Twenty entries per page, newest first, with Show older practice tests. Failed reads have Retry and do not show a false empty state. Completed records show stored marks; unfinished/abandoned attempts have status only. Review tests allow a null chapter and show Across chapters.
- No resume or question-by-question result replay is claimed. That capability would need a separate owner-scoped read/replay implementation.

## Verification

Live read-only check on9 November: authenticated owner-filtered `test_attempts` select with chapter/subject joins returns200 and saved completed/in-progress/review summaries. Existing generated-paper history also loads. No DB, schema, settings or paid-generation changes.

`cd frontend && npx tsc --noEmit`, `npm test`, `npm run build` and `node qa-history.mjs` pass. Browser harness intercepts all external requests, verifying home CTA and desktop/mobile menu discovery, completed score/unfinished status/review title, paging, owned GET-only queries, existing paper link, error retry and empty state. Inspected actual1440px desktop and390px mobile screenshots; no horizontal overflow. Native software keyboard and screen-reader behavior are not tested.

This branch is based on main independently of signup PR82. Neither PR was merged by the assistant. Production verification follows only after the user's merge.
