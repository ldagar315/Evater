# Signup and chapter-loading QA, 9 November 2026

## Verified locally, not deployed

Commands from frontend/: `npx tsc --noEmit`, `npm test`, `npm run build`, `node qa-signup.mjs`.

13 unit tests. Browser harness runs the production build and intercepts every external request: no production write, no real credentials, no paid generation. System Chrome path is configurable with CHROME_PATH.

Browser checks cover existing custom school preservation; choosing a canonical campus with keyboard; unselected arbitrary text cannot enable Save; mobile390px touch selection/no horizontal overflow; new-profile Other; unchanged100-credit insert default; existing PATCH/new POST; success navigating to /home only after persisted-row response; errors or null saved row staying on profile; signed-in /auth redirect; chapter loading, failure, retry, subject switching and return-to-subject cache hit. This does not validate production DB write permissions or actual Google credential/consent/callback flow.

Visual inspection: desktop profile, mobile suggestions with distinct campus addresses, chapter failure/retry. Mock learner data is visibly used. No claim of screen-reader testing; semantic combobox/listbox, arrow keys, Enter and Escape are provided.

## Live read-only measurements

Production https://evater.xyz/practice with an existing signed-in session. Bounded QA samples, not a statistically representative benchmark:

| Navigation | Initial Users read | Practice JS load | Chapter request starts after navigation | Chapter request duration | Chapter response end |
| --- | --- | --- | --- | --- | --- |
| First observed |501ms|85ms|1240ms|1796ms|3036ms|
| Later reload |790ms|219ms|2229ms|1077ms|3306ms|

Two additional authenticated same-endpoint reads returned200/13 chapters in966ms and1677ms. These are warm/repeat observations, not proven cold starts. Browser resource timing is cross-origin and does not expose server TTFB here; the HTTP duration includes network and server time, so it cannot isolate backend processing.

Code confirms ProtectedRoute waits for a Users profile read before mounting the lazy practice page; that page then requests chapters in an effect. The endpoint performs curriculum-version then chapter reads serially. The route/header separately read the profile again. Thus the observed delay includes frontend auth/profile gating and a sizable chapter HTTP wait. No evidence supports attributing it solely to a cold Modal container or solely to React rendering.

This patch makes loading and retry visible, deduplicates concurrent chapter reads and caches successful chapter catalog responses per authenticated account and subject for five minutes, in memory only. Subject changes clear the previous list while loading; failed responses are never cached. It does not speed up an uncached first request, change database behavior, alter backend capacity, or establish a measured production improvement. Cache freshness tradeoff: catalog updates can take up to five minutes to appear in an existing session.

## Google400 status: unresolved

Anonymous production /auth -> Google click reached Google's normal email sign-in, with no400 at OAuth launch. Read-only Supabase provider check: Google enabled; Site URL evater.xyz; redirect allowlist covers evater.xyz/**, www.evater.xyz/**, localhost:5173/**. No changes saved. These checks do not prove Google credential, consent or callback stages work. Need the actual400 page/error text and originating URL (without tokens) to choose the next check. Signed-in /auth previously404s in code and now redirects /home; this is a separate bug, not a claimed fix for Google400.

Production configuration, credentials and DB were not changed. User merges the PR.

Mobile follow-up: touch-swiped a20-option scroll list to its last Other option, selected it, edited the custom school, scrolled to the enabled Save button, tapped it and reached /home after the mocked response. Added CSS overscroll containment to avoid scrolling the underlying page at list boundaries. The native software keyboard is not simulated by desktop Chrome device emulation.
