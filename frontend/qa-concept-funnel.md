# Article-to-concept practice QA

Base: main ec41294c7d7e42aa6450333f73d894a7c39b59a2 (includes PR99).

Run `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome npm run build`, then `QA_WIDTH=320 node qa-concept-funnel.mjs /path/to/evidence`. Repeat at 390 and 1280. `QA_ORIGIN` switches to a hosted preview, but all auth, profile, target and test service calls remain intercepted synthetic fixtures. These are UI/integration-contract checks, not real OAuth, email delivery, saved profiles, bank availability or student test completion proof.

Final local result: 39 checks passed at 320/390/1280, zero browser errors or unexpected service routes. The first harness run caught an AuthPage completion race: an existing learner was sent to the profile editor rather than the requested concept. Completion now enters the requested protected route and lets the normal profile guard decide whether a profile is required. The updated run passed. Another harness assertion referenced a Node variable inside the browser; corrected by passing its argument. No app change was made for that harness error.

Coverage: exact article target, signed-out return path, email sign-in, profile-save error/success, setup reload, short seven-question inventory, explicit start payload, zero inventory, invalid IDs, absent chapter, missing/new endpoint, inventory change between check/start, explicit topic change, Google provider initiation URL, and email-confirmation return URL.

The new `/concept-tests` route is intentionally distinct from `/tests`: older servers ignore unknown JSON fields on the latter. New concept requests also require a single complete block, so older submit servers cannot select a later block from the whole chapter. An old backend returns an error; the frontend must never retry through the generic chapter endpoint. Deploy the backend endpoint before expecting the new CTA to work. No migration, production data import or setting change is included.

Backend selected suite: 35 passed, including 11 new concept tests and existing question selection/review/submission recovery tests. Tests use fake data clients; no production writes. They cover published status, exact chapter/concept intersection, empty/insufficient inventory, response-model answer-key omission, auth, different-user submission rejection and single-block mixed-version safety.

Frontend unit suite: 22 passed. Full app typecheck is NOT clean: exact-main baseline has 57 diagnostics, proposed has 56, with no new normalized diagnostic and one removed unused React import. Command: `npx tsc --noEmit -p tsconfig.app.json`. Both exit nonzero. Build/prerender passed separately.

No new article prose, student content, production settings, DB data, merge or spending is included. Four existing chapter articles receive static target metadata, with live availability checked only after sign-in. Original content drafts and whole-Class-8 coverage are separate work.
