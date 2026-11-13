# Chapter dropdown accessible-name regression

Scope: one existing CreateTestPage heading is connected to its Select with React useId and aria-labelledby. No visual label duplication, hard-coded ID collisions, changed options, submit, auth, data or credit behavior. Compatible with current main's shared Select and separate draft PR95.

From frontend, install locked dependencies with npm ci (Puppeteer needs Chromium), then:

```sh
PRERENDER=false npm run build
node qa-chapter-name.mjs /tmp/chapter-proof
npm test
./node_modules/.bin/tsc --noEmit -p tsconfig.app.json
```

Actual App/router/AuthProvider/useProfile/Header/Select production bundle, not substituted hooks or auth bypass. Browser-local synthetic user/session/profile. All Supabase routes are intercepted; only synthetic GET fixtures are allowed. All generation/test writes and other service routes fail closed. No real authentication, email, profile saving, generation, test submission, scoring, credits or full real student journey is proven.

15 checks at320/390/1280: disabled control's name/placeholder/heading association, grade then subject enabling/filtering, native keyboard focus and selection/onChange, subject reset/filter, grade reset/disable/reselect. Accessibility-tree name is exactly 'Select Topic/Chapter'. Root geometry and full-page screenshots at four states per width. No visual layout change expected; inspect screenshot pixels.

For negative baseline, build main f7fe706's CreateTestPage, run same script with --baseline. That mode expects the empty name on disabled and enabled control, retaining all interaction checks. Its PASS means the known defect was observed, not fixed. The archived local baseline captures3 disabled+6 enabled nameless observations across3widths.

For a verified exact-head hosted preview, QA_ORIGIN=https://verified-preview-origin node qa-chapter-name.mjs /tmp/chapter-hosted. Service interception is identical, with hosted fonts allowed. Verify the deploy's commit first. Artifacts from an earlier build are not exact-head proof.

Local results Nov13: baseline15 expected outcomesPASS including nameless defect; patched15PASS;16 existing unit testsPASS;Vite production bundlePASS with prerender off. Full app TypeScript fails identically on baseline and patch; empty diff is not a clean typecheck. No merge or live deployment authorized.
