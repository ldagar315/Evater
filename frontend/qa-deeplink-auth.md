# Signed-out app deep-link regression

Scope: known app routes remain registered and use the existing ProtectedRoute. The profile editor has its own user check so a missing-profile user can still fill it without a redirect loop. Unknown paths keep NotFound. No auth bypass, profile guard removal, new backend route, submit, account or data logic change. Does not preserve a return URL after sign-in, which the existing guard did not do either.

From frontend, npm ci with Chromium available, then:

```sh
PRERENDER=false npm run build
node qa-deeplink-auth.mjs /tmp/deeplink-proof
npm test
./node_modules/.bin/tsc --noEmit -p tsconfig.app.json
```

81 patched checks across320/390/1280 use actual production App/router/AuthProvider/useProfile/Header, no substituted hooks. All18 known signed-out app paths including optional/dynamic examples reach /auth, with no service GETs in signed-out mode, no protected content or horizontal auth overflow. Four unknown routes retain their existing not-found UI. Redirect uses replace and browserBack goes to previous /about/ without an auth/404 loop. Synthetic authenticated missing-profile state redirects to the editor; existing profile opens and reloads My tests; the editor has no loop; unknown authenticated URL stays404. Browser-local synthetic session and intercepted auth/profile/history GETs only, unknown service routes/writes abort. No real sign-in, account, profile saving, generation, test submission, scoring/credits or full real student journey proved.

For baseline: build exact main f891a5a App.tsx, run --baseline.78 expected outcomes PASS, including18 signed-out paths at each width showing NotFound. Those54 PASS observe the defect, not a working redirect. Baseline omits the new replace/back assertion. Build the patched file again before the normal command.

QA_ORIGIN=https://verified-host-origin runs the same checks against an exact-head hosted bundle. Host-only /blog/not-real may be a static404 rather than the SPA's '404 article' UI; the script must account for that without accepting a home-page200. Local server not-found UI assertions alone do not prove HTTP404. Hosting rules are unchanged; actual unknown-blog HTTP404 is independently covered by PR90, which is not stacked into this PR.

Local Nov14: baseline78 expected outcomesPASS,patched81PASS,zero browser errors/unexpected service routes,16 unit testsPASS,Vite bundlePASS with prerender disabled. Full app TypeScript fails on both baseline and patch with identical diagnostics after source-position normalization. No claim of clean typecheck. Existing profile layout/labels/Home issues in separate PR93-97 are not included. Actual320signed-outMytests→signin,390missingprofileeditor,1280existingMytests pixels inspected.

Initial test expected lowercase visible blog404 text while CSS uppercased innerText; switched to textContent for the existing '404 article' node. No production code was changed to satisfy a mistaken test expectation. A long bundled bash run timed out during baseline typecheck; restored patchedsource and reran baseline/patch suites separately. Final results are complete.
