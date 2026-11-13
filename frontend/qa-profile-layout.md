# Profile mobile layout regression

From frontend, with existing project dependencies and Chrome installed:

```sh
node qa-profile-layout.mjs --baseline --evidence-dir /tmp/profile-layout-proof
node qa-profile-layout.mjs --evidence-dir /tmp/profile-layout-proof
PRERENDER=false npm run build
npm test -- --pool=threads --poolOptions.threads.singleThread
npx tsc -p tsconfig.app.json --noEmit
```

CHROME_PATH can override /usr/bin/google-chrome. No new dependency or production route added.

Uses the actual ProfilePage and shared UI with synthetic auth/profile hooks. Header is omitted. A loopback server serves CSS/assets; all external or non-GET requests are blocked. Save outcomes are mock functions only. No account, email, provider, API or database write.

Three fixtures (new, existing and long unbroken name/school) at320,390,1280px. The baseline loads ProfilePage from mainf7fe706 and records actual geometry, not a blanket expected failure. Baseline scroll widths are359/406/1006 at320,390/406/1006 at390, and1280 for all desktop fixtures. The patch yields exact viewport width in all nine cases without overflow:hidden. Screenshots preserve the card order and desktop columns; existing truncate styling is retained.

30 patched checks cover geometry/content/card bounds, responsive columns, disabled incomplete form, mock save success/error/missing/throw with exact payload and single-call assertions, and resetting edits without a request. This is not proof of real auth, school search, profile persistence or the full student journey.

Local result:30PASS,0external requests/browser errors.16 existing unit tests PASS. Vite bundle PASS with prerender disabled. Full app TypeScript FAIL on both original and patched ProfilePage with identical diagnostics, not a clean typecheck claim. No auth, form, data model or submit logic changed. Separate from the Input label and auth-completion drafts; no merge or production mutation.
