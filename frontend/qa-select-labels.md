# Select label regression

Run from frontend with existing dependencies/Chrome:

```sh
node qa-select-labels.mjs --baseline --evidence-dir /tmp/select-proof
node qa-select-labels.mjs --evidence-dir /tmp/select-proof
PRERENDER=false npm run build
npm test -- --pool=threads --poolOptions.threads.singleThread
npx tsc -p tsconfig.app.json --noEmit
```

CHROME_PATH can override /usr/bin/google-chrome. Baseline Select is read from mainf7fe706. Six baseline controls reproduce missing association/accessibility name in shared controls and actual ProfilePage at320/390/1280px.33patched checks pass: generated/explicit ID association, accessible names, label click focus, disabled behavior, props/options/controlled onChange, stable rerender IDs, dynamic remount, keyboard selection and Tab order, actual profile Class dropdown and CreateTestPage existing no-label/disabled/options props. No new dependency.

Actual components/shared UI are bundled with synthetic auth/profile/AppState/database fixtures, Header omitted. No real auth/profile/generation or API requests. Loopback assets only. CreateTestPage's pre-existing no-label chapter dropdown is not repaired by associating labels supplied to Select: this regression verifies its existing props, not a claim that every dropdown is now named. Profile's existing320px horizontal overflow is likewise recorded, not fixed; separatePR94 fixes that layout. This patch is independent of PR93Input labels and PR94grid layout.

33PASS,0external requests/browser errors;16existingunitPASS,VitebundlePASS with prerender disabled. Full app TypeScript FAIL with identical baseline/patch diagnostics, not clean. No submit/data/route/auth logic change or production mutation.
