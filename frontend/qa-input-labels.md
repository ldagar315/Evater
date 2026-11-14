# Shared Input label association

The visible labels on the shared Input component were not associated with their controls. The patch uses React useId for a stable generated ID and label htmlFor, retaining caller-provided IDs and all existing input props. No auth/session, validation, visual layout or database behavior is changed.

From frontend with dependencies and Chrome installed:

```sh
node qa-input-labels.mjs --baseline --evidence-dir /tmp/input-label-proof
node qa-input-labels.mjs --evidence-dir /tmp/input-label-proof
PRERENDER=false npm run build
npm test -- --pool=threads --poolOptions.threads.singleThread
npx tsc -p tsconfig.app.json --noEmit
```

Set CHROME_PATH if Chrome is not at /usr/bin/google-chrome. The original component baseline comes from main commit f7fe706. Fetch that commit if absent from a shallow checkout.

The loopback Chrome harness exercises the actual shared Input, AuthForm and ProfilePage at320/390/1280. Auth/profile hooks return read-only synthetic state; Header is omitted. Auth and profile writes throw and all external requests are blocked. No credentials, real account, email, profile change or real authenticated journey are involved. CSS is generated from the app Tailwind config, with network font import removed for offline checks.

November12 results:
-9baselinecontrols reproduce missing label associations for shared input/signup/full name at3widths.
-Patched33checks:32PASS,1FAIL. Label focus/name/ID/prop/keyboard/rerender checks pass. The remaining FAIL is actual ProfilePage horizontal overflow at320:359px content. Baseline has exactly the same359px overflow. It is a separate pre-existing layout bug, not fixed or hidden by this label patch.
-0externalrequests,0browsererrors.
-16existingunit testsPASS. VitebundlePASS with prerender skipped.
-Full app TypeScript remainsFAIL, baseline and patch diagnostics exactly identical. No app-clean claim.
-Actual signup and profile pixels inspected at320/390;320profile overflow confirmed. No claim that the full profile UI is ready at320 or that all profile labels (e.g. Select) are repaired by this Input-only patch.

This is a review patch only. No production deployment or merge.
