# Session-aware auth completion regression

## Bug and patch scope

AuthForm previously called onSuccess after any successful signup API response, including a null session while email confirmation is required. AuthPage then navigates to /profile even though that route is only registered for signed-in users. It also called onSuccess immediately after OAuth initiation, before a callback session existed.

The patch does not change authentication provider settings, redirects, route guards or database state. It only calls the existing completion callback for an email response with a session user ID, shows neutral confirmation instructions after signup without a session, and leaves OAuth completion to the existing auth-state listener. A sign-in response without a session is an explicit error. No account existence or email delivery is asserted.

The confirmation state clears both password values, hides the signup form (so it cannot be resubmitted accidentally), and offers a switch back to sign in while retaining the email address. It sends no additional request.

## Run

From frontend after npm ci, with Chrome available:

```sh
CHROME_PATH=/usr/bin/google-chrome node qa-auth-session.mjs --evidence-dir /tmp/auth-session-proof
CHROME_PATH=/usr/bin/google-chrome node qa-auth-session.mjs --baseline --evidence-dir /tmp/auth-session-proof
PRERENDER=false npm run build
npm test -- --pool=threads --poolOptions.threads.singleThread
npx tsc -p tsconfig.app.json --noEmit
```

The baseline control reads the original AuthForm from commit f7fe706 (the patch's main base). Fetch that commit if using a shallow checkout without it. Baseline PASS means the old false-completion bug was reproduced, not that baseline is correct. The patched run exercises the actual AuthForm and shared UI components with mocked auth responses in Chrome at 320,390,1280 pixels. It uses the app Tailwind configuration, removing only the external font import so all requests stay on loopback. No actual signup, email, OAuth request, profile save or test submission occurs. The fixture has no real credentials. esbuild and Puppeteer are existing build dependencies.

## Evidence from November12,2026

- Baseline: both premature callbacks reproduced at all3widths,6controls.
- Patched:48checks PASS (16cases x3widths),0externalrequests,0browsererrors.
- Cases: confirmation without session; Google initiation without completion; confirmation-to-signin with no extra request and cleared passwords; signin/signup with session once; signin with no session; both email service errors and throws; password mismatch without request; both OAuth errors and throws; pending request no completion/no duplicate; existing invalid-credentials hint; no horizontal overflow.
- Existing Vitest tests:16PASS in4files.
- Vite production bundle:PASS, prerender explicitly skipped. This is not a real hosting or authenticated callback test.
- Full app TypeScript:FAIL on pre-existing diagnostics. No claim of full typecheck success. The baseline/patch diagnostic comparison is included in the review evidence, with no new diagnostics in AuthForm.
- Initial fixture attempts failed because the local Chrome path was unset, then a blocked network-font import triggered the network guard. Pixel inspection caught a test-CSS stripping issue; fixed the fixture and repeated both entire runs. Those failures were fixture setup, not hidden application failures.

## Unresolved, not fixed by this PR

The real Google400/callback issue, current email-confirmation setting, actual email receipt, actual signup-to-profile save and full authenticated learning journey remain unverified. This is a component-level reliability fix, not proof of a working live signup journey. No production deployment or merge is authorized by these checks.
