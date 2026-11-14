# Loaded Home layout regression

From frontend using existing dependencies and Chrome:

```sh
node qa-home-layout.mjs --baseline --evidence-dir /tmp/home-proof
node qa-home-layout.mjs --evidence-dir /tmp/home-proof
PRERENDER=false npm run build
npm test -- --pool=threads --poolOptions.threads.singleThread
npx tsc -p tsconfig.app.json --noEmit
```

CHROME_PATH overrides /usr/bin/google-chrome. Actual HomePage/shared UI, synthetic delayed120ms profile hook, Header omitted, loopbackassets only. Await loaded greeting before measuring existing/long fixtures. New/no-profile, normal name/school, long unbroken name/school at320/390/1280. Baseline loads HomePage from mainf7fe706; longname overflow1215/1215/2030px. Patched all9layouts match viewport320/390/1280.12checksPASS include card bounds, responsive order/desktop columns, no root/body overflow hiding, practice button route/no saved calls. Full long name wraps rather than being clipped; existing school-stat truncation retained and constrained to its card.

16existingunitPASS,VitebundlePASSprerenderoff,fullappTSFAILidenticalmain baseline. Inspected actual320long/390normal/1280long pixels.

Separately, local combined exactPR92-95 production App/router/AuthProvider/useProfile run plus this Home patch:18expected outcomesPASS after waiting Home's own profile GET, including signup confirmation/back, synthetic signin→missingprofile, mocked save error retention/success→Home, practice setup/start/nextscroll and cancelled unansweredFinish/no submit, Back, explicit remaining unnamedchapterselector observation. ThreePASS record that existing page-level defect, not a fix. Every external Supabase/Modal request intercepted locally; no real account/auth/email/profile/generation/test write or scoring/credits/fullstudentjourney proof. Combined source hashes, before15PASS3FAIL and after18PASS retained in separate evidence package. No claim this scoped PR includes the other drafts, and no merges/production mutations.
