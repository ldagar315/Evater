# Curiosity coverage: eleven new chapter guides

Prepared 15 November 2026. Reviewable content, not an instruction to merge. Owner review and merge remain required. Article status is published in the proposed code so owner merge releases the approved articles through the existing blog system. Nothing has been deployed by this change alone.

## Coverage

`inventory.json` lists the 13 current NCERT Curiosity chapters. Existing Chapter 5 (Forces) and Chapter 10 (Light) articles are preserved. The integrated batch adds Chapters 1-4, 6-9 and 11-13, completing one article per chapter alongside the two existing articles. Owner review/merge remains pending. Earlier checkpoints below describe earlier branch states, not final delivery proof.

## Sources and editorial checks

The full text of each of these official chapters was read before preparing its guide:

- Chapter 1: https://ncert.nic.in/textbook/pdf/hecu101.pdf
- Chapter 2: https://ncert.nic.in/textbook/pdf/hecu102.pdf
- Chapter 3: https://www.ncert.nic.in/textbook/pdf/hecu103.pdf

The current 2026-27 reprint identifies the chapter names. The chapter-link inventory was cross-checked against https://padhaiwise.com/resources/class-8/books/science/ and its official NCERT links, rather than constructed from a guessed chapter count.

| Guide | Focus and original learning aid | Accuracy and safety check |
| --- | --- | --- |
| Investigative world | Focused question, change/measure/keep, repeated observations; original spinner data and four answer checks | Data explicitly made up; limited conclusion; safe low-height teacher-approved comparison; no unsupervised hot-oil activity |
| Invisible living world | Cells to organisms, structure/function, microbes, fermentation, microalgae and decomposition; five answer checks | Microorganism is not a synonym for unicellular; viruses separated from cells; plant wall not claimed unique; no unknown microbial culture/tasting; held Chapter 2 draft rechecked and substantially expanded |
| Health | Physical/mental/social health, signs/symptoms, transmission, immunity and antibiotic resistance; five answer checks | Pathogen and vector distinct; no disease blame; resistance belongs to bacteria; not a diagnosis, prescription or vaccine schedule |

Questions, comparisons and diagrams are original learning aids, not official exercise answers. Dubious or oversimplified textbook sidelines were not copied into the guides. Covers and diagrams are local original SVG assets, with alternative text. No external stock-image dependency is added.

All three links go to the existing generic practice entry. The copy explicitly says a dedicated set is not verified. No blogPracticeTargets mapping, grading behavior, backend data, credit balance or account setting is changed.

## Validation

- `npm test`: 43 tests across 12 files pass, including three new guide-integrity/safety tests.
- `PRERENDER=false npm run build`: pass.
- `npx tsc --noEmit -p tsconfig.app.json`: fails on existing baseline errors; no new science-guide file errors. A Vite build is not a clean typecheck.
- `node qa-science-guides.mjs`: nine actual-app cases across 320, 390 and 1280px; no horizontal overflow, local diagrams load, page errors absent, canonical and Article/Breadcrumb data present, table-of-contents target and hash reload work.
- Complete article pixels inspected at all three widths after fixing busy covers and small diagram labels.
- Mobile category/search -> Chapter 2 -> practice link checked. Signed-out practice lands at auth with `next=/practice`; dedicated authenticated chapter inventory remains unverified.
- Source URLs were fetched and read; no claim that a browser click to an official PDF was separately exercised.
- Full prerender: default build cannot find its pinned Chrome. Retrying with system Chrome reached the prerender stage but did not finish within a bounded 65 seconds (Unable to prerender all routes). Exact hosting build verification remains required; no deployment setting was changed.
- Deployed exact-preview and private review-page checks are still pending. Do not call the branch live or indexed.

Reproduce browser QA after building, from frontend. Backend requests are intercepted with empty fixture responses; this is not an owner-session or practice-attempt test. Screenshots and results are written under `/downloads/evater-science-guides-evidence` by default.

## Second batch: Chapters 4, 6, 7 (local preparation checkpoint)

Full official chapter PDFs were read on 15 November 2026: https://ncert.nic.in/textbook/pdf/hecu104.pdf , https://ncert.nic.in/textbook/pdf/hecu106.pdf , https://ncert.nic.in/textbook/pdf/hecu107.pdf . Original guides, diagnostics and local SVGs are prepared. Source coverage includes electromagnets/heating/cells; force-area/liquid/air pressure/weather; particle spacing/attraction/motion/states. No copied official exercise answers, invented dedicated practice sets, or backend/account/credit changes.

Ch6 safety discrepancy: its roof paragraph recommends open windows and its lightning paragraph suggests an outdoor crouch. These are not repeated as safe actions. Specialist sources read in full: Andhra Pradesh State Disaster Management Authority https://apsdma.ap.gov.in/cyclones.php (doors/windows shut, evacuation and official all-clear); US National Weather Service https://www.weather.gov/safety/lightning-safety-overview and https://www.weather.gov/safety/lightning-crouch (substantial shelter; crouching does not provide safety). Curriculum pressure explanation is separate from real emergency instructions. No US emergency number was transplanted into the India-facing article.

Ch4: no touching heated wires, mains circuit, short-circuit, battery dismantling or single-use cell charging instructions. Same-current/circuit conditions are not silently swapped in heating comparisons. Ch7: no taste test, smoke/iodine/permanganate instructions; particle size and arrangement distinguished; ice spacing exception named; solid grain vs flowing pile distinguished.

Nine isolated local article browser cases at320/390/1280 pass; all full article pixels inspected. Three new integrity/safety tests pass; isolated batch43tests pass. Combined six-article source and final remote/hosting proof must be rerun before review delivery.


## Third checkpoint: Chapters 8 and 9 prepared locally

Full official chapters read: https://ncert.nic.in/textbook/pdf/hecu108.pdf and https://ncert.nic.in/textbook/pdf/hecu109.pdf . New original explanations, classification/measurement diagrams, and answer checks cover pure substances versus uniform mixtures, elements versus compounds, physical separation, saturation versus concentration, temperature trends, mass versus weight, density, and displacement. No chemical heating, gas preparation, tasting, or home medical recipe is proposed. Mineral classification is not confused with chemical composition; solvent amount is not turned into an absolute mass rule. Density examples and ratios were independently calculated.

Combined eight-article source: 46 tests in 13 files pass; Vite build passes; 15 further-guide browser cases (five guides at 320/390/1280) pass with no page errors/overflow and loaded diagrams, TOC/hash reload and metadata checks. Full new article pixels inspected at all three widths. Existing first-three guides preserved. Expanded remote, hosting and private full-word pack proof remain pending. Chapters 11-13 still require full source review and preparation.


## Integrated eleven-article checkpoint

All eleven missing chapters now have original source-reviewed guides, original learning diagrams and diagnostic answer checks. Final official PDFs read: https://ncert.nic.in/textbook/pdf/hecu111.pdf , https://ncert.nic.in/textbook/pdf/hecu112.pdf , https://ncert.nic.in/textbook/pdf/hecu113.pdf . Ch11 separates phases/eclipses/rotation/calendar types and uses safe Moon-only observation (NASA https://science.nasa.gov/eclipses/safety/ and https://science.nasa.gov/moon/viewing-tips/). Ch12 uses food-to-eater arrows and conditional population reasoning, distinguishes energy flow/nutrient cycling, and avoids hazardous habitat exploration. Ch13 does not repeat textbook conflations of pollen/gametes or ovule/egg: supporting NCERT sources https://ncert.nic.in/textbook/pdf/lebo101.pdf and https://ncert.nic.in/textbook/pdf/kebo105.pdf . Paris temperature wording verified against https://unfccc.int/files/press/press_releases_advisories/application/pdf/pr20151112_cop21_final.pdf . No current mission/discovery/climate-progress claims are inferred.

Integrated browser run split into bounded widths: 8 cases at390 and16 at320/1280 all pass (24 total for further guides). All final-three article body/footer/diagram pixels inspected at390/320 and1280, as were earlier guides. New cover differentiation and editorial guards are being finalized; final remote exact-byte verification, host build and private full-word review pack still pending. This checkpoint does not claim production publication or dedicated practice-set inventory.
