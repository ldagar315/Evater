# Curiosity coverage: first three chapter guides

Prepared 15 November 2026. Reviewable content, not an instruction to merge. Owner review and merge remain required. Article status is published in the proposed code so owner merge releases the approved articles through the existing blog system. Nothing has been deployed by this change alone.

## Coverage

`inventory.json` lists the 13 current NCERT Curiosity chapters. Existing Chapter 5 (Forces) and Chapter 10 (Light) articles are preserved. This batch adds Chapters 1-3, leaving eight chapters requiring full source review and article preparation. Held drafts are leads, not approved coverage.

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
