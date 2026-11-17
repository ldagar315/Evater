# Original Class 8 reasoning expansion, 17 November 2026

24 original MCQ candidates: 15 Maths, 9 Science, across 8 chapters and 24 repository-mapped topics. These are review material, not a migration, an automatic import, or a claim of 24 new live questions.

## Why this batch

The sample-paper review found a format gap: Evater's public collection has 27 chapter papers, each with 10 objective and 5 written responses. Several search-result pages offer mixed-chapter term papers with longer reasoning and case-based sections. This first bank batch adds interpretation of measurements, error analysis, limits of an inference, unit consistency and multi-step applications. It does not establish that those topics are missing from the current live bank. Live per-topic inventory and final import mapping must be checked before any authorized production import.

- Maths chapters: A Square and A Cube; Power Play; Quadrilaterals; We Distribute, Yet Things Multiply; Proportional Reasoning-1.
- Science chapters: Health; Electricity; Particulate Nature of Matter.
- Repository catalog chapter/concept UUIDs are resolved using existing catalog code and topic slugs, not invented identifiers. They are not certified against the current production database.
- Every candidate is `review`, source tier `manual`, with original-authorship provenance and a concept-background NCERT URL. The existing publishing/import gates are unchanged.
- Difficulty and timing are editorial estimates, not student-calibrated or teacher-certified.

## Originality and correctness

Questions were independently authored from concept-level goals. Competitor wording, numbers, diagrams and answer banks were not copied or lightly paraphrased. Public sources informed the assessment formats, not a rewrite template. Exact stems were checked against the existing 405 static-paper questions and the earlier 80-question reviewed pack. This is a bounded local check, not a global plagiarism guarantee.

All 24 pass the existing QuestionCandidate model and repository chapter/topic mapping checks. Thirteen arithmetic annotations, additional identity and geometry checks, distinct options, distinct stems, review-only status and answer-key distribution have tests. Science explanations retain uncertainty, safe low-voltage framing and no personal medical advice. The tests do not prove pedagogical calibration or production schema parity.

Run: `python -m pytest -q tests/test_reasoning_expansion_pack.py`

## Next release steps

1. Review the content in `questions.json` and compare with current live per-topic inventory.
2. Validate live curriculum/concept identifiers and current schema under separately established database authority.
3. Prepare a bounded import/rehearsal plan only when that authority and access exist. Do not treat a repository merge as a production import.
4. Keep full-subject term-paper development separate from chapter PDFs. Class 8 school exam patterns vary; never label an Evater paper as an official CBSE paper.

The research audit is in `research-audit.md`. No copied third-party PDF is committed to this repository.
