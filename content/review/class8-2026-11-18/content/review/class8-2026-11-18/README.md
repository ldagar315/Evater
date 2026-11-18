# Class 8 original MCQ review pack - 18 November 2026

54 review-only candidates: two per chapter across 14 Mathematics and 13 Science chapters. Each has four options, one key, an explanation, difficulty, chapter/topic identifiers and a source-fit note. These questions were independently authored, not copied from another question bank.

## Not a release

`publication_approved` is false and every `review_state` is `review`. This directory is outside the app, seed and migration paths. It provides no database import or automatic publication. Merging these review artifacts alone does not add questions to the live bank. Review does not imply approval to publish.

Readable chapter-by-chapter review: https://files.instinct.com/06tf012ag3rz-54-new-class-8-practice-questions

Accepted candidate SHA256: `84a088f64b8032ebf75065155ba22599734f69cb33030fca3b38a88f3756a173` (`questions.json`, final revision 7).

## Checks

Run `python3 validate.py` from this directory. It checks the exact accepted bytes, 54 questions, chapter coverage, review-only flags, unique IDs and exact stems, four distinct options, valid keys and all 26 arithmetic assertions using a restricted arithmetic parser, not eval. Actual answer positions: A11, B17, C14, D12. They are varied, not perfectly balanced.

The arithmetic assertions are mechanical spot checks, not proof of every question's meaning. Independent subject review checked keys, ambiguity, chapter fit and similarity to existing practice/review questions. A semantic challenge identified 11 Maths candidates that were numerical variants of earlier tasks; those were replaced and rechecked. Exact-stem uniqueness alone would not have found them. This is a bounded comparison against the reviewed Evater corpus, not a global originality guarantee.

## Material review decisions

- The Roman-numeral item asks recognition of an invalid numeral. The older written paper already covers 99 and subtractive-pair rules, so this is a different question form/reasoning task, not wholly new rule coverage.
- The linear-pair item explicitly identifies adjacent angles on a straight line.
- The area item applies primary-text unit-fraction area reasoning; it is not described as an exact textbook example.
- The electromagnet comparison holds coil turns, cell and connections fixed while comparing iron and wood cores.
- The mixture explanation correctly says the dissolving-first distractor fails because filtration is omitted. It does not claim wet iron cannot be separated magnetically.
- The antibiotics question is framed around a doctor's prescribed course, not self-treatment advice.

Primary NCERT text and actual existing PDFs were used where available. Number Play's full primary chapter text could not be retrieved during the final integration check: a live publisher chapter summary supports the selected divisibility/remainder/digits scope, while arithmetic was independently checked. Do not present this as a fresh full-primary-text audit of every chapter. Source-fit notes are review metadata, not student teaching text and not a claim of exhaustive syllabus coverage or an official CBSE pattern.
