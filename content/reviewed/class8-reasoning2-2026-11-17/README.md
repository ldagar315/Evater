# Class 8 reasoning expansion, second review pack

24 original single-answer MCQ candidates: 12 Mathematics and 12 Science, across 15 chapter categories and 24 mapped topics. Those chapter categories were not represented in the preceding 24-question reasoning pack. This is a comparison of repository review packs, NOT evidence that the live bank lacks these chapters or topics.

Adds place-value reasoning, reverse percentages, percentage bases, Pythagorean applications, constraints on cube nets, interpreting time-series readings, pyramid equations, cell evidence, friction, electrostatic inference, pressure comparison, limits of concentration data, mirror selection, Moon-observation design, nutrient recycling and habitability evidence. Answers and explanations included.

Review only. No production import, publishing approval or database access. Mapping uses current repository chapter and concept UUID derivation, not the live database. Difficulty and timing are editorial estimates, not student calibrated.

## Validation
Run `PYTHONPATH=. python -m pytest -q tests/test_reasoning_expansion_pack2.py`. Four targeted tests cover candidate schema, review status, repository mapping, 24 distinct stems/topics, balanced A6/B6/C6/D6 keys, nine literal arithmetic annotations, pyramid invariance and exact-stem dedup against the existing 405 static-paper questions, earlier 80-question review pack, preceding 24-question reasoning pack and mixed papers when present. Exact dedup is not a semantic uniqueness guarantee. Familiar concepts remain familiar; the goal is useful practice, not a claim that each question teaches a new concept.

## Concept sources and fit
Each question records the relevant NCERT chapter URL as concept background. No question is represented as an NCERT exercise, and no competitor wording or diagrams are copied. Read current chapter content rather than treating catalog labels as syllabus certification. In particular, the current Forces chapter explicitly postpones formal balanced-forces study to higher grades, so a candidate on zero resultant and constant velocity was discarded before this pack. The final question uses friction instead. A generic shadow-observation draft was replaced with Moon-cycle observation, matching the actual chapter.

NCERT geometry/data origins were temporarily unavailable. These actual textbook mirrors were read for cube-net constraints and time-series line graphs:
- https://pdf.ncertarchive.com/ncert-books/new/2026/class-8/mathematics/class-8-maths-chapter-11.pdf
- https://pdf.ncertarchive.com/ncert-books/new/2026/class-8/mathematics/class-8-maths-chapter-12.pdf

Maths background: hegp103 (number systems), hegp201 (percentages), hegp202 (Pythagoras), hegp204 (geometry), hegp205 (data), hegp206 (algebra). Science background: hecu102, hecu105, hecu106, hecu108, hecu109, hecu110, hecu111, hecu112, hecu113. Complete source URLs are in each candidate.

No ranking guarantee, live-bank gap claim, full-backend-suite claim or teacher/student calibration claim.
