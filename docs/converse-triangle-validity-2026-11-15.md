# Converse questions must describe possible triangles

Read-only production chapter-9 inspection on 15 November 2026 found four
published questions whose stems call impossible side lengths a triangle:

| Question ID | Sides (cm) | Existing test references |
| --- | --- | --- |
| 78fe6925-12f7-431d-bbe3-cc2fa40f13fa | 2, 2, 15 | 3 |
| ebaa5074-8fd4-435d-844e-a234545a3788 | 2, 20, 25 | 3 |
| 7b23cd6d-0ad3-4be8-85e8-bfa95409aca1 | 5, 11, 18 | 3 |
| f424ef7b-e0df-4eef-93da-e6003c67c648 | 4, 10, 23 | 5 |

The no/false answer follows the square arithmetic, but each stem has a false
premise: the two shorter sides do not sum to more than the longest side. That
is a content defect, not a semantic pass because the arithmetic key is false.

## Prevention fix

Both `gen_converse_check` and `gen_tf_pyth_triple` now require positive side
lengths forming a non-degenerate triangle before using a non-right example.
Their independent verifiers reject impossible-side stems instead of returning
a valid false/no result. Existing valid right/non-right answers are unchanged.
Tests cover all four observed payloads, degenerate/non-positive cases, and
200 seeds for each generator. No broad chapter re-generation is published.

## Separate data repair, not performed

This PR changes authoring only. It does not rewrite, delete or retire existing
bank rows and does not change historic attempts or scores. It is separate from
the four exponent/theorem rows in PR113. Neither merge alone repairs production.

Prepare a separate exact-row proposal after fresh verification. Review original
replacement stems/options and answer reasoning, stage new records with new IDs,
and retire only the four confirmed invalid rows without changing historical
references. Rehearse the exact transaction on real PostgreSQL, including payload
mismatch, replay, partial overlap, history FK, rollback and concurrency checks.
Confirm current production schema and curriculum mapping. Obtain scoped approval
before any production data mutation. No combined eight-row repair is implied.
