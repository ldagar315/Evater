"""Tests for the inspiration-item rework + QC pipeline."""

from __future__ import annotations

import random

from backend.etl.rework_questions import (
    qc_check,
    rework_chapter_pack,
    shuffle_options_with_key,
    token_similarity,
    vary_incidental_numbers,
)


def _mcq(**overrides):
    base = {
        "question_text": "A farmer keeps 12 goats in a pen. Which organism ferments milk into curd?",
        "options": [
            {"id": "A", "text": "Lactobacillus"},
            {"id": "B", "text": "Amoeba"},
            {"id": "C", "text": "Paramecium"},
            {"id": "D", "text": "Euglena"},
        ],
        "correct_option_id": "B",
        "explanation": "Lactobacillus ferments milk.",
        "source_question_id": "test:q1",
    }
    # make the keyed option B the correct text
    base["options"][1]["text"] = "Lactobacillus"
    base["options"][0]["text"] = "Amoeba"
    base.update(overrides)
    return base


def test_shuffle_options_remaps_key():
    q = _mcq()
    rng = random.Random(42)
    out = shuffle_options_with_key(q, rng)
    assert out is not None
    keyed = next(o for o in out["options"] if o["id"] == out["correct_option_id"])
    assert keyed["text"] == "Lactobacillus"
    assert sorted(o["text"] for o in out["options"]) == sorted(o["text"] for o in q["options"])


def test_shuffle_rejects_bad_shape():
    assert shuffle_options_with_key({"options": [], "correct_option_id": "A"}, random.Random(1)) is None


def test_numeric_variation_skips_numbers_in_options_and_explanation():
    q = _mcq()
    rng = random.Random(7)
    out, changes = vary_incidental_numbers(q, rng)
    assert changes, "the incidental 12 should be varied"
    assert "12" not in out["question_text"]
    keyed = next(o for o in out["options"] if o["id"] == out["correct_option_id"])
    assert keyed["text"] == "Lactobacillus"  # re-solve: key still matches


def test_numeric_variation_protects_keyed_numbers():
    q = _mcq(question_text="What is 6 plus 6?", options=[
        {"id": "A", "text": "10"}, {"id": "B", "text": "11"},
        {"id": "C", "text": "12"}, {"id": "D", "text": "13"},
    ], correct_option_id="C", explanation="6 + 6 = 12.")
    out, changes = vary_incidental_numbers(q, random.Random(3))
    assert changes == []  # every number appears in options/explanation, nothing varied
    assert out["question_text"] == q["question_text"]


def test_verbatim_guard_flags_unchanged_stem():
    q = _mcq()
    qc = qc_check(q, dict(q), seen_hashes=set())
    assert qc["verbatim_guard_passed"] is False
    assert qc["similarity_to_source"] == 1.0


def test_dedupe_flags_seen_hash():
    q = _mcq()
    qc1 = qc_check(q, dict(q), seen_hashes=set())
    qc2 = qc_check(q, dict(q), seen_hashes={qc1["content_hash"]})
    assert "duplicate" in qc2["problems"]


def test_token_similarity_drops_stopwords():
    assert token_similarity("What is the cell wall?", "Cell?") == 0.5


def test_rework_chapter_pack_routes_conceptual_items_to_human_review(tmp_path):
    inspiration = {
        "curriculum_version_id": "v1",
        "chapter_id": "c1",
        "chapter_title": "Test chapter",
        "questions": [_mcq()],
        "items": [],
    }
    pack = tmp_path / "inspiration.json"
    pack.write_text(__import__("json").dumps(inspiration), encoding="utf-8")
    result = rework_chapter_pack(pack, None, tmp_path / "out")
    assert result["input_count"] == 1
    assert result["needs_human_review"] == 1
    assert result["reworked_qc_passed"] == 0
