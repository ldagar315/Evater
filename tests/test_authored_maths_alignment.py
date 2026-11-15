"""Answer correctness is not enough: authored questions must fit their concept."""
import random
from dataclasses import replace
from uuid import uuid5

import pytest

from backend.etl import author_class8_maths as author
from backend.etl.class8_maths_catalog import get_chapter


def test_original_power_question_cannot_be_mapped_into_pythagoras():
    item = author.gen_power(random.Random(0), 9)
    assert author.v_power(item.question_text) == item.key
    # Reproduce the formerly permitted remap. A valid concept UUID is not proof
    # that a correctly answered exponent exercise belongs to that concept.
    item = replace(item, topic_slug="pythagoras-statement")
    with pytest.raises(ValueError, match="Invalid Pythagoras family/concept"):
        author.build_candidate(item, get_chapter(9), random.Random(0))


@pytest.mark.parametrize("seed", range(12))
def test_pythagoras_pack_has_real_theorem_statements_and_no_exponent_family(seed):
    questions, report = author.author_chapter(9, seed)
    assert not any("ERROR" in line or "DROPPED" in line for line in report)
    concept = str(uuid5(get_chapter(9).id, "concept:pythagoras-statement"))
    statements = [q for q in questions if q["concept_id"] == concept]
    assert len(statements) == 4
    assert all(q["generation_spec"]["generator_family"] == "pyth_statement" for q in statements)
    for question in statements:
        key = author.v_pyth_statement(question["question_text"])
        options = question["options"]
        assert next(o["text"] for o in options if o["id"] == question["correct_option_id"]) == key
        assert len({o["text"] for o in options}) == 4
        assert "right-angled triangle" in question["question_text"]
        assert "hypotenuse" in question["explanation"]
    assert not any(q["generation_spec"]["generator_family"] == "power" for q in questions)


def test_power_play_still_contains_exponent_questions():
    questions, _ = author.author_chapter(2, 20261024)
    powers = [q for q in questions if q["generation_spec"]["generator_family"] == "power"]
    assert len(powers) == 6
    concept = str(uuid5(get_chapter(2).id, "concept:exponents"))
    assert all(q["concept_id"] == concept for q in powers)


def test_generation_drops_wrong_family_even_when_slug_exists(monkeypatch):
    def wrong_family(rng, chapter):
        return replace(author.gen_power(rng, chapter), topic_slug="pythagoras-statement")
    monkeypatch.setitem(author.GENERATORS, 9, [(wrong_family, author.v_power, 1)])
    questions, report = author.author_chapter(9, 0)
    assert questions == []
    assert any("Invalid Pythagoras family/concept" in line for line in report)


@pytest.mark.parametrize("change", [
    {"chapter": 2}, {"topic_slug": "not-a-concept"},
    {"topic_slug": "applications"}, {"family": "unknown"},
])
def test_candidate_boundary_rejects_bad_identity(change):
    item = replace(author.gen_pyth_statement(random.Random(0), 9), **change)
    with pytest.raises(ValueError):
        author.build_candidate(item, get_chapter(9), random.Random(0))


@pytest.mark.parametrize("chapter", range(1, 15))
def test_all_current_chapter_generators_pass_identity_guard(chapter):
    questions, report = author.author_chapter(chapter, 20261022 + chapter)
    assert questions
    assert not any("ERROR" in line for line in report)


def test_statement_verifier_rejects_missing_right_angle_or_ambiguous_labels():
    item = author.gen_pyth_statement(random.Random(0), 9)
    with pytest.raises(ValueError):
        author.v_pyth_statement(item.question_text.replace("right-angled triangle", "triangle"))
    with pytest.raises(ValueError):
        author.v_pyth_statement("In a right-angled triangle, a is the hypotenuse and a and b are the legs. Which equation states the Pythagoras theorem?")


def test_publisher_rejects_stale_exponent_pack_before_any_remote_client(monkeypatch):
    from backend.etl.models import QuestionCandidate
    from backend.etl.seed_class8_maths import seed_remote
    from backend.etl.validate import QuestionPackError
    import app.supabase_client as supabase
    questions, _ = author.author_chapter(9, 0)
    row = next(q for q in questions if q["generation_spec"]["generator_family"] == "pyth_statement")
    stale = QuestionCandidate.model_validate(row)
    stale.generation_spec["generator_family"] = "power"
    calls = []
    monkeypatch.setattr(supabase, "create_supabase_service_client", lambda: calls.append(True))
    with pytest.raises(QuestionPackError, match="Invalid authored Pythagoras"):
        seed_remote([(get_chapter(9), [stale])], replace_existing=True)
    assert calls == []


def test_correct_pythagoras_pack_passes_publisher_preflight():
    from backend.etl.models import QuestionCandidate
    from backend.etl.seed_class8_maths import validate_authored_pythagoras_pack
    questions, _ = author.author_chapter(9, 0)
    candidates = [QuestionCandidate.model_validate(q) for q in questions if q.get("question_type", "mcq_single") == "mcq_single"]
    validate_authored_pythagoras_pack(get_chapter(9), candidates)
