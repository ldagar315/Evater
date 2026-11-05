"""Mocked-LLM tests for the core test-generation path (testgen_service)."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import testgen_service
from app.models import InputDataQuestion


def _req(test_type):
    return InputDataQuestion(
        grade="8", subject="Maths", topic="Linear Equations",
        difficulty_level="Medium", length="short",
        test_type=test_type, special_instructions=["show steps"],
    )


def _stub_gen():
    q = types.SimpleNamespace(model_dump=lambda: {
        "question_text": "What is 2+2?", "question_type": "short_answer",
        "question_number": 1, "difficulty": "Easy",
        "contains_math_expression": False, "options": None})
    return types.SimpleNamespace(test=[q], test_structure={"short_answer_count": 1})


def test_objective_routes_to_mcq_distribution():
    seen = {}

    def mcq(**kwargs):
        seen["mcq"] = kwargs
        return types.SimpleNamespace(test_structure={"mcq_single_count": 5})

    out = testgen_service.generate_questions(
        _req("objective"), jwt="jwt-1",
        distribution_fns={"result_distribution_mcq": mcq},
        chapter_summary_fn=lambda **kw: "chapter-summary",
        test_generation_fn=lambda **kw: _stub_gen(),
        client_factory=lambda jwt: ("client", jwt),
    )
    assert "mcq" in seen
    assert out[0]["maximum_marks"] == 2  # short_answer -> 2 marks


def test_mixed_uses_default_distribution_and_passes_jwt_and_summary():
    seen = {}

    def default_dist(**kwargs):
        seen["dist"] = kwargs
        return types.SimpleNamespace(test_structure={"short_answer_count": 1})

    def summary_fn(**kwargs):
        seen["summary"] = kwargs
        return "SUM"

    def gen_fn(**kwargs):
        seen["gen"] = kwargs
        return _stub_gen()

    testgen_service.generate_questions(
        _req("mixed"), jwt="jwt-2",
        distribution_fns={"result_distribution": default_dist},
        chapter_summary_fn=summary_fn,
        test_generation_fn=gen_fn,
        client_factory=lambda jwt: ("client", jwt),
    )
    assert seen["summary"]["supabase_client"] == ("client", "jwt-2")
    assert seen["gen"]["topic_covered"] == "SUM"
    assert seen["gen"]["test_structure"] == {"short_answer_count": 1}
    assert seen["dist"]["special_instructions"] == ["show steps"]


def test_subjective_routes_to_subjective_distribution():
    seen = {}
    testgen_service.generate_questions(
        _req("subjective"),
        distribution_fns={"result_distribution_subjective": lambda **kw: (seen.update(s=kw), types.SimpleNamespace(test_structure={}))[1]},
        chapter_summary_fn=lambda **kw: "",
        test_generation_fn=lambda **kw: _stub_gen(),
        client_factory=lambda jwt: None,
    )
    assert "s" in seen
