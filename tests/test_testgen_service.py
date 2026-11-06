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


def _stub_gen_with_mcq_options():
    """Fixture fidelity: the real gen_question wire carries options as
    {text, is_correct} objects (MCQOption), not plain strings."""
    q = types.SimpleNamespace(model_dump=lambda: {
        "question_text": "Pick the even number.",
        "question_type": "mcq_single",
        "question_number": 1,
        "difficulty": "Easy",
        "contains_math_expression": False,
        "options": [
            {"text": "3", "is_correct": False},
            {"text": "4", "is_correct": True},
        ],
    })
    return types.SimpleNamespace(test=[q], test_structure={"mcq_single_count": 1})


def test_answer_keys_stripped_from_mcq_options():
    """Integrity: is_correct must never leave the service boundary."""
    out = testgen_service.generate_questions(
        _req("objective"), jwt="jwt-1",
        distribution_fns={"result_distribution_mcq": lambda **kw: types.SimpleNamespace(test_structure={"mcq_single_count": 1})},
        chapter_summary_fn=lambda **kw: "",
        test_generation_fn=lambda **kw: _stub_gen_with_mcq_options(),
        client_factory=lambda jwt: ("client", jwt),
    )
    opts = out[0]["options"]
    assert [o["text"] for o in opts] == ["3", "4"]  # order and text preserved
    assert all("is_correct" not in o for o in opts)


def test_router_never_ships_answer_keys(monkeypatch):
    """Negative fixture: even if the service regressed and returned options
    WITH is_correct, the wired response_model must not ship them."""
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from supabase_auth.types import User

    import app.testgen_service as ts
    from app.auth import AuthContext, require_user
    from app.main import app

    app.dependency_overrides[require_user] = lambda: AuthContext(
        jwt="j", tenant_id="t", user=User(
            id="u1", app_metadata={}, user_metadata={}, aud="authenticated",
            email="u@e.com", created_at=datetime.now(timezone.utc)))

    def leaky_service(*args, **kwargs):
        return [{
            "question_text": "Pick the even number.",
            "question_type": "mcq_single",
            "question_number": 1,
            "difficulty": "Easy",
            "contains_math_expression": False,
            "maximum_marks": 1,
            "options": [
                {"text": "3", "is_correct": False},
                {"text": "4", "is_correct": True},
            ],
        }]

    monkeypatch.setattr(ts, "generate_questions", leaky_service)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/gen_question", json={
            "grade": "8", "subject": "Maths", "topic": "Linear Equations",
            "difficulty_level": "Medium", "length": "short",
            "test_type": "objective", "special_instructions": [],
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        opts = body["questions"][0]["options"]
        assert [o["text"] for o in opts] == ["3", "4"]
        assert all("is_correct" not in o for o in opts)
        assert "is_correct" not in resp.text
    finally:
        app.dependency_overrides.clear()
