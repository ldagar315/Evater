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


def test_service_returns_keyed_questions_for_server_side_persistence():
    """The service keeps keys internally: the router persists them to the
    key store and only sanitizes at the response boundary."""
    out = testgen_service.generate_questions(
        _req("objective"), jwt="jwt-1",
        distribution_fns={"result_distribution_mcq": lambda **kw: types.SimpleNamespace(test_structure={"mcq_single_count": 1})},
        chapter_summary_fn=lambda **kw: "",
        test_generation_fn=lambda **kw: _stub_gen_with_mcq_options(),
        client_factory=lambda jwt: ("client", jwt),
    )
    keys = [o["is_correct"] for o in out[0]["options"]]
    assert keys == [False, True]


def test_sanitize_questions_strips_keys_without_mutating_originals():
    keyed = [{
        "question_text": "q", "question_type": "mcq_single", "question_number": 1,
        "difficulty": "Easy", "contains_math_expression": False, "maximum_marks": 1,
        "options": [{"text": "3", "is_correct": False}, {"text": "4", "is_correct": True}],
    }]
    out = testgen_service.sanitize_questions(keyed)
    assert out[0]["options"] == [{"text": "3"}, {"text": "4"}]
    assert keyed[0]["options"][0]["is_correct"] is False  # original untouched


def test_gen_question_route_persists_keys_and_ships_sanitized(monkeypatch):
    """End-to-end boundary: the route stores keyed questions server-side and
    responds with key-free options + an opaque key_id."""
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from supabase_auth.types import User

    import app.testgen_service as ts
    from app.auth import AuthContext, require_user
    from app.main import app
    from app.routers import test_router

    app.dependency_overrides[require_user] = lambda: AuthContext(
        jwt="j", tenant_id="t", user=User(
            id="u1", app_metadata={}, user_metadata={}, aud="authenticated",
            email="u@e.com", created_at=datetime.now(timezone.utc)))

    keyed = [{
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
    monkeypatch.setattr(ts, "generate_questions", lambda *a, **kw: keyed)

    inserted = {}

    class FakeTable:
        def insert(self, row):
            inserted["row"] = row
            return self

        def execute(self):
            return types.SimpleNamespace(data=[{"id": "12345678-1234-5678-1234-567812345678"}])

    class FakeServiceClient:
        def table(self, name):
            assert name == "generated_test_keys"
            return FakeTable()

    monkeypatch.setattr(test_router, "create_supabase_service_client",
                        lambda: FakeServiceClient())
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/gen_question", json={
            "grade": "8", "subject": "Maths", "topic": "Linear Equations",
            "difficulty_level": "Medium", "length": "short",
            "test_type": "objective", "special_instructions": [],
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["key_id"] == "12345678-1234-5678-1234-567812345678"
        opts = body["questions"][0]["options"]
        assert [o["text"] for o in opts] == ["3", "4"]
        assert "is_correct" not in resp.text
        # keys persisted server-side, owned by the caller
        assert inserted["row"]["user_id"] == "u1"
        assert inserted["row"]["questions"][0]["options"][1]["is_correct"] is True
    finally:
        app.dependency_overrides.clear()


def test_gen_question_route_fails_closed_when_key_store_unavailable(monkeypatch):
    """No key persistence, no test: a generation whose keys were not stored
    could never be graded objectively, so it must not ship."""
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from supabase_auth.types import User

    import app.testgen_service as ts
    from app.auth import AuthContext, require_user
    from app.main import app
    from app.routers import test_router

    app.dependency_overrides[require_user] = lambda: AuthContext(
        jwt="j", tenant_id="t", user=User(
            id="u1", app_metadata={}, user_metadata={}, aud="authenticated",
            email="u@e.com", created_at=datetime.now(timezone.utc)))
    monkeypatch.setattr(ts, "generate_questions", lambda *a, **kw: [])
    monkeypatch.setattr(test_router, "create_supabase_service_client",
                        lambda: (_ for _ in ()).throw(RuntimeError("no service key")))
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/gen_question", json={
            "grade": "8", "subject": "Maths", "topic": "Linear Equations",
            "difficulty_level": "Medium", "length": "short",
            "test_type": "objective", "special_instructions": [],
        })
        assert resp.status_code == 500, resp.text
    finally:
        app.dependency_overrides.clear()
