import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app


def test_api_me_requires_auth():
    client = TestClient(app)
    resp = client.get("/api/me")
    assert resp.status_code == 401


def test_api_me_happy_path():
    from app.auth import AuthContext, require_user
    from supabase_auth.types import User
    from datetime import datetime, timezone

    def _stub_require_user():
        return AuthContext(
            jwt="jwt",
            tenant_id="tenant_abc",
            user=User(
                id="user_123",
                app_metadata={"tenant_id": "tenant_abc"},
                user_metadata={},
                aud="authenticated",
                email="user@example.com",
                created_at=datetime.now(timezone.utc),
            ),
        )

    app.dependency_overrides[require_user] = _stub_require_user
    try:
        client = TestClient(app)
        resp = client.get("/api/me")
        assert resp.status_code == 200
        assert resp.json() == {
            "user_id": "user_123",
            "email": "user@example.com",
            "tenant_id": "tenant_abc",
        }
    finally:
        app.dependency_overrides.clear()


def test_tenant_id_ignores_user_editable_metadata():
    from app.auth import _tenant_id_from_user
    from supabase_auth.types import User
    from datetime import datetime, timezone

    user = User(
        id="user_123",
        app_metadata={},
        user_metadata={"tenant_id": "attacker-controlled"},
        aud="authenticated",
        email="user@example.com",
        created_at=datetime.now(timezone.utc),
    )

    assert _tenant_id_from_user(user) is None


def test_ws_rejects_missing_auth():
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/viva"):
            pass


def test_ws_accepts_with_token(monkeypatch):
    from app.auth import AuthContext
    from supabase_auth.types import User
    from datetime import datetime, timezone

    import app.routers.viva_router as viva_router_module

    async def _stub_require_user_websocket(_websocket):
        return AuthContext(
            jwt="jwt",
            tenant_id="tenant_abc",
            user=User(
                id="user_123",
                app_metadata={"tenant_id": "tenant_abc"},
                user_metadata={},
                aud="authenticated",
                email="user@example.com",
                created_at=datetime.now(timezone.utc),
            ),
        )

    def _stub_create_supabase_client(_jwt=None):
        class StubSupabase:
            def table(self, _name):
                raise RuntimeError("DB should not be hit by this test")

        return StubSupabase()

    monkeypatch.setattr(viva_router_module, "require_user_websocket", _stub_require_user_websocket)
    monkeypatch.setattr(viva_router_module, "create_supabase_client", _stub_create_supabase_client)

    client = TestClient(app)
    with client.websocket_connect("/ws/viva?access_token=test") as ws:
        msg = ws.receive_json()
        assert msg["status"] == "connected"
        ws.close()


def test_rest_routes_require_auth():
    client = TestClient(app)

    resp = client.post("/api/gen_question", json={})
    assert resp.status_code == 401

    resp = client.post("/api/gen_answer", json={})
    assert resp.status_code == 401

    resp = client.post("/api/gen_feedback_direct", json={})
    assert resp.status_code == 401


def test_gen_question_happy_path(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from app.auth import AuthContext, require_user
    from app.models import Question
    from supabase_auth.types import User

    import app.routers.test_router as test_router_module

    def _stub_require_user():
        return AuthContext(
            jwt="jwt",
            tenant_id="tenant_abc",
            user=User(
                id="user_123",
                app_metadata={"tenant_id": "tenant_abc"},
                user_metadata={},
                aud="authenticated",
                email="user@example.com",
                created_at=datetime.now(timezone.utc),
            ),
        )

    def _stub_create_supabase_client(_jwt=None):
        return object()

    def _stub_get_chapter_summary(*, chapter_name, grade, subject, supabase_client):
        assert supabase_client is not None
        return "summary"

    def _stub_result_distribution(**_kwargs):
        return SimpleNamespace(test_structure={"mcq_single_count": 1})

    def _stub_test_generation(**_kwargs):
        return SimpleNamespace(
            test=[
                Question(
                    question_text="2+2?",
                    question_type="mcq_single",
                    difficulty="Easy",
                    question_number=1,
                    options=None,
                    contains_math_expression=True,
                )
            ]
        )

    app.dependency_overrides[require_user] = _stub_require_user
    monkeypatch.setattr(test_router_module, "create_supabase_client", _stub_create_supabase_client)
    monkeypatch.setattr(test_router_module, "get_chapter_summary", _stub_get_chapter_summary)
    monkeypatch.setattr(test_router_module, "result_distribution", _stub_result_distribution)
    monkeypatch.setattr(test_router_module, "result_distribution_mcq", _stub_result_distribution)
    monkeypatch.setattr(test_router_module, "result_distribution_subjective", _stub_result_distribution)
    monkeypatch.setattr(test_router_module, "test_generation", _stub_test_generation)

    try:
        client = TestClient(app)
        payload = {
            "grade": "6",
            "subject": "Math",
            "topic": "Addition",
            "difficulty_level": "Easy",
            "length": "Short",
            "test_type": "mixed",
            "special_instructions": [],
        }
        resp = client.post("/api/gen_question", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "questions" in body
        assert body["questions"][0]["maximum_marks"] == 1
    finally:
        app.dependency_overrides.clear()


def test_gen_answer_happy_path(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from app.auth import AuthContext, require_user
    from app.models import Answer, Feedback
    from supabase_auth.types import User

    import app.routers.feedback_router as feedback_router_module

    def _stub_require_user():
        return AuthContext(
            jwt="jwt",
            tenant_id="tenant_abc",
            user=User(
                id="user_123",
                app_metadata={"tenant_id": "tenant_abc"},
                user_metadata={},
                aud="authenticated",
                email="user@example.com",
                created_at=datetime.now(timezone.utc),
            ),
        )

    def _stub_answer_ocr_extraction(_image_urls):
        return "ocr text"

    def _stub_answer_seperation(*, answer_sheet_text):
        assert answer_sheet_text == "ocr text"
        return SimpleNamespace(answers=[Answer(answer="4", question_number=1)])

    def _stub_feedback_generation(*, question, answer):
        assert question.question_number == 1
        assert answer == "4"
        return SimpleNamespace(
            feedback=Feedback(
                question_number=1,
                explanation="ok",
                max_scored=1,
                error_type="No mistake",
                next_step="none",
            )
        )

    app.dependency_overrides[require_user] = _stub_require_user
    monkeypatch.setattr(feedback_router_module, "answer_ocr_extraction", _stub_answer_ocr_extraction)
    monkeypatch.setattr(feedback_router_module, "answer_seperation", _stub_answer_seperation)
    monkeypatch.setattr(feedback_router_module, "feedback_generation", _stub_feedback_generation)

    try:
        client = TestClient(app)
        payload = {
            "image_url": ["https://example.com/a.png"],
            "questions": {
                "questions": [
                    {
                        "question_text": "2+2?",
                        "question_type": "mcq_single",
                        "difficulty": "Easy",
                        "question_number": 1,
                        "options": None,
                        "contains_math_expression": True,
                        "maximum_marks": 1,
                    }
                ]
            },
        }
        resp = client.post("/api/gen_answer", json=payload)
        assert resp.status_code == 200
        merged = resp.json()["merged"]
        assert merged[0]["question_number"] == 1
        assert merged[0]["answer"]["answer"] == "4"
        assert merged[0]["feedback"]["max_scored"] == 1
    finally:
        app.dependency_overrides.clear()
