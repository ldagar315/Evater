from types import SimpleNamespace
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.auth import AuthContext, require_user
from app.main import app
from app.routers import question_bank_router
from supabase_auth.types import User


QUESTION_ID = "00000000-0000-0000-0000-000000000123"
USER_ID = "00000000-0000-0000-0000-000000000456"


def test_question_flag_endpoint_requires_authentication():
    response = TestClient(app).post(f"/api/v1/questions/{QUESTION_ID}/flag")
    assert response.status_code == 401


def test_question_flag_endpoint_is_idempotent(monkeypatch):
    class Query:
        def __init__(self, table_name):
            self.table_name = table_name
            self.rows = []

        def select(self, *_args):
            return self

        def eq(self, *_args):
            return self

        def upsert(self, rows, **kwargs):
            assert self.table_name == "question_review_flags"
            assert kwargs["on_conflict"] == "user_id,question_id"
            self.rows = rows
            return self

        def execute(self):
            if self.table_name == "question_bank":
                return SimpleNamespace(data=[{"id": QUESTION_ID}])
            return SimpleNamespace(data=self.rows)

    class Client:
        def table(self, table_name):
            return Query(table_name)

    def authenticated_user():
        return AuthContext(
            user=User(
                id=USER_ID,
                app_metadata={},
                user_metadata={},
                aud="authenticated",
                email="student@example.com",
                created_at=datetime.now(timezone.utc),
            ),
            jwt="test-jwt",
            tenant_id=None,
        )

    monkeypatch.setattr(question_bank_router, "create_supabase_service_client", lambda: Client())
    app.dependency_overrides[require_user] = authenticated_user
    try:
        response = TestClient(app).post(f"/api/v1/questions/{QUESTION_ID}/flag")
    finally:
        app.dependency_overrides.pop(require_user, None)

    assert response.status_code == 200
    assert response.json() == {"question_id": QUESTION_ID, "flagged": True}
