"""Route test: GET /api/v1/chapters through ChapterListResponse."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from supabase_auth.types import User

import app.routers.question_bank_router as qb_router
from app.auth import AuthContext, require_user
from app.main import app


def _auth():
    return AuthContext(jwt="j", tenant_id="t", user=User(
        id="u1", app_metadata={}, user_metadata={}, aud="authenticated",
        email="u@e.com", created_at=datetime.now(timezone.utc)))


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_a, **_kw):
        return self

    def eq(self, *_a, **_kw):
        return self

    def in_(self, *_a, **_kw):
        return self

    def order(self, *_a, **_kw):
        return self

    def execute(self):
        return types.SimpleNamespace(data=self.rows)


class _Client:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return _Query(self.tables[name])


def test_chapters_returns_typed_list(monkeypatch):
    rows = {
        "curriculum_versions": [{"id": "cv-1", "grade": 8, "subject": "Science", "version_label": "v1"}],
        "chapters": [
            {"id": "ch-1", "sequence_number": 1, "title": "Crop Production",
             "description": None, "curriculum_version_id": "cv-1"},
            {"id": "ch-2", "sequence_number": 2, "title": "Microorganisms",
             "description": "Tiny things", "curriculum_version_id": "cv-1"},
        ],
    }
    app.dependency_overrides[require_user] = _auth
    monkeypatch.setattr(qb_router, "create_supabase_service_client", lambda: _Client(rows))
    try:
        resp = TestClient(app).get("/api/v1/chapters")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert [c["title"] for c in body["chapters"]] == ["Crop Production", "Microorganisms"]
        assert body["chapters"][0] == {
            "id": "ch-1", "sequence_number": 1, "title": "Crop Production",
            "description": None, "curriculum_version_id": "cv-1"}
    finally:
        app.dependency_overrides.clear()


def test_chapters_empty_when_no_published_curriculum(monkeypatch):
    rows = {"curriculum_versions": [], "chapters": []}
    app.dependency_overrides[require_user] = _auth
    monkeypatch.setattr(qb_router, "create_supabase_service_client", lambda: _Client(rows))
    try:
        resp = TestClient(app).get("/api/v1/chapters?subject=Maths")
        assert resp.status_code == 200 and resp.json() == {"chapters": []}
    finally:
        app.dependency_overrides.clear()


def test_chapters_requires_auth():
    assert TestClient(app).get("/api/v1/chapters").status_code == 401
