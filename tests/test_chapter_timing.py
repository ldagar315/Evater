"""Chapter phase measurements must not change auth or disclose request data."""
from datetime import datetime, timezone
import logging
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from supabase_auth.types import User
import app.auth as auth_module
import app.routers.question_bank_router as router
from app.main import app


def user():
    return User(id='private-user-id', app_metadata={}, user_metadata={},
                aud='authenticated', email='private@example.invalid',
                created_at=datetime.now(timezone.utc))


def test_auth_retains_authoritative_validation_and_records_duration(monkeypatch):
    calls = []
    clock = iter([1.0, 1.125])
    monkeypatch.setattr(auth_module, 'perf_counter', lambda: next(clock))
    monkeypatch.setattr(auth_module, 'create_supabase_client', lambda: types.SimpleNamespace(
        auth=types.SimpleNamespace(get_user=lambda jwt: calls.append(jwt) or types.SimpleNamespace(user=user()))))
    result = auth_module.require_user('Bearer sensitive-jwt')
    assert calls == ['sensitive-jwt']
    assert result.jwt == 'sensitive-jwt' and result.user.id == 'private-user-id'
    assert result.validation_ms == 125.0


@pytest.mark.parametrize('header', [None, '', 'Basic secret', 'Bearer '])
def test_auth_failure_still_rejects_without_client(monkeypatch, header):
    def unexpected():
        pytest.fail('Invalid header reached auth client')
    monkeypatch.setattr(auth_module, 'create_supabase_client', unexpected)
    with pytest.raises(HTTPException) as error:
        auth_module.require_user(header)
    assert error.value.status_code == 401


@pytest.mark.parametrize('auth_ms,expected', [(125.0, '125.0'), (None, 'unknown')])
@pytest.mark.parametrize('subject,label', [('Science', 'Science'), ('Mathematics', 'Mathematics'),
                                         ('private-subject-injected\nnew log', 'other')])
def test_catalogue_logs_only_durations_and_bounded_label(monkeypatch, caplog, auth_ms, expected, subject, label):
    class Query:
        def select(self, *args): return self
        def eq(self, *args): return self
        def order(self, *args): return self
        def execute(self): return types.SimpleNamespace(data=[])
    clock = iter([1.0, 1.01, 1.06])
    monkeypatch.setattr(router, 'perf_counter', lambda: next(clock))
    monkeypatch.setattr(router, 'create_supabase_service_client', lambda: types.SimpleNamespace(table=lambda _: Query()))
    app.dependency_overrides[auth_module.require_user] = lambda: auth_module.AuthContext(
        jwt='sensitive-jwt', user=user(), tenant_id='private-tenant', validation_ms=auth_ms)
    try:
        with caplog.at_level(logging.INFO, logger=router.__name__):
            response = TestClient(app).get('/api/v1/chapters', params={'subject': subject})
        assert response.status_code == 200 and response.json() == {'chapters': []}
        assert 'server-timing' not in response.headers
        messages = [r.getMessage() for r in caplog.records if r.name == router.__name__]
        assert messages == [f'chapter_catalogue_timing auth_ms={expected} client_ms=10.0 query_ms=50.0 subject={label}']
        assert not any(secret in messages[0] for secret in ['sensitive-jwt', 'private-user', 'private-tenant', 'private@example', 'private-subject'])
    finally:
        app.dependency_overrides.clear()
