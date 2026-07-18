from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.cors import add_cors_middleware


def test_preflight_allowed_origin_sets_cors_headers(monkeypatch):
    monkeypatch.setenv("APP_ORIGINS", "https://example.com")
    monkeypatch.setenv("ENV", "production")

    app = FastAPI()
    add_cors_middleware(app)
    app.add_api_route("/api/gen_question", lambda: {"ok": True}, methods=["POST"])
    client = TestClient(app)

    resp = client.options(
        "/api/gen_question",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "https://example.com"
    assert resp.headers.get("vary") == "Origin"
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_preflight_disallowed_origin_omits_allow_origin(monkeypatch):
    monkeypatch.setenv("APP_ORIGINS", "https://example.com")
    monkeypatch.setenv("ENV", "production")

    app = FastAPI()
    add_cors_middleware(app)
    app.add_api_route("/api/gen_question", lambda: {"ok": True}, methods=["POST"])
    client = TestClient(app)

    resp = client.options(
        "/api/gen_question",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert resp.status_code in (200, 204, 400)
    assert resp.headers.get("access-control-allow-origin") is None


def test_no_wildcard_origin_in_production(monkeypatch):
    monkeypatch.setenv("APP_ORIGINS", "https://example.com")
    monkeypatch.setenv("ENV", "production")

    app = FastAPI()
    add_cors_middleware(app)
    app.add_api_route("/api/gen_question", lambda: {"ok": True}, methods=["POST"])
    client = TestClient(app)

    resp = client.options(
        "/api/gen_question",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resp.headers.get("access-control-allow-origin") != "*"


def test_localhost_not_allowed_when_env_unset(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("NODE_ENV", raising=False)
    monkeypatch.delenv("APP_ORIGINS", raising=False)

    app = FastAPI()
    add_cors_middleware(app)
    app.add_api_route("/api/gen_question", lambda: {"ok": True}, methods=["POST"])
    client = TestClient(app)

    resp = client.options(
        "/api/gen_question",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resp.headers.get("access-control-allow-origin") is None


def test_evater_production_origin_is_allowed_by_default(monkeypatch):
    monkeypatch.delenv("APP_ORIGINS", raising=False)
    monkeypatch.setenv("ENV", "production")

    app = FastAPI()
    add_cors_middleware(app)
    app.add_api_route("/api/gen_question", lambda: {"ok": True}, methods=["POST"])
    client = TestClient(app)

    resp = client.options(
        "/api/gen_question",
        headers={
            "Origin": "https://evater.xyz",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "https://evater.xyz"
