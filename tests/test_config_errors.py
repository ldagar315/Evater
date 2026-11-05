import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings, missing_critical_settings
from app.errors import AppError, ConfigurationError, RequestIDMiddleware, register_error_handlers
from app.remote_image import RemoteImageError


def test_settings_read_fresh_from_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://a.example")
    assert get_settings().supabase_url == "https://a.example"
    monkeypatch.setenv("SUPABASE_URL", "https://b.example")
    assert get_settings().supabase_url == "https://b.example"


def test_env_aliases(monkeypatch):
    for k in ("ENV", "APP_ENV", "NODE_ENV"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NODE_ENV", "production")
    assert get_settings().env == "production"


def test_missing_critical_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_API_KEY", "")
    assert set(missing_critical_settings(get_settings())) == {"supabase_url", "supabase_api_key"}


def _app():
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    @app.get("/app-error")
    def a():
        raise AppError("boom", status_code=502)

    @app.get("/remote")
    def r():
        raise RemoteImageError(status_code=400, detail="bad image")

    @app.get("/config-error")
    def rt():
        raise ConfigurationError()

    @app.get("/runtime")
    def rtp():
        raise RuntimeError("some non-config bug with secret-ish detail")

    @app.get("/generic")
    def g():
        raise ValueError("internal detail")

    return TestClient(app, raise_server_exceptions=False)


def test_handlers_map_errors_without_leaking_detail():
    c = _app()
    r = c.get("/app-error")
    assert r.status_code == 502 and r.json() == {"detail": "boom"}
    assert "x-request-id" in r.headers
    r = c.get("/remote")
    assert r.status_code == 400 and r.json() == {"detail": "bad image"}
    r = c.get("/config-error")
    assert r.status_code == 500 and r.json() == {"detail": "Server configuration error."}
    r = c.get("/runtime")
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error."}
    assert "secret-ish" not in r.text
    r = c.get("/generic")
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error."}
