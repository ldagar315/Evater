def test_create_supabase_client_attaches_authorization(monkeypatch):
    import app.supabase_client as supabase_client_module

    seen = {}

    def _stub_create_client(url, key, options=None):
        seen["url"] = url
        seen["key"] = key
        seen["options"] = options
        return object()

    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_API_KEY", "anon")
    monkeypatch.setattr(supabase_client_module, "create_client", _stub_create_client)

    supabase_client_module.create_supabase_client("jwt_123")

    assert seen["url"] == "http://localhost"
    assert seen["key"] == "anon"
    assert seen["options"] is not None
    assert seen["options"].headers["Authorization"] == "Bearer jwt_123"
    assert seen["options"].headers["apikey"] == "anon"
