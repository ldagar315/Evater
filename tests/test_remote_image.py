import ipaddress

import httpx
import pytest


def _patch_public_resolution(monkeypatch, module, ip="1.1.1.1"):
    monkeypatch.setattr(module, "_resolve_host_ips", lambda _host, _port: {ipaddress.ip_address(ip)})


def test_validate_remote_image_url_allows_allowlisted_https(monkeypatch):
    import app.remote_image as m

    _patch_public_resolution(monkeypatch, m)

    url = "https://project.supabase.co/storage/v1/object/public/bucket/a.png"
    assert m.validate_remote_image_url(url, allowed_hosts={"project.supabase.co"}) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://project.supabase.co/storage/v1/object/public/bucket/a.png",
        "file:///etc/passwd",
        "data:image/png;base64,AA==",
    ],
)
def test_validate_remote_image_url_rejects_non_https(monkeypatch, url):
    import app.remote_image as m

    _patch_public_resolution(monkeypatch, m)

    with pytest.raises(m.RemoteImageError) as excinfo:
        m.validate_remote_image_url(url, allowed_hosts={"project.supabase.co"})
    assert excinfo.value.status_code == 400


def test_validate_remote_image_url_rejects_userinfo(monkeypatch):
    import app.remote_image as m

    _patch_public_resolution(monkeypatch, m)

    url = "https://user:pass@project.supabase.co/storage/v1/object/public/bucket/a.png"
    with pytest.raises(m.RemoteImageError) as excinfo:
        m.validate_remote_image_url(url, allowed_hosts={"project.supabase.co"})
    assert excinfo.value.status_code == 400


def test_validate_remote_image_url_rejects_ip_literal(monkeypatch):
    import app.remote_image as m

    _patch_public_resolution(monkeypatch, m)

    url = "https://127.0.0.1/a.png"
    with pytest.raises(m.RemoteImageError) as excinfo:
        m.validate_remote_image_url(url, allowed_hosts={"project.supabase.co"})
    assert excinfo.value.status_code == 400


def test_fetch_image_bytes_returns_bytes(monkeypatch):
    import app.remote_image as m

    _patch_public_resolution(monkeypatch, m)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "project.supabase.co"
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=b"\x89PNG\r\n\x1a\nfake",
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    try:
        data = m.fetch_image_bytes(
            "https://project.supabase.co/storage/v1/object/public/bucket/a.png",
            allowed_hosts={"project.supabase.co"},
            client=client,
        )
        assert data.startswith(b"\x89PNG")
    finally:
        client.close()


def test_fetch_image_bytes_blocks_redirect_to_unallowlisted_host(monkeypatch):
    import app.remote_image as m

    _patch_public_resolution(monkeypatch, m)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://evil.example/a.png"},
            content=b"",
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    try:
        with pytest.raises(m.RemoteImageError) as excinfo:
            m.fetch_image_bytes(
                "https://project.supabase.co/a",
                allowed_hosts={"project.supabase.co"},
                client=client,
            )
        assert excinfo.value.status_code == 400
    finally:
        client.close()


def test_fetch_image_bytes_allows_redirect_within_allowlist(monkeypatch):
    import app.remote_image as m

    _patch_public_resolution(monkeypatch, m)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(
                302,
                headers={"location": "/b"},
                content=b"",
                request=request,
            )
        if request.url.path == "/b":
            return httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=b"\xff\xd8fakejpg",
                request=request,
            )
        return httpx.Response(404, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    try:
        data = m.fetch_image_bytes(
            "https://project.supabase.co/a",
            allowed_hosts={"project.supabase.co"},
            client=client,
        )
        assert data.startswith(b"\xff\xd8")
    finally:
        client.close()


def test_fetch_image_bytes_enforces_max_size(monkeypatch):
    import app.remote_image as m

    _patch_public_resolution(monkeypatch, m)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png", "content-length": "10"},
            content=b"0123456789",
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    try:
        with pytest.raises(m.RemoteImageError) as excinfo:
            m.fetch_image_bytes(
                "https://project.supabase.co/a",
                allowed_hosts={"project.supabase.co"},
                max_bytes=5,
                client=client,
            )
        assert excinfo.value.status_code == 413
    finally:
        client.close()

