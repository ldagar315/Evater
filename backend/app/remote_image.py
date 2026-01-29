"""
Helpers for safely fetching remote images for OCR.

We intentionally do NOT allow arbitrary URL fetches from the backend, since that
can be abused for SSRF (e.g. metadata endpoints, internal services, port scans).

By default, only images hosted on the configured Supabase project host
(`SUPABASE_URL`) are allowed. You can extend the allowlist via the
`OCR_ALLOWED_IMAGE_HOSTS` env var (comma-separated hostnames).
"""

from __future__ import annotations

import ipaddress
import os
import socket
import base64
from functools import lru_cache
from typing import Iterable, Optional, Set
from urllib.parse import urljoin, urlsplit

import httpx


class RemoteImageError(Exception):
    def __init__(self, *, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _parse_csv_hosts(value: str) -> Set[str]:
    hosts: Set[str] = set()
    for item in value.split(","):
        host = item.strip().lower()
        if host:
            hosts.add(host)
    return hosts


def get_allowed_image_hosts() -> Set[str]:
    """
    Returns the hostname allowlist for remote image fetching.

    Defaults to the hostname derived from SUPABASE_URL.
    Additional hosts can be added via OCR_ALLOWED_IMAGE_HOSTS.
    """
    hosts: Set[str] = set()

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    if supabase_url:
        parsed = urlsplit(supabase_url)
        if parsed.hostname:
            hosts.add(parsed.hostname.lower())

    extra = os.getenv("OCR_ALLOWED_IMAGE_HOSTS", "").strip()
    if extra:
        hosts |= _parse_csv_hosts(extra)

    return hosts


def _is_disallowed_ip(ip: ipaddress._BaseAddress) -> bool:
    # Block anything that isn't globally routable.
    return not ip.is_global


@lru_cache(maxsize=64)
def _resolve_host_ips(hostname: str, port: int) -> Set[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return set()

    ips: Set[ipaddress._BaseAddress] = set()
    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ips.add(ipaddress.ip_address(ip_str))
        except ValueError:
            # If it's not a valid IP, ignore it.
            continue
    return ips


def validate_remote_image_url(url: str, *, allowed_hosts: Optional[Iterable[str]] = None) -> str:
    """
    Validate a user-provided image URL before any network fetch.

    Rules:
    - https only
    - no userinfo (username:password@host)
    - port must be 443 (or omitted)
    - hostname must be in the allowlist
    - hostname must not be an IP literal
    - DNS resolution must not include private/reserved IPs (defense-in-depth)
    """
    if not url or not isinstance(url, str):
        raise RemoteImageError(status_code=400, detail="Missing image URL.")

    parsed = urlsplit(url)

    if parsed.scheme.lower() != "https":
        raise RemoteImageError(status_code=400, detail="Only https image URLs are allowed.")

    if parsed.username or parsed.password:
        raise RemoteImageError(status_code=400, detail="Image URL must not include userinfo.")

    host = (parsed.hostname or "").lower()
    if not host:
        raise RemoteImageError(status_code=400, detail="Image URL must include a hostname.")

    # Reject IP-literals directly.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        raise RemoteImageError(status_code=400, detail="IP-literal hosts are not allowed for image URLs.")

    port = parsed.port
    if port not in (None, 443):
        raise RemoteImageError(status_code=400, detail="Only port 443 is allowed for image URLs.")

    allow = {h.lower() for h in (allowed_hosts or get_allowed_image_hosts()) if h}
    if not allow:
        raise RemoteImageError(
            status_code=500,
            detail="Remote image fetching is not configured (no allowed hosts).",
        )
    if host not in allow:
        raise RemoteImageError(status_code=400, detail=f"Image host '{host}' is not allowlisted.")

    # Defense-in-depth: ensure the allowlisted hostname doesn't resolve to non-global IPs.
    resolved_ips = _resolve_host_ips(host, 443)
    if not resolved_ips:
        raise RemoteImageError(status_code=400, detail="Image host could not be resolved.")
    for resolved_ip in resolved_ips:
        if _is_disallowed_ip(resolved_ip):
            raise RemoteImageError(status_code=400, detail="Image host resolves to a non-public IP.")

    return url


def _fetch_image(
    url: str,
    *,
    allowed_hosts: Optional[Iterable[str]] = None,
    max_bytes: int = 10 * 1024 * 1024,
    max_redirects: int = 3,
    timeout: Optional[httpx.Timeout] = None,
    client: Optional[httpx.Client] = None,
) -> tuple[bytes, str]:
    allowed = set(allowed_hosts or get_allowed_image_hosts())
    current_url = validate_remote_image_url(url, allowed_hosts=allowed)

    owned_client = client is None
    if timeout is None:
        timeout = httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=3.0)

    if owned_client:
        client = httpx.Client(timeout=timeout, follow_redirects=False)

    assert client is not None

    try:
        redirects = 0
        while True:
            with client.stream("GET", current_url, headers={"Accept": "image/*"}) as resp:
                if resp.status_code in _REDIRECT_STATUSES:
                    if redirects >= max_redirects:
                        raise RemoteImageError(status_code=400, detail="Too many redirects while fetching image.")
                    location = resp.headers.get("location")
                    if not location:
                        raise RemoteImageError(status_code=400, detail="Redirect response missing Location header.")
                    next_url = urljoin(current_url, location)
                    current_url = validate_remote_image_url(next_url, allowed_hosts=allowed)
                    redirects += 1
                    continue

                if resp.status_code != 200:
                    raise RemoteImageError(
                        status_code=400,
                        detail=f"Image fetch failed with status {resp.status_code}.",
                    )

                content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                if not content_type.startswith("image/"):
                    raise RemoteImageError(status_code=415, detail="Fetched content is not an image.")

                content_length = resp.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            raise RemoteImageError(status_code=413, detail="Image is too large.")
                    except ValueError:
                        pass

                chunks = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise RemoteImageError(status_code=413, detail="Image is too large.")
                    chunks.append(chunk)
                return b"".join(chunks), content_type
    finally:
        if owned_client:
            client.close()


def fetch_image_bytes(
    url: str,
    *,
    allowed_hosts: Optional[Iterable[str]] = None,
    max_bytes: int = 10 * 1024 * 1024,
    max_redirects: int = 3,
    timeout: Optional[httpx.Timeout] = None,
    client: Optional[httpx.Client] = None,
) -> bytes:
    """
    Fetch a remote image (bytes) with SSRF protections.

    This function only fetches from allowlisted hosts (see validate_remote_image_url),
    enforces timeouts, enforces a max download size, and validates Content-Type.
    """
    raw, _content_type = _fetch_image(
        url,
        allowed_hosts=allowed_hosts,
        max_bytes=max_bytes,
        max_redirects=max_redirects,
        timeout=timeout,
        client=client,
    )
    return raw


def fetch_image_data_uri(
    url: str,
    *,
    allowed_hosts: Optional[Iterable[str]] = None,
    max_bytes: int = 10 * 1024 * 1024,
    max_redirects: int = 3,
    timeout: Optional[httpx.Timeout] = None,
    client: Optional[httpx.Client] = None,
) -> str:
    """
    Fetch a remote image and return it as a `data:` URI.

    This avoids requiring Pillow in environments where only `backend/requirements.txt`
    is installed (DSPy only needs Pillow when *it* has to decode raw bytes).
    """
    raw, content_type = _fetch_image(
        url,
        allowed_hosts=allowed_hosts,
        max_bytes=max_bytes,
        max_redirects=max_redirects,
        timeout=timeout,
        client=client,
    )
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{content_type};base64,{b64}"
