import os
import time
from typing import Any, Callable, Optional

import httpx

from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions


def create_supabase_client(jwt: Optional[str] = None) -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")

    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_API_KEY")

    if jwt:
        options = SyncClientOptions(headers={"Authorization": f"Bearer {jwt}", "apikey": supabase_key})
        return create_client(supabase_url, supabase_key, options=options)

    return create_client(supabase_url, supabase_key)


def create_supabase_service_client() -> Client:
    """Create the trusted backend client used for answer-key reads.

    The service-role key must never be sent to the browser. Keeping this
    separate from create_supabase_client prevents an anon/publishable key from
    being silently promoted to a privileged key.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not service_role_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

    return create_client(supabase_url, service_role_key)


# Transient transport failures worth one transparent retry: the Supabase edge
# (or the pooler in front of it) occasionally severs a live HTTP/2 connection
# mid-request ("Server disconnected"), which otherwise surfaces as a 500 even
# though the very next attempt succeeds.
TRANSIENT_TRANSPORT_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
)


def run_with_retry(operation: Callable[[], Any], *, attempts: int = 3, base_delay: float = 0.3) -> Any:
    """Run a read-only Supabase operation, retrying transient transport drops.

    Only wrap operations that are safe to repeat (reads and idempotent
    upserts); never wrap one-shot inserts.
    """
    for attempt in range(attempts):
        try:
            return operation()
        except TRANSIENT_TRANSPORT_ERRORS:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
