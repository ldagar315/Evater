import os
from typing import Optional

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
