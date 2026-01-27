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
        options = SyncClientOptions(headers={"Authorization": f"Bearer {jwt}"})
        return create_client(supabase_url, supabase_key, options=options)

    return create_client(supabase_url, supabase_key)

