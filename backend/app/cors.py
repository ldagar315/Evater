import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def _split_origins(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [o.strip().rstrip("/") for o in value.split(",") if o.strip()]


def is_production_env() -> bool:
    # Common environment markers across platforms
    env = (os.getenv("ENV") or os.getenv("APP_ENV") or os.getenv("NODE_ENV") or "").lower()
    # Default to production-safe behavior if unset/unknown to avoid accidentally allowing localhost in prod.
    return env not in {"dev", "development", "local", "test"}


def allowed_origins() -> List[str]:
    # Comma-separated list of allowed origins, e.g.:
    # APP_ORIGINS="https://evater.com,https://www.evater.com,https://staging.evater.com"
    configured = _split_origins(os.getenv("APP_ORIGINS"))
    if configured:
        return configured
    if is_production_env():
        # Keep the deployed Evater frontend usable even when the existing
        # Modal secret has not yet gained APP_ORIGINS. Operators can still
        # override this list explicitly for another deployment domain.
        return ["https://evater.xyz", "https://www.evater.xyz"]
    return []


def allowed_origin_regex() -> Optional[str]:
    # Safe local-dev default without opening production. In prod, require explicit origins.
    if is_production_env():
        return None
    return r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def add_cors_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_origin_regex=allowed_origin_regex(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        # Include common headers used by Supabase clients as well.
        allow_headers=["Authorization", "Content-Type", "apikey", "x-client-info"],
    )
