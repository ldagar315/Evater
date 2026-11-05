"""Single validated settings object for the Evater backend.

Modules read configuration through get_settings() instead of calling
os.getenv / load_dotenv directly. get_settings() returns a fresh Settings on
each call on purpose: pydantic validates on construction, and tests can
monkeypatch environment variables without clearing a cache. app.main logs
missing critical variables at startup so misconfiguration surfaces there
instead of as a mid-request failure.
"""
from typing import List

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables.

    Local-dev server knobs (PORT, FASTAPI_RELOAD) deliberately stay out of
    Settings: they are only read by app.main's __main__ block with the
    historical semantics (int(PORT), FASTAPI_RELOAD=="true"), and a malformed
    value must not fail every get_settings() call.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Supabase
    supabase_url: str = ""
    supabase_api_key: str = ""
    supabase_service_role_key: str = ""

    # LLM providers
    groq_api_key: str = ""
    gemini_api_key: str = ""
    cerebras_api_key: str = ""

    # CORS / environment
    app_origins: str = ""
    env: str = Field(default="", validation_alias=AliasChoices("ENV", "APP_ENV", "NODE_ENV"))

    # OCR image fetching
    ocr_allowed_image_hosts: str = ""

# Settings the app cannot serve requests without. Checked and logged at
# startup; request-time code still raises if they are missing.
CRITICAL_SETTINGS = ("supabase_url", "supabase_api_key")


def get_settings() -> Settings:
    """Build and validate a fresh Settings from the current environment."""
    return Settings()


def missing_critical_settings(settings: Settings) -> List[str]:
    return [name for name in CRITICAL_SETTINGS if not getattr(settings, name)]
