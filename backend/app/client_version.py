"""Client-version gate for endpoints whose responses are only safe for
key-free-aware clients.

The answer-key integrity change makes /gen_question return SANITIZED
questions (no is_correct) plus an opaque key_id. A pre-change client would
store that sanitized payload as a bare array, and its TakeTestPage would then
silently mark the FIRST option correct (its legacy fallback) - wrong grading
shown as right. Rejecting such clients here (before any generation spend or
key write) is the deterministic stop: stale clients get a visible error and
can never create a keyless bare-array row.
"""
from typing import Optional

from fastapi import HTTPException

# Header sent by key-free-aware web clients, e.g. "web/2".
CLIENT_HEADER = "x-evater-client"
_MIN_MAJOR = 2


def client_supports_server_keys(value: Optional[str]) -> bool:
    if not value:
        return False
    family, _, ver = value.partition("/")
    if family.strip().lower() != "web":
        return False
    try:
        return int(ver.strip()) >= _MIN_MAJOR
    except ValueError:
        return False


def require_supported_client(value: Optional[str]) -> None:
    if not client_supports_server_keys(value):
        raise HTTPException(
            status_code=409,
            detail="client_outdated: this app version is out of date. Please reload the page and try again.",
        )
