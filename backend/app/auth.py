from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, WebSocket
from starlette.websockets import WebSocketDisconnect
from supabase_auth.types import User

from .supabase_client import create_supabase_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthContext:
    jwt: str
    user: User
    tenant_id: Optional[str]


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token


def _tenant_id_from_user(user: User) -> Optional[str]:
    app_md = user.app_metadata or {}
    user_md = user.user_metadata or {}

    return (
        app_md.get("tenant_id")
        or user_md.get("tenant_id")
        or app_md.get("org_id")
        or user_md.get("org_id")
    )


def require_user(authorization: Optional[str] = Header(default=None)) -> AuthContext:
    jwt = _extract_bearer_token(authorization)

    try:
        supabase = create_supabase_client()
        user_response = supabase.auth.get_user(jwt)
    except RuntimeError as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="Server misconfigured")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to validate Supabase JWT")
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    tenant_id = _tenant_id_from_user(user_response.user)
    return AuthContext(jwt=jwt, user=user_response.user, tenant_id=tenant_id)


async def require_user_websocket(websocket: WebSocket) -> AuthContext:
    token = (
        websocket.query_params.get("access_token")
        or websocket.query_params.get("token")
        or websocket.query_params.get("jwt")
    )

    if not token:
        auth_header = websocket.headers.get("authorization")
        if auth_header:
            scheme, _, maybe_token = auth_header.partition(" ")
            if scheme.lower() == "bearer" and maybe_token:
                token = maybe_token

    if not token:
        await websocket.close(code=1008, reason="Unauthorized")
        raise WebSocketDisconnect(code=1008)

    try:
        supabase = create_supabase_client()
        user_response = supabase.auth.get_user(token)
    except RuntimeError as e:
        logger.error(str(e))
        await websocket.close(code=1011, reason="Server misconfigured")
        raise WebSocketDisconnect(code=1011)
    except Exception:
        logger.exception("Failed to validate Supabase JWT for websocket")
        await websocket.close(code=1008, reason="Unauthorized")
        raise WebSocketDisconnect(code=1008)

    if not user_response or not user_response.user:
        await websocket.close(code=1008, reason="Unauthorized")
        raise WebSocketDisconnect(code=1008)

    tenant_id = _tenant_id_from_user(user_response.user)
    return AuthContext(jwt=token, user=user_response.user, tenant_id=tenant_id)
