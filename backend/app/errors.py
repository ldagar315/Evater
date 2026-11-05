"""Central exception types and handlers.

Routers raise AppError subclasses (or let known exceptions bubble) instead of
building error responses inline; the handlers here produce consistent error
JSON and log with the request ID.
"""
import logging
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .remote_image import RemoteImageError

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base application error with an HTTP mapping."""

    status_code = 500
    detail = "Internal server error."

    def __init__(self, detail: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(detail or self.detail)
        if detail is not None:
            self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class LLMServiceError(AppError):
    """A backing AI service failed to produce a usable response."""

    status_code = 502
    detail = "The AI service failed to generate a response. Please try again."


class ConfigurationError(AppError):
    """Required configuration is missing or invalid."""

    status_code = 500
    detail = "Server configuration error."


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to each request and echo it in a response header."""

    async def dispatch(self, request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("request %s failed: %s", request_id_of(request), exc)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RemoteImageError)
    async def remote_image_error_handler(request: Request, exc: RemoteImageError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        # RuntimeErrors reaching the server are configuration-style failures
        # (e.g. missing Supabase env vars). Return a clean 500, log the detail.
        logger.error("request %s configuration failure: %s", request_id_of(request), exc)
        return JSONResponse(status_code=500, content={"detail": "Server configuration error."})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("request %s unhandled error", request_id_of(request))
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
