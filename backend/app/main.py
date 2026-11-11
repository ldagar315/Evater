import logging
import os

from fastapi import FastAPI

from .routers import test_router, feedback_router, viva_router, question_bank_router, leaderboard_router, sample_papers_router
from .cors import add_cors_middleware
from .config import get_settings, missing_critical_settings
from .errors import RequestIDMiddleware, register_error_handlers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Evater_v1")

    missing = missing_critical_settings(get_settings())
    if missing:
        logger.error(
            "Missing critical configuration: %s - requests that depend on it will fail",
            ", ".join(missing),
        )

    add_cors_middleware(app)
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    @app.get("/")
    def root():
        return {"message": "Welcome to the Evater API"}

    # Include Routers
    app.include_router(test_router.router, prefix="/api", tags=["Test Generation"])
    app.include_router(feedback_router.router, prefix="/api", tags=["Feedback Generation"])
    app.include_router(question_bank_router.router, prefix="/api/v1", tags=["Question Bank"])
    app.include_router(leaderboard_router.router, prefix="/api/v1", tags=["Leaderboard"])
    app.include_router(viva_router.router, tags=["Viva WebSocket"])
    # Public (unauthenticated) read-only routes for the SEO sample-paper pages.
    app.include_router(sample_papers_router.router, prefix="/api/v1", tags=["Sample Papers"])
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=os.environ.get("FASTAPI_RELOAD", "false").lower() == "true")
