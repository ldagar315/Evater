import os
import logging
from fastapi import FastAPI
from dotenv import load_dotenv
from .routers import auth_router, test_router, feedback_router, viva_router
from .cors import add_cors_middleware

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(title="Evater_v1")

    add_cors_middleware(app)

    @app.get("/")
    def root():
        return {"message": "Welcome to the Evater API"}

    # Include Routers
    app.include_router(test_router.router, prefix="/api", tags=["Test Generation"])
    app.include_router(feedback_router.router, prefix="/api", tags=["Feedback Generation"])
    app.include_router(auth_router.router, prefix="/api", tags=["Auth"])
    app.include_router(viva_router.router, tags=["Viva WebSocket"])
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=os.environ.get("FASTAPI_RELOAD", "false").lower() == "true")
