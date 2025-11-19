import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .routers import test_router, feedback_router, viva_router

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title = "Evater_v1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the Evater API"}

# Include Routers
app.include_router(test_router.router, prefix="/api", tags=["Test Generation"])
app.include_router(feedback_router.router, prefix="/api", tags=["Feedback Generation"])
app.include_router(viva_router.router, tags=["Viva WebSocket"])

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=os.environ.get("FASTAPI_RELOAD", "false").lower() == "true")
