import os
from modal import Image, App, Secret, asgi_app

# Define the Modal Image with necessary dependencies
# We explicitly include the local 'app' package using add_local_python_source
image = Image.debian_slim(python_version="3.12").pip_install(
    "fastapi",
    "uvicorn",
    "groq",
    "dspy",
    "python-dotenv", 
    "pydantic", 
    "supabase", 
    "google-genai",
    "httpx-aiohttp>=0.1.5,<0.2",
).add_local_python_source("app")

app = App("Evater_v1", image=image)

@app.function(
    secrets=[
        Secret.from_name("groq-secret"),
        Secret.from_name("evater-supabase-config"),
    ],
    min_containers=1
)
@asgi_app()
def wrapper():
    from app.main import app as web_app

    return web_app

if __name__ == "__main__":
    # This allows running locally via `python application.py` if needed, 
    # though `uvicorn app.main:app` is preferred for local dev.
    import uvicorn
    from app.main import app as web_app

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(web_app, host="0.0.0.0", port=port)
