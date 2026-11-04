from modal import Image, App, Secret, asgi_app

# Define the Modal Image with necessary dependencies.
# requirements.txt is the single source of truth for backend dependencies.
# We explicitly include the local 'app' package using add_local_python_source
image = Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    "requirements.txt"
).add_local_python_source("app")

app = App("Evater_v1", image=image)

@app.function(
    secrets=[
        Secret.from_name("groq-secret"),
        Secret.from_name("evater-supabase-config"),
    ],
    # Cost rule (Oct 21): no always-on containers on the $30/month Modal plan.
    # Sleep after 10 min idle; first request after that takes a cold start,
    # which is acceptable while the app has no traffic.
    min_containers=0,
    scaledown_window=600,
)
@asgi_app()
def wrapper():
    from app.main import app as web_app

    return web_app
