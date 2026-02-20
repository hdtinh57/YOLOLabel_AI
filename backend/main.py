"""YOLOLabel AI — FastAPI application entry point."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.routers import images, labels, model, predict, projects, settings as settings_router, mlops, playground
from backend.services.project_service import ProjectService

# Create FastAPI app
app = FastAPI(
    title="YOLOLabel AI",
    description="Bounding box labeling tool with Active Learning for YOLO training",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(projects.router)
app.include_router(settings_router.router)
app.include_router(images.router)
app.include_router(labels.router)
app.include_router(model.router)
app.include_router(predict.router)
app.include_router(mlops.router)
app.include_router(playground.router)


@app.on_event("startup")
async def startup():
    """Initialize database, migrate legacy data, and restore project state."""
    from backend.database import init_db
    from backend.migrate import migrate_legacy

    # Step 1: Initialize SQLite database (creates tables if needed)
    init_db(settings.data_dir)

    # Step 2: Migrate legacy JSON/CSV data → SQLite (runs once)
    migrate_legacy(settings.data_dir, settings.project_root)

    # Step 3: Load project state from SQLite
    ProjectService.load_state()

    # Step 4: Auto-load production model (or fallback to pretrained)
    from backend.services.model_service import ModelService
    model_info = ModelService.auto_load_production_model()
    print(f"[MLOps] Model loaded: {model_info['model_name']} (source: {model_info['source']})")


# Mount additional static directories
playground_results_dir = settings.data_dir / "playground_results"
playground_results_dir.mkdir(parents=True, exist_ok=True)
app.mount("/playground-results", StaticFiles(directory=str(playground_results_dir)), name="playground_results")

# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
