"""MLOps API router — dashboard endpoints for training history, model registry, and stats."""
from fastapi import APIRouter, HTTPException, Query

from backend.services.training_history_service import TrainingHistoryService
from backend.services.model_registry_service import ModelRegistryService
from backend.services.active_learning import ActiveLearningService
from backend.config import settings

router = APIRouter(prefix="/api/mlops", tags=["mlops"])


# ──────────────────────────────────────────────
#  Dashboard Stats
# ──────────────────────────────────────────────

@router.get("/dashboard")
def get_dashboard():
    """Get dashboard KPIs for current project."""
    stats = TrainingHistoryService.get_dashboard_stats(settings.active_project)
    al_stats = ActiveLearningService.get_stats()
    return {
        "project": settings.active_project,
        "training": stats,
        "active_learning": al_stats,
    }


# ──────────────────────────────────────────────
#  Training Runs
# ──────────────────────────────────────────────

@router.get("/runs")
def list_runs(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="started_at"),
    order: str = Query(default="DESC"),
):
    """List training runs with pagination."""
    return TrainingHistoryService.get_runs(
        settings.active_project, limit, offset, sort_by, order
    )


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    """Get full details of a training run including epoch metrics."""
    detail = TrainingHistoryService.get_run_detail(run_id)
    if not detail:
        raise HTTPException(404, f"Run {run_id} not found")
    return detail


@router.post("/runs/compare")
def compare_runs(body: dict):
    """Compare multiple training runs side by side."""
    run_ids = body.get("run_ids", [])
    if len(run_ids) < 2:
        raise HTTPException(400, "Need at least 2 runs to compare")
    if len(run_ids) > 5:
        raise HTTPException(400, "Maximum 5 runs for comparison")
    return TrainingHistoryService.compare_runs(run_ids)


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, delete_files: bool = Query(default=False)):
    """Delete a training run record."""
    TrainingHistoryService.delete_run(run_id, delete_files)
    return {"deleted": run_id}


# ──────────────────────────────────────────────
#  Model Registry
# ──────────────────────────────────────────────

@router.get("/models")
def list_model_versions():
    """List all model versions for current project."""
    return ModelRegistryService.get_versions(settings.active_project)


@router.get("/models/production")
def get_production_model():
    """Get the current production model."""
    model = ModelRegistryService.get_production_model(settings.active_project)
    if not model:
        return {"production": None, "message": "No production model set"}
    return model


@router.post("/models/promote/{run_id}")
def promote_run(run_id: str, notes: str = ""):
    """Promote a training run to a new model version."""
    try:
        return ModelRegistryService.promote_run(
            run_id, settings.active_project, notes
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.put("/models/{version_id}/stage")
def set_model_stage(version_id: int, stage: str = Query(...)):
    """Transition a model version's stage."""
    valid = {"none", "staging", "production", "archived"}
    if stage not in valid:
        raise HTTPException(400, f"Invalid stage. Must be one of: {valid}")
    try:
        return ModelRegistryService.transition_stage(version_id, stage)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ──────────────────────────────────────────────
#  Active Learning Cycles
# ──────────────────────────────────────────────

@router.get("/al/cycles")
def list_al_cycles(limit: int = Query(default=20, le=100)):
    """List active learning cycle history."""
    return ActiveLearningService.get_cycle_history(limit)


@router.post("/al/cycles/start")
def start_al_cycle():
    """Start a new active learning cycle."""
    return ActiveLearningService.start_cycle()


@router.post("/al/cycles/{cycle_id}/complete")
def complete_al_cycle(cycle_id: int, run_id: str | None = None):
    """Mark an AL cycle as completed."""
    ActiveLearningService.complete_cycle(cycle_id, run_id)
    return {"completed": cycle_id}


@router.post("/al/predictions/{image_name}/accept")
def accept_prediction(image_name: str):
    """Accept an auto-prediction for an image."""
    return {"accepted": ActiveLearningService.accept_prediction(image_name)}


@router.post("/al/predictions/{image_name}/reject")
def reject_prediction(image_name: str):
    """Reject an auto-prediction for an image."""
    return {"rejected": ActiveLearningService.reject_prediction(image_name)}


# ──────────────────────────────────────────────
#  Project Config
# ──────────────────────────────────────────────

@router.get("/config")
def get_project_config():
    """Get per-project training/augment/AL config."""
    return {
        "training": settings.training_config,
        "augmentation": settings.augment_config,
        "active_learning": settings.al_config,
    }


@router.put("/config")
def update_project_config(config: dict):
    """Update per-project config (partial update)."""
    if "training" in config:
        settings.training_config.update(config["training"])
    if "augmentation" in config:
        settings.augment_config.update(config["augmentation"])
    if "active_learning" in config:
        settings.al_config.update(config["active_learning"])
    settings.save(include_global=False, include_project=True)
    return {"updated": True}


# ──────────────────────────────────────────────
#  MLflow Integration (Optional)
# ──────────────────────────────────────────────

@router.get("/mlflow/status")
def mlflow_status():
    """Get MLflow integration status."""
    from backend.services.mlflow_integration import MLflowIntegration
    return MLflowIntegration.get_status()


@router.post("/mlflow/enable")
def mlflow_enable(tracking_uri: str = "http://localhost:5000", experiment: str = "yololabel-ai"):
    """Enable MLflow tracking."""
    from backend.services.mlflow_integration import MLflowIntegration
    return MLflowIntegration.enable(tracking_uri, experiment)


@router.post("/mlflow/disable")
def mlflow_disable():
    """Disable MLflow tracking."""
    from backend.services.mlflow_integration import MLflowIntegration
    return MLflowIntegration.disable()

