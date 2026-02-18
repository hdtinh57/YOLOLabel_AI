"""Settings API routes — reads from/writes to SQLite-backed config dicts."""
from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import settings
from backend.services.model_service import ModelService

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    """Partial update for project settings."""

    # Active Learning
    auto_train_threshold: int | None = None
    prediction_confidence: float | None = None

    # Model
    base_model: str | None = None

    # Basic Training
    epochs: int | None = None
    batch: int | None = None
    imgsz: int | None = None
    patience: int | None = None

    # Advanced Training
    optimizer: str | None = None
    device: str | None = None
    lr0: float | None = None
    lrf: float | None = None
    momentum: float | None = None
    weight_decay: float | None = None
    warmup_epochs: float | None = None
    warmup_momentum: float | None = None
    box: float | None = None
    cls: float | None = None
    dfl: float | None = None

    # Augmentation
    fliplr: float | None = None
    mosaic: float | None = None
    mixup: float | None = None
    copy_paste: float | None = None


@router.get("")
async def get_settings():
    """Get current application settings."""
    tc = settings.training_config
    ac = settings.augment_config
    al = settings.al_config

    return {
        # AL
        "auto_train_threshold": al.get("auto_train_threshold", 10),
        "prediction_confidence": al.get("prediction_confidence", 0.25),

        # Model
        "base_model": tc.get("model", "yolo26n.pt"),
        "available_models": settings.available_models + _get_custom_models(),
        "active_project": settings.active_project,

        # Basic Training
        "epochs": tc.get("epochs", 50),
        "batch": tc.get("batch", 16),
        "imgsz": tc.get("imgsz", 640),
        "patience": tc.get("patience", 10),

        # Advanced
        "optimizer": tc.get("optimizer", "auto"),
        "device": tc.get("device", "0"),
        "lr0": tc.get("lr0", 0.01),
        "lrf": tc.get("lrf", 0.01),
        "momentum": tc.get("momentum", 0.937),
        "weight_decay": tc.get("weight_decay", 0.0005),
        "warmup_epochs": tc.get("warmup_epochs", 3.0),
        "warmup_momentum": tc.get("warmup_momentum", 0.8),
        "box": tc.get("box", 7.5),
        "cls": tc.get("cls", 0.5),
        "dfl": tc.get("dfl", 1.5),

        # Augmentation
        "fliplr": ac.get("fliplr", 0.5),
        "mosaic": ac.get("mosaic", 1.0),
        "mixup": ac.get("mixup", 0.0),
        "copy_paste": ac.get("copy_paste", 0.0),
    }


@router.post("")
async def update_settings(data: SettingsUpdate):
    """Update application settings."""
    updates = data.model_dump(exclude_none=True)

    # Categorize updates into training / augment / al config
    training_keys = {
        "epochs", "batch", "imgsz", "patience", "optimizer", "device",
        "lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "box", "cls", "dfl",
    }
    augment_keys = {"fliplr", "mosaic", "mixup", "copy_paste"}
    al_keys = {"auto_train_threshold", "prediction_confidence"}

    for key, value in updates.items():
        if key == "base_model":
            settings.training_config["model"] = value
        elif key in training_keys:
            settings.training_config[key] = value
        elif key in augment_keys:
            settings.augment_config[key] = value
        elif key in al_keys:
            settings.al_config[key] = value

    settings.save(include_global=False)

    return {"status": "updated", "settings": await get_settings()}


def _get_custom_models() -> list[str]:
    """List custom model names from runs directory."""
    custom = []
    if settings.runs_dir.exists():
        for pt_file in settings.runs_dir.rglob("best.pt"):
            run_name = pt_file.parent.parent.name
            custom.append(f"{run_name}/best.pt")
    return custom
