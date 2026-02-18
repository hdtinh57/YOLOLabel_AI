"""Model management and training API routes."""
from fastapi import APIRouter

from backend.config import settings
from backend.models.schemas import TrainingConfig, ActiveLearningSettings
from backend.services.model_service import ModelService

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/info")
async def model_info():
    """Get current model information."""
    return ModelService.get_model_info()


@router.get("/available")
async def available_models():
    """List all available models (pretrained + custom)."""
    return ModelService.get_available_models()


@router.post("/load")
async def load_model(model_name: str | None = None, custom_path: str | None = None):
    """Load a YOLO model."""
    return ModelService.load_model(model_name, custom_path)


@router.post("/train")
async def train_model(config: TrainingConfig):
    """Start training a new model."""
    cfg = config.model_dump()

    # Persist training params to project config
    training_keys = {
        "epochs", "imgsz", "batch", "patience", "optimizer", "device",
        "lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "box", "cls", "dfl",
    }
    augment_keys = {"fliplr", "mosaic", "mixup", "copy_paste"}

    for k, v in cfg.items():
        if k == "model_name":
            settings.training_config["model"] = v
        elif k in training_keys:
            settings.training_config[k] = v
        elif k in augment_keys:
            settings.augment_config[k] = v

    settings.save(include_global=False)

    # Start training with all config as kwargs
    result = ModelService.start_training(**cfg)
    return result


@router.post("/stop-training")
async def stop_training():
    """Stop the current training session."""
    return ModelService.stop_training()


@router.get("/training-status")
async def training_status():
    """Get current training progress."""
    return ModelService.get_training_status()


@router.post("/settings")
async def update_settings(al_settings: ActiveLearningSettings):
    """Update active learning settings."""
    # AL config
    settings.al_config["auto_train_threshold"] = al_settings.auto_train_threshold
    settings.al_config["prediction_confidence"] = al_settings.prediction_confidence

    # Training config
    settings.training_config["model"] = al_settings.base_model
    settings.training_config["optimizer"] = al_settings.optimizer
    settings.training_config["device"] = al_settings.device
    settings.training_config["lr0"] = al_settings.lr0
    settings.training_config["lrf"] = al_settings.lrf
    settings.training_config["momentum"] = al_settings.momentum
    settings.training_config["weight_decay"] = al_settings.weight_decay
    settings.training_config["warmup_epochs"] = al_settings.warmup_epochs
    settings.training_config["warmup_momentum"] = al_settings.warmup_momentum
    settings.training_config["box"] = al_settings.box
    settings.training_config["cls"] = al_settings.cls
    settings.training_config["dfl"] = al_settings.dfl

    # Augmentation
    settings.augment_config["fliplr"] = al_settings.fliplr
    settings.augment_config["mosaic"] = al_settings.mosaic
    settings.augment_config["mixup"] = al_settings.mixup
    settings.augment_config["copy_paste"] = al_settings.copy_paste

    settings.save(include_global=False)

    return {"updated": True}


@router.get("/settings")
async def get_settings():
    """Get current active learning settings."""
    tc = settings.training_config
    ac = settings.augment_config
    al = settings.al_config

    return {
        "auto_train_threshold": al.get("auto_train_threshold", 10),
        "prediction_confidence": al.get("prediction_confidence", 0.25),
        "base_model": tc.get("model", "yolo26n.pt"),
        "available_models": settings.available_models,

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

        "fliplr": ac.get("fliplr", 0.5),
        "mosaic": ac.get("mosaic", 1.0),
        "mixup": ac.get("mixup", 0.0),
        "copy_paste": ac.get("copy_paste", 0.0),
    }
