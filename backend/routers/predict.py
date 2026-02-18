"""Auto-prediction / Active Learning API routes."""
from fastapi import APIRouter

from backend.services.active_learning import ActiveLearningService

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/unlabeled")
async def predict_unlabeled(split: str = "train", confidence: float | None = None, model_path: str = None):
    """Run predictions on all unlabeled images."""
    results = ActiveLearningService.predict_unlabeled(split, confidence, model_path)
    return {
        "predictions": results,
        "count": len(results),
    }


@router.get("/queue")
async def get_smart_queue(split: str = "train"):
    """Get images sorted by active learning priority."""
    return ActiveLearningService.get_smart_queue(split)


@router.get("/cached/{image_name}")
async def get_cached_predictions(image_name: str):
    """Get cached predictions for an image."""
    preds = ActiveLearningService.get_predictions(image_name)
    return {"image_name": image_name, "boxes": preds}


@router.delete("/cached")
async def clear_predictions():
    """Clear cached predictions."""
    return ActiveLearningService.clear_predictions()


@router.get("/stats")
async def get_stats():
    """Get active learning statistics."""
    return ActiveLearningService.get_stats()


@router.get("/should-train")
async def should_train():
    """Check if enough labels to trigger auto-training."""
    return ActiveLearningService.should_auto_train()
