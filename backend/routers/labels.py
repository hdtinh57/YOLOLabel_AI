"""Label / annotation API routes."""
from fastapi import APIRouter

from backend.models.schemas import SaveAnnotationRequest, ClassInfo
from backend.services.label_service import LabelService

router = APIRouter(prefix="/api/labels", tags=["labels"])


# --- Class management (MUST be before dynamic /{split}/{image_name} routes) ---

@router.get("/classes/list")
async def get_classes():
    """Get all class definitions."""
    return LabelService.get_classes()


@router.post("/classes/add")
async def add_class(name: str):
    """Add a new class."""
    cls = LabelService.add_class(name)
    return cls.model_dump()


@router.delete("/classes/{class_id}")
async def remove_class(class_id: int):
    """Remove a class by ID."""
    LabelService.remove_class(class_id)
    return {"deleted": True}


@router.put("/classes/{class_id}")
async def rename_class(class_id: int, new_name: str):
    """Rename a class."""
    LabelService.rename_class(class_id, new_name)
    return {"renamed": True}


@router.post("/classes/set")
async def set_classes(classes: list[ClassInfo]):
    """Set the full class list."""
    LabelService.set_classes(classes)
    return {"saved": True, "count": len(classes)}


# --- Export ---

@router.get("/export")
async def export_dataset():
    """Export dataset info and generate dataset.yaml."""
    return LabelService.export_dataset()


# --- Label CRUD (dynamic routes LAST to avoid catching /classes/*, /export, etc.) ---

@router.get("/{split}/{image_name}")
async def get_labels(split: str, image_name: str):
    """Get annotations for an image."""
    boxes = LabelService.read_labels(image_name, split)
    return {"image_name": image_name, "boxes": [b.model_dump() for b in boxes]}


@router.post("/save")
async def save_labels(req: SaveAnnotationRequest):
    """Save annotations for an image."""
    LabelService.save_labels(req.image_name, req.boxes, req.split)
    return {"saved": True, "image_name": req.image_name, "count": len(req.boxes)}


@router.delete("/{split}/{image_name}")
async def delete_labels(split: str, image_name: str):
    """Delete all annotations for an image."""
    LabelService.delete_labels(image_name, split)
    return {"deleted": True, "image_name": image_name}
