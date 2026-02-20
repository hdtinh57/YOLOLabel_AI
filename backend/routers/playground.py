"""Playground API router for running inferences."""
import asyncio
from pathlib import Path
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from backend.config import settings
from backend.services.playground_service import PlaygroundService

router = APIRouter(prefix="/api/playground", tags=["playground"])

@router.post("/predict")
async def predict_playground(
    file: UploadFile = File(...),
    model_version_id: int = Form(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    max_det: int = Form(300),
    agnostic_nms: bool = Form(False)
):
    """Run model inference on uploaded file."""
    
    # 1. Save uploaded file to temp input directory
    input_dir = settings.data_dir / "playground_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = input_dir / file.filename
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    # 2. Run inference in a background thread to avoid blocking main thread
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            PlaygroundService.run_inference,
            file_path,
            model_version_id,
            confidence,
            iou,
            imgsz,
            max_det,
            agnostic_nms
        )
        return result
    except Exception as e:
        # Cleanup on failure
        if file_path.exists():
             file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
