"""Image management API routes."""
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from backend.config import settings
from backend.services.image_service import ImageService

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("")
async def list_images(split: str = "train"):
    """List all images with metadata."""
    return ImageService.get_images(split)


@router.get("/{split}/{image_name}")
async def get_image(split: str, image_name: str):
    """Serve an image file."""
    path = ImageService.get_image_path(image_name, split)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@router.get("/thumbnail/{split}/{image_name}")
async def get_thumbnail(split: str, image_name: str):
    """Get thumbnail for an image."""
    thumb = ImageService.generate_thumbnail(image_name, split)
    if not thumb:
        # Fallback to full image
        path = ImageService.get_image_path(image_name, split)
        if not path:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(path)
    return FileResponse(thumb)


@router.post("/upload")
async def upload_images(split: str = "train", files: list[UploadFile] = File(...)):
    """Upload images to the dataset."""
    file_data = []
    for f in files:
        data = await f.read()
        file_data.append((data, f.filename))

    saved = ImageService.upload_images(file_data, split)
    return {"uploaded": saved, "count": len(saved)}


@router.get("/dimensions/{split}/{image_name}")
async def get_dimensions(split: str, image_name: str):
    """Get image dimensions."""
    w, h = ImageService.get_image_dimensions(image_name, split)
    if w == 0 and h == 0:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"width": w, "height": h}
