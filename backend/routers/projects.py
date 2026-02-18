"""Project management API routes."""
from fastapi import APIRouter, HTTPException

from backend.models.schemas import Project, ProjectCreate
from backend.services.project_service import ProjectService
from backend.config import settings

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("")
async def list_projects():
    """List all available projects."""
    return ProjectService.list_projects()


@router.post("")
async def create_project(data: ProjectCreate):
    """Create a new project."""
    try:
        return ProjectService.create_project(data.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{name}/activate")
async def switch_project(name: str):
    """Switch the active project."""
    try:
        from backend.services.label_service import LabelService
        result = ProjectService.switch_project(name)
        LabelService.reset()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/{name}")
async def delete_project(name: str):
    """Delete a project."""
    try:
        return ProjectService.delete_project(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/active")
async def get_active_project():
    """Get the currently active project name."""
    return {"active_project": settings.active_project}
