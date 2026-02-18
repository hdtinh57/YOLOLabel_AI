"""Project service — CRUD for labeling projects (SQLite-backed)."""
import json
import time
from pathlib import Path

from backend.config import (
    settings,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_AUGMENT_CONFIG,
    DEFAULT_AL_CONFIG,
)
from backend.database import get_connection, set_setting, rows_to_list


class ProjectService:
    """Manages labeling projects."""

    @classmethod
    def list_projects(cls) -> list[dict]:
        """List all projects with stats."""
        projects = []
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT name, created_at FROM projects ORDER BY created_at DESC"
            ).fetchall()

        for row in rows:
            name = row["name"]
            project_path = settings.projects_dir / name

            # Count images and labels
            train_images = 0
            val_images = 0
            labeled = 0

            img_train = project_path / "images" / "train"
            img_val = project_path / "images" / "val"
            lbl_train = project_path / "labels" / "train"

            if img_train.exists():
                train_images = sum(
                    1 for f in img_train.iterdir()
                    if f.suffix.lower() in settings.image_extensions
                )
            if img_val.exists():
                val_images = sum(
                    1 for f in img_val.iterdir()
                    if f.suffix.lower() in settings.image_extensions
                )
            if lbl_train.exists():
                labeled = sum(
                    1 for f in lbl_train.iterdir()
                    if f.suffix == ".txt" and f.stat().st_size > 0
                )

            # Count training runs
            run_count = 0
            with get_connection() as conn:
                r = conn.execute(
                    "SELECT COUNT(*) as cnt FROM training_runs WHERE project = ?",
                    (name,),
                ).fetchone()
                run_count = r["cnt"] if r else 0

            projects.append({
                "name": name,
                "created_at": row["created_at"],
                "total_images": train_images + val_images,
                "train_images": train_images,
                "val_images": val_images,
                "labeled_count": labeled,
                "training_runs": run_count,
                "is_active": name == settings.active_project,
            })

        return projects

    @classmethod
    def create_project(cls, name: str) -> dict:
        """Create a new project with dirs + DB record."""
        # Validate name
        if not name or not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError("Project name can only contain letters, numbers, hyphens, underscores")

        project_path = settings.projects_dir / name
        if project_path.exists():
            raise ValueError(f"Project '{name}' already exists")

        # Create directories
        for d in [
            project_path / "images" / "train",
            project_path / "images" / "val",
            project_path / "labels" / "train",
            project_path / "labels" / "val",
            project_path / "runs",
            project_path / "thumbnails",
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Insert into DB
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO projects
                   (name, created_at, training_config, augment_config, al_config)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    name,
                    time.time(),
                    json.dumps(DEFAULT_TRAINING_CONFIG),
                    json.dumps(DEFAULT_AUGMENT_CONFIG),
                    json.dumps(DEFAULT_AL_CONFIG),
                ),
            )
            conn.commit()

        return {"name": name, "created": True}

    @classmethod
    def switch_project(cls, name: str) -> dict:
        """Switch active project."""
        from backend.services.model_service import ModelService

        if ModelService.is_training():
            raise RuntimeError("Cannot switch project during training")

        project_path = settings.projects_dir / name
        if not project_path.exists():
            raise ValueError(f"Project '{name}' not found")

        # Update settings
        settings.active_project = name
        set_setting("active_project", name)
        settings.load()
        settings.ensure_dirs()

        return {"active_project": name}

    @classmethod
    def delete_project(cls, name: str) -> dict:
        """Delete a project (files + DB records)."""
        import shutil

        if name == settings.active_project:
            raise ValueError("Cannot delete the active project")

        project_path = settings.projects_dir / name
        if project_path.exists():
            shutil.rmtree(project_path)

        with get_connection() as conn:
            # Manually delete model_versions first because it lacks ON DELETE CASCADE
            conn.execute("DELETE FROM model_versions WHERE project = ?", (name,))
            conn.execute("DELETE FROM projects WHERE name = ?", (name,))
            conn.commit()

        return {"deleted": name}

    @classmethod
    def load_state(cls) -> None:
        """Load active project state on startup."""
        from backend.database import get_setting

        active = get_setting("active_project")
        if active:
            project_path = settings.projects_dir / active
            if project_path.exists():
                settings.active_project = active
            else:
                # Active project folder missing -> No active project
                settings.active_project = None
        else:
            settings.active_project = None

        settings.load()
        settings.ensure_dirs()

        # Ensure active project exists in DB (only if set)
        if settings.active_project:
            with get_connection() as conn:
                existing = conn.execute(
                    "SELECT name FROM projects WHERE name = ?",
                    (settings.active_project,),
                ).fetchone()
                if not existing:
                    # Sync DB with setting if folder exists but DB row missing
                    conn.execute(
                        """INSERT INTO projects
                           (name, created_at, training_config, augment_config, al_config)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            settings.active_project,
                            time.time(),
                            json.dumps(DEFAULT_TRAINING_CONFIG),
                            json.dumps(DEFAULT_AUGMENT_CONFIG),
                            json.dumps(DEFAULT_AL_CONFIG),
                        ),
                    )
                    conn.commit()
