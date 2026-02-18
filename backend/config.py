"""Configuration — settings backed by SQLite."""
import json
from pathlib import Path

from pydantic import BaseModel


# Default training config for new projects
DEFAULT_TRAINING_CONFIG = {
    "model": "yolo26n.pt",
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "patience": 10,
    "device": "0",
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}

DEFAULT_AUGMENT_CONFIG = {
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

DEFAULT_AL_CONFIG = {
    "auto_train_threshold": 10,
    "prediction_confidence": 0.25,
}


class Settings(BaseModel):
    """Global application settings."""

    # Paths (computed from project_root)
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Path(__file__).parent.parent / "data"

    # Current active project
    active_project: str | None = None

    # Available pretrained models
    available_models: list[str] = [
        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    ]

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    thumbnail_size: tuple[int, int] = (200, 200)
    image_extensions: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    # ──────────────────────────────────────────────
    #  Per-project config (loaded from SQLite)
    # ──────────────────────────────────────────────
    training_config: dict = dict(DEFAULT_TRAINING_CONFIG)
    augment_config: dict = dict(DEFAULT_AUGMENT_CONFIG)
    al_config: dict = dict(DEFAULT_AL_CONFIG)

    # ──────────────────────────────────────────────
    #  Path properties
    # ──────────────────────────────────────────────

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def project_path(self) -> Path:
        if not self.active_project:
            # Fallback or raise? Raising prevents accidental writes to root
            raise ValueError("No active project set")
        return self.projects_dir / self.active_project

    @property
    def images_dir(self) -> Path:
        return self.project_path / "images"

    @property
    def labels_dir(self) -> Path:
        return self.project_path / "labels"

    @property
    def thumbnails_dir(self) -> Path:
        return self.project_path / "thumbnails"

    @property
    def runs_dir(self) -> Path:
        """Per-project training runs directory."""
        return self.project_path / "runs"

    @property
    def weights_dir(self) -> Path:
        """Shared pretrained weights directory."""
        return self.project_root / "weights"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "yololabel.db"

    # ──────────────────────────────────────────────
    #  Backward compatibility shims
    # ──────────────────────────────────────────────

    @property
    def models_dir(self) -> Path:
        """Backward compat: training output dir = runs_dir."""
        return self.runs_dir

    @property
    def default_model(self) -> str:
        return self.training_config.get("model", "yolo26n.pt")

    @property
    def default_imgsz(self) -> int:
        return self.training_config.get("imgsz", 640)

    @property
    def default_epochs(self) -> int:
        return self.training_config.get("epochs", 50)

    @property
    def default_batch(self) -> int:
        return self.training_config.get("batch", 16)

    @property
    def auto_train_threshold(self) -> int:
        return self.al_config.get("auto_train_threshold", 10)

    @property
    def prediction_confidence(self) -> float:
        return self.al_config.get("prediction_confidence", 0.25)

    @property
    def uncertainty_borderline_low(self) -> float:
        return 0.3

    @property
    def uncertainty_borderline_high(self) -> float:
        return 0.7

    # ──────────────────────────────────────────────
    #  Directory management
    # ──────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create required directories."""
        # Global dirs
        for d in [self.data_dir, self.projects_dir, self.weights_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Project-specific dirs (only if active project is set)
        if self.active_project:
            for d in [
                self.images_dir / "train",
                self.images_dir / "val",
                self.labels_dir / "train",
                self.labels_dir / "val",
                self.runs_dir,
                self.thumbnails_dir,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    #  SQLite persistence
    # ──────────────────────────────────────────────

    def save(self, include_global: bool = True, include_project: bool = True) -> None:
        """Save settings to SQLite database."""
        from backend.database import get_connection

        with get_connection() as conn:
            if include_global:
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                    ("active_project", json.dumps(self.active_project)),
                )

            if include_project:
                conn.execute(
                    """UPDATE projects
                       SET training_config = ?, augment_config = ?, al_config = ?
                       WHERE name = ?""",
                    (
                        json.dumps(self.training_config),
                        json.dumps(self.augment_config),
                        json.dumps(self.al_config),
                        self.active_project,
                    ),
                )
            conn.commit()

    def load(self) -> None:
        """Load settings from SQLite database."""
        from backend.database import get_connection, get_setting

        # 1. Load active project
        active = get_setting("active_project")
        if active:
            self.active_project = active

        # 2. Load project-specific config
        try:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT training_config, augment_config, al_config FROM projects WHERE name = ?",
                    (self.active_project,),
                ).fetchone()

                if row:
                    if row["training_config"]:
                        loaded = json.loads(row["training_config"])
                        self.training_config = {**DEFAULT_TRAINING_CONFIG, **loaded}
                    if row["augment_config"]:
                        loaded = json.loads(row["augment_config"])
                        self.augment_config = {**DEFAULT_AUGMENT_CONFIG, **loaded}
                    if row["al_config"]:
                        loaded = json.loads(row["al_config"])
                        self.al_config = {**DEFAULT_AL_CONFIG, **loaded}
                else:
                    self._reset_to_defaults()
        except Exception:
            self._reset_to_defaults()

    def _reset_to_defaults(self) -> None:
        """Reset per-project config to defaults."""
        self.training_config = dict(DEFAULT_TRAINING_CONFIG)
        self.augment_config = dict(DEFAULT_AUGMENT_CONFIG)
        self.al_config = dict(DEFAULT_AL_CONFIG)

    # Legacy compat: allow attribute-style access to training params
    def __getattr__(self, name: str):
        """Fallback for legacy code accessing settings.lr0, settings.epochs, etc."""
        for cfg in (self.training_config, self.augment_config, self.al_config):
            if name in cfg:
                return cfg[name]
        raise AttributeError(f"Settings has no attribute '{name}'")


settings = Settings()
