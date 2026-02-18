"""Pydantic schemas for YOLOLabel AI API."""
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """A single bounding box annotation."""

    class_id: int = Field(ge=0, description="Class index")
    x_center: float = Field(ge=0.0, le=1.0, description="Normalized center X")
    y_center: float = Field(ge=0.0, le=1.0, description="Normalized center Y")
    width: float = Field(gt=0.0, le=1.0, description="Normalized width")
    height: float = Field(gt=0.0, le=1.0, description="Normalized height")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Prediction confidence (None if manual)"
    )
    is_predicted: bool = Field(default=False, description="Whether this is an auto-prediction")


class ImageAnnotation(BaseModel):
    """All annotations for a single image."""

    image_name: str
    boxes: list[BoundingBox] = []


class ClassInfo(BaseModel):
    """A class definition."""

    id: int
    name: str
    color: str = "#00ff00"


class ProjectConfig(BaseModel):
    """Project configuration."""

    name: str = "My Dataset"
    classes: list[ClassInfo] = []
    split: str = "train"


class SaveAnnotationRequest(BaseModel):
    """Request to save annotations for an image."""

    image_name: str
    split: str = "train"
    boxes: list[BoundingBox] = []


class TrainingConfig(BaseModel):
    """Training configuration."""

    model_name: str = "yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    patience: int = 10
    device: str = "0"
    
    # Advanced / Hyperparameters
    optimizer: str = "auto"  # SGD, Adam, etc.
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    
    # Augmentation
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0


class TrainingStatus(BaseModel):
    """Training progress status."""

    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    metrics: dict = {}
    model_path: str | None = None
    error: str | None = None


class PredictionRequest(BaseModel):
    """Request for auto-prediction."""

    image_names: list[str] = []
    split: str = "train"
    confidence: float = 0.25


class Project(BaseModel):
    name: str
    created_at: float | None = None
    image_count: int = 0
    label_count: int = 0


class ProjectCreate(BaseModel):
    name: str


class PredictionResult(BaseModel):
    """Auto-prediction result for one image."""

    image_name: str
    boxes: list[BoundingBox] = []
    uncertainty_score: float = 1.0


class ActiveLearningSettings(BaseModel):
    """Active learning configuration."""

    auto_train_threshold: int = 10
    prediction_confidence: float = 0.25
    base_model: str = "yolo26n.pt"
    
    # Advanced Training Params
    optimizer: str = "auto"
    device: str = "0"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    
    # Augmentation
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0


class ImageInfo(BaseModel):
    """Image metadata for gallery."""

    name: str
    split: str
    width: int = 0
    height: int = 0
    thumbnail: str = ""
    status: str = "unlabeled"  # unlabeled | labeled | predicted
    annotation_count: int = 0
    uncertainty_score: float = 1.0


class ExportResult(BaseModel):
    """YOLO export result."""

    dataset_yaml_path: str
    total_images: int
    total_labels: int
    classes: list[ClassInfo]


# ──────────────────────────────────────────────
#  MLOps Schemas
# ──────────────────────────────────────────────

class RunSummary(BaseModel):
    """Training run summary for list views."""

    id: str
    project: str
    started_at: float
    finished_at: float | None = None
    duration_sec: float | None = None
    status: str = "running"
    base_model: str = ""
    epochs_total: int = 0
    epochs_done: int = 0
    best_mAP50: float | None = None
    best_mAP50_95: float | None = None
    best_precision: float | None = None
    best_recall: float | None = None
    run_dir: str | None = None
    notes: str = ""


class ModelVersion(BaseModel):
    """Model registry version."""

    id: int
    run_id: str
    project: str
    version: int
    stage: str = "none"
    promoted_at: float | None = None
    mAP50_95: float | None = None
    notes: str = ""


class ALCycleSummary(BaseModel):
    """Active learning cycle summary."""

    id: int
    project: str
    cycle_num: int
    started_at: float
    finished_at: float | None = None
    model_used: str | None = None
    images_predicted: int = 0
    images_labeled: int = 0
    images_accepted: int = 0
    images_rejected: int = 0
    run_id: str | None = None

