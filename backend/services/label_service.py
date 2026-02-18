"""Label service — YOLO format read/write and export."""
import yaml
from pathlib import Path

from backend.config import settings
from backend.models.schemas import BoundingBox, ClassInfo


class LabelService:
    """Manages YOLO-format annotation files."""

    # In-memory class registry
    _classes: list[ClassInfo] = []

    # Default color palette for classes
    COLORS = [
        "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
        "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
        "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
        "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
    ]

    @classmethod
    def get_classes(cls) -> list[ClassInfo]:
        """Get current class list."""
        if not cls._classes:
            cls._load_classes_from_yaml()
        return cls._classes

    @classmethod
    def reset(cls) -> None:
        """Clear cached classes (called on project switch)."""
        cls._classes = []

    @classmethod
    def set_classes(cls, classes: list[ClassInfo]) -> None:
        """Set class list and save to YAML."""
        cls._classes = classes
        cls._save_dataset_yaml()

    @classmethod
    def add_class(cls, name: str) -> ClassInfo:
        """Add a new class and return it."""
        new_id = len(cls._classes)
        color = cls.COLORS[new_id % len(cls.COLORS)]
        new_class = ClassInfo(id=new_id, name=name, color=color)
        cls._classes.append(new_class)
        cls._save_dataset_yaml()
        return new_class

    @classmethod
    def remove_class(cls, class_id: int) -> None:
        """Remove a class by ID and re-index remaining classes."""
        cls._classes = [c for c in cls._classes if c.id != class_id]
        # Re-index
        for i, c in enumerate(cls._classes):
            c.id = i
        cls._save_dataset_yaml()

    @classmethod
    def rename_class(cls, class_id: int, new_name: str) -> None:
        """Rename a class."""
        for c in cls._classes:
            if c.id == class_id:
                c.name = new_name
                break
        cls._save_dataset_yaml()

    @staticmethod
    def read_labels(image_name: str, split: str = "train") -> list[BoundingBox]:
        """Read YOLO labels for an image."""
        label_path = settings.labels_dir / split / f"{Path(image_name).stem}.txt"
        if not label_path.exists():
            return []

        boxes = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append(
                BoundingBox(
                    class_id=int(parts[0]),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                    confidence=float(parts[5]) if len(parts) > 5 else None,
                    is_predicted=False,
                )
            )
        return boxes

    @staticmethod
    def save_labels(image_name: str, boxes: list[BoundingBox], split: str = "train") -> None:
        """Save YOLO labels for an image."""
        label_dir = settings.labels_dir / split
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{Path(image_name).stem}.txt"

        lines = []
        for box in boxes:
            lines.append(
                f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} "
                f"{box.width:.6f} {box.height:.6f}"
            )

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    @staticmethod
    def delete_labels(image_name: str, split: str = "train") -> None:
        """Delete labels for an image."""
        label_path = settings.labels_dir / split / f"{Path(image_name).stem}.txt"
        if label_path.exists():
            label_path.unlink()

    @classmethod
    def get_labeled_count(cls, split: str = "train") -> int:
        """Count images that have labels."""
        label_dir = settings.labels_dir / split
        if not label_dir.exists():
            return 0
        return sum(
            1 for f in label_dir.iterdir()
            if f.suffix == ".txt" and f.stat().st_size > 0
        )

    @classmethod
    def has_label(cls, image_stem: str, split: str = "train") -> bool:
        """Check if an image has a non-empty label file."""
        label_path = settings.labels_dir / split / f"{image_stem}.txt"
        return label_path.exists() and label_path.stat().st_size > 0

    @classmethod
    def _save_dataset_yaml(cls) -> None:
        """Write dataset.yaml for YOLO training."""
        # 1. Validation set fallback
        train_img_dir = settings.images_dir / "train"
        val_img_dir = settings.images_dir / "val"
        
        has_val = val_img_dir.exists() and any(
            f.suffix.lower() in settings.image_extensions for f in val_img_dir.iterdir()
        )
        
        val_path = "images/val" if has_val else "images/train"

        # 2. Auto-generate classes if missing but labels exist
        if not cls._classes:
            # check for used class IDs
            max_id = -1
            label_dir = settings.labels_dir / "train"
            if label_dir.exists():
                for f in label_dir.glob("*.txt"):
                    if f.stat().st_size > 0:
                        try:
                            for line in f.read_text().strip().splitlines():
                                parts = line.split()
                                if parts:
                                    max_id = max(max_id, int(parts[0]))
                        except Exception:
                            pass
            
            if max_id >= 0:
                # generate default classes
                for i in range(max_id + 1):
                    color = cls.COLORS[i % len(cls.COLORS)]
                    cls._classes.append(ClassInfo(id=i, name=f"class_{i}", color=color))

        yaml_path = settings.project_path / "dataset.yaml"
        data = {
            "path": str(settings.project_path.resolve()),
            "train": "images/train",
            "val": val_path,
            "names": {c.id: c.name for c in cls._classes},
        }
        yaml_path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    @classmethod
    def _load_classes_from_yaml(cls) -> None:
        """Load classes from existing dataset.yaml."""
        yaml_path = settings.project_path / "dataset.yaml"
        if not yaml_path.exists():
            return

        try:
            data = yaml.safe_load(yaml_path.read_text())
            names = data.get("names", {})
            
            if isinstance(names, list):
                cls._classes = [
                    ClassInfo(
                        id=i,
                        name=name,
                        color=cls.COLORS[i % len(cls.COLORS)],
                    )
                    for i, name in enumerate(names)
                ]
            else:
                cls._classes = [
                    ClassInfo(
                        id=int(k),
                        name=v,
                        color=cls.COLORS[int(k) % len(cls.COLORS)],
                    )
                    for k, v in sorted(names.items(), key=lambda x: int(x[0]))
                ]
        except Exception:
            cls._classes = []

    @classmethod
    def export_dataset(cls) -> dict:
        """Export full dataset structure and validate."""
        cls._save_dataset_yaml()

        train_imgs = len(list((settings.images_dir / "train").glob("*"))) if (settings.images_dir / "train").exists() else 0
        val_imgs = len(list((settings.images_dir / "val").glob("*"))) if (settings.images_dir / "val").exists() else 0
        train_labels = cls.get_labeled_count("train")
        val_labels = cls.get_labeled_count("val")

        return {
            "dataset_yaml_path": str(settings.project_path / "dataset.yaml"),
            "total_images": train_imgs + val_imgs,
            "total_labels": train_labels + val_labels,
            "train_images": train_imgs,
            "train_labels": train_labels,
            "val_images": val_imgs,
            "val_labels": val_labels,
            "classes": [c.model_dump() for c in cls._classes],
        }
