"""Image service — loading, thumbnails, and file management."""
import hashlib
from pathlib import Path

from PIL import Image

from backend.config import settings


class ImageService:
    """Manages image files and thumbnails."""

    @staticmethod
    def get_images(split: str = "train") -> list[dict]:
        """List all images in a split directory with metadata."""
        img_dir = settings.images_dir / split
        if not img_dir.exists():
            return []

        images = []
        label_dir = settings.labels_dir / split

        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in settings.image_extensions:
                continue

            label_file = label_dir / f"{f.stem}.txt"
            has_label = label_file.exists() and label_file.stat().st_size > 0

            try:
                with Image.open(f) as img:
                    w, h = img.size
            except Exception:
                w, h = 0, 0

            # Count annotations
            ann_count = 0
            if has_label:
                ann_count = sum(
                    1 for line in label_file.read_text().strip().splitlines() if line.strip()
                )

            images.append(
                {
                    "name": f.name,
                    "split": split,
                    "width": w,
                    "height": h,
                    "status": "labeled" if has_label else "unlabeled",
                    "annotation_count": ann_count,
                }
            )

        return images

    @staticmethod
    def get_image_path(image_name: str, split: str = "train") -> Path | None:
        """Get full path to an image file."""
        path = settings.images_dir / split / image_name
        return path if path.exists() else None

    @staticmethod
    def generate_thumbnail(image_name: str, split: str = "train") -> Path | None:
        """Generate a thumbnail for an image."""
        src = settings.images_dir / split / image_name
        if not src.exists():
            return None

        thumb_dir = settings.thumbnails_dir / split
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"{src.stem}_thumb.jpg"

        if thumb_path.exists():
            # Check if source is newer
            if src.stat().st_mtime <= thumb_path.stat().st_mtime:
                return thumb_path

        try:
            with Image.open(src) as img:
                img.thumbnail(settings.thumbnail_size)
                img = img.convert("RGB")
                img.save(thumb_path, "JPEG", quality=85)
            return thumb_path
        except Exception:
            return None

    @staticmethod
    def get_image_dimensions(image_name: str, split: str = "train") -> tuple[int, int]:
        """Get image dimensions without loading full image."""
        path = settings.images_dir / split / image_name
        if not path.exists():
            return (0, 0)
        try:
            with Image.open(path) as img:
                return img.size
        except Exception:
            return (0, 0)

    @staticmethod
    def upload_images(files: list, split: str = "train") -> list[str]:
        """Save uploaded image files to the dataset directory."""
        dest_dir = settings.images_dir / split
        dest_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for file_data, filename in files:
            dest = dest_dir / filename
            if dest.exists():
                # Add hash suffix to avoid overwrite
                h = hashlib.md5(file_data[:1024]).hexdigest()[:6]
                dest = dest_dir / f"{dest.stem}_{h}{dest.suffix}"

            dest.write_bytes(file_data)
            saved.append(dest.name)

        return saved
