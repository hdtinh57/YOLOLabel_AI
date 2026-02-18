"""Active learning service — SQLite-backed predictions, cycle tracking, smart queue."""
import json
import time
from pathlib import Path

from backend.config import settings
from backend.database import get_connection, row_to_dict, rows_to_list


class ActiveLearningService:
    """Manages active learning cycles with persistent prediction caching."""

    # ──────────────────────────────────────────────
    #  Uncertainty Scoring
    # ──────────────────────────────────────────────

    @staticmethod
    def compute_uncertainty(predictions: list[dict]) -> float:
        """Multi-factor uncertainty scoring.

        Factors:
        1. Mean confidence (40%) — lower confidence = higher uncertainty
        2. Borderline detections (30%) — detections near decision boundary
        3. Detection count anomaly (20%) — too few/many = uncertain
        4. Confidence variance (10%) — inconsistent predictions = uncertain
        """
        if not predictions:
            return 1.0  # No detections = maximum uncertainty

        confs = [p.get("confidence", 0.5) for p in predictions]
        n = len(confs)

        # Factor 1: Mean confidence
        mean_score = 1.0 - (sum(confs) / n)

        # Factor 2: Borderline detections ratio
        low = settings.uncertainty_borderline_low
        high = settings.uncertainty_borderline_high
        borderline = sum(1 for c in confs if low < c < high)
        borderline_score = borderline / n

        # Factor 3: Detection count anomaly
        expected_range = (1, 20)
        if n < expected_range[0]:
            count_score = 0.8
        elif n > expected_range[1]:
            count_score = 0.6
        else:
            count_score = 0.0

        # Factor 4: Confidence variance
        mean_c = sum(confs) / n
        variance = sum((c - mean_c) ** 2 for c in confs) / n
        var_score = min(variance * 4, 1.0)

        return round(
            0.4 * mean_score + 0.3 * borderline_score + 0.2 * count_score + 0.1 * var_score,
            4,
        )

    # ──────────────────────────────────────────────
    #  Cycle Management
    # ──────────────────────────────────────────────

    @classmethod
    def start_cycle(cls, project: str | None = None) -> dict:
        """Start a new active learning cycle."""
        project = project or settings.active_project

        with get_connection() as conn:
            # Get next cycle num
            row = conn.execute(
                "SELECT MAX(cycle_num) as max_num FROM al_cycles WHERE project = ?",
                (project,),
            ).fetchone()
            next_num = (row["max_num"] or 0) + 1

            # Get current model info
            from backend.services.model_service import ModelService
            model_info = ModelService.get_model_info()

            conn.execute(
                """INSERT INTO al_cycles
                   (project, cycle_num, started_at, model_used)
                   VALUES (?, ?, ?, ?)""",
                (project, next_num, time.time(), model_info.get("model_name", "")),
            )
            cycle_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()

        return {"cycle_id": cycle_id, "cycle_num": next_num}

    @classmethod
    def get_current_cycle(cls, project: str | None = None) -> dict | None:
        """Get the latest unfinished cycle, or None."""
        project = project or settings.active_project
        with get_connection() as conn:
            row = conn.execute(
                """SELECT * FROM al_cycles
                   WHERE project = ? AND finished_at IS NULL
                   ORDER BY cycle_num DESC LIMIT 1""",
                (project,),
            ).fetchone()
            return row_to_dict(row)

    @classmethod
    def complete_cycle(cls, cycle_id: int, run_id: str | None = None) -> None:
        """Mark a cycle as completed, optionally linking to a training run."""
        with get_connection() as conn:
            # Aggregate stats
            stats = conn.execute(
                """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN status = 'accepted' THEN 1 ELSE 0 END) as accepted,
                     SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                     SUM(CASE WHEN status = 'labeled' THEN 1 ELSE 0 END) as labeled
                   FROM predictions WHERE cycle_id = ?""",
                (cycle_id,),
            ).fetchone()

            conn.execute(
                """UPDATE al_cycles SET
                   finished_at = ?,
                   images_predicted = ?,
                   images_accepted = ?,
                   images_rejected = ?,
                   images_labeled = ?,
                   run_id = ?
                   WHERE id = ?""",
                (
                    time.time(),
                    stats["total"] or 0,
                    stats["accepted"] or 0,
                    stats["rejected"] or 0,
                    (stats["labeled"] or 0) + (stats["accepted"] or 0),
                    run_id,
                    cycle_id,
                ),
            )
            conn.commit()

    # ──────────────────────────────────────────────
    #  Predictions
    # ──────────────────────────────────────────────

    @classmethod
    def predict_unlabeled(
        cls, split: str = "train", confidence: float | None = None, model_path: str | None = None
    ) -> list[dict]:
        """Run predictions on unlabeled images and cache results to SQLite."""
        from backend.services.model_service import ModelService
        from backend.services.label_service import LabelService

        if model_path:
            print(f"[ActiveLearning] Switching model to: {model_path}")
            ModelService.load_model(custom_path=model_path)

        project = settings.active_project
        conf = confidence or settings.prediction_confidence

        # Ensure we have a cycle
        cycle = cls.get_current_cycle()
        if not cycle:
            cycle = cls.start_cycle()
        cycle_id = cycle["cycle_id"] if "cycle_id" in cycle else cycle["id"]

        # Find unlabeled images
        img_dir = settings.images_dir / split
        if not img_dir.exists():
            return []

        all_images = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in settings.image_extensions
        )

        unlabeled = [
            f for f in all_images
            if not LabelService.has_label(f.stem, split)
        ]

        results = []
        for img_path in unlabeled:
            preds = ModelService.predict(str(img_path), conf)
            uncertainty = cls.compute_uncertainty(preds)

            # Cache to SQLite
            with get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO predictions
                       (project, cycle_id, image_name, split, uncertainty, boxes, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                    (
                        project, cycle_id, img_path.name, split,
                        uncertainty, json.dumps(preds), time.time(),
                    ),
                )
                conn.commit()

            results.append({
                "image_name": img_path.name,
                "boxes": preds,
                "uncertainty_score": uncertainty,
                "prediction_count": len(preds),
            })

        # Update cycle stats
        with get_connection() as conn:
            conn.execute(
                "UPDATE al_cycles SET images_predicted = ? WHERE id = ?",
                (len(results), cycle_id),
            )
            conn.commit()

        results.sort(key=lambda x: x["uncertainty_score"], reverse=True)
        return results

    @classmethod
    def get_smart_queue(cls, split: str = "train") -> list[dict]:
        """Get prioritized image queue from cached predictions."""
        from backend.services.label_service import LabelService

        project = settings.active_project
        img_dir = settings.images_dir / split
        if not img_dir.exists():
            return []

        # Get cached predictions (most recent cycle)
        cached_predictions = {}
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT image_name, uncertainty, boxes, status
                   FROM predictions
                   WHERE project = ? AND split = ? AND status = 'pending'
                   ORDER BY uncertainty DESC""",
                (project, split),
            ).fetchall()
            for r in rows:
                cached_predictions[r["image_name"]] = {
                    "uncertainty": r["uncertainty"],
                    "boxes": json.loads(r["boxes"]) if r["boxes"] else [],
                    "status": r["status"],
                }

        queue = []
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in settings.image_extensions:
                continue

            has_label = LabelService.has_label(f.stem, split)
            cached = cached_predictions.get(f.name)

            if cached and not has_label:
                status = "predicted"
                priority = 0
                uncertainty = cached["uncertainty"]
                pred_count = len(cached["boxes"])
            elif not has_label:
                status = "unlabeled"
                priority = 1
                uncertainty = 1.0
                pred_count = 0
            else:
                status = "labeled"
                priority = 2
                uncertainty = 0.0
                pred_count = 0

            queue.append({
                "name": f.name,
                "split": split,
                "status": status,
                "priority": priority,
                "uncertainty_score": uncertainty,
                "prediction_count": pred_count,
            })

        # Sort: predicted (by uncertainty) → unlabeled → labeled
        queue.sort(key=lambda x: (x["priority"], -x["uncertainty_score"]))
        return queue

    @classmethod
    def get_predictions(cls, image_name: str) -> list[dict]:
        """Get cached predictions for an image."""
        with get_connection() as conn:
            row = conn.execute(
                """SELECT boxes FROM predictions
                   WHERE project = ? AND image_name = ? AND status = 'pending'
                   ORDER BY created_at DESC LIMIT 1""",
                (settings.active_project, image_name),
            ).fetchone()
            if row and row["boxes"]:
                return json.loads(row["boxes"])
            return []

    @classmethod
    def accept_prediction(cls, image_name: str) -> bool:
        """Mark a prediction as accepted."""
        with get_connection() as conn:
            conn.execute(
                """UPDATE predictions SET status = 'accepted'
                   WHERE project = ? AND image_name = ? AND status = 'pending'""",
                (settings.active_project, image_name),
            )
            conn.commit()
            return True

    @classmethod
    def reject_prediction(cls, image_name: str) -> bool:
        """Mark a prediction as rejected."""
        with get_connection() as conn:
            conn.execute(
                """UPDATE predictions SET status = 'rejected'
                   WHERE project = ? AND image_name = ? AND status = 'pending'""",
                (settings.active_project, image_name),
            )
            conn.commit()
            return True

    @classmethod
    def clear_predictions(cls) -> dict:
        """Clear all pending predictions for current project."""
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM predictions WHERE project = ? AND status = 'pending'",
                (settings.active_project,),
            )
            conn.commit()
        return {"cleared": True}

    # ──────────────────────────────────────────────
    #  Statistics
    # ──────────────────────────────────────────────

    @classmethod
    def get_stats(cls) -> dict:
        """Get active learning statistics."""
        from backend.services.model_service import ModelService
        from backend.services.label_service import LabelService

        project = settings.active_project
        img_dir = settings.images_dir / "train"
        total_images = sum(
            1 for f in img_dir.iterdir()
            if f.suffix.lower() in settings.image_extensions
        ) if img_dir.exists() else 0

        labeled = LabelService.get_labeled_count("train")

        # Pending predictions
        with get_connection() as conn:
            pending = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE project = ? AND status = 'pending'",
                (project,),
            ).fetchone()["cnt"]

            cycle_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM al_cycles WHERE project = ?",
                (project,),
            ).fetchone()["cnt"]

        return {
            "total_images": total_images,
            "labeled_count": labeled,
            "unlabeled_count": total_images - labeled,
            "pending_predictions": pending,
            "cycle_count": cycle_count,
            "auto_train_threshold": settings.auto_train_threshold,
            "should_train": labeled >= settings.auto_train_threshold,
            "model": ModelService.get_model_info(),
        }

    @classmethod
    def get_cycle_history(cls, limit: int = 20) -> list[dict]:
        """Get active learning cycle history."""
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM al_cycles
                   WHERE project = ?
                   ORDER BY cycle_num DESC LIMIT ?""",
                (settings.active_project, limit),
            ).fetchall()
            return rows_to_list(rows)

    @classmethod
    def should_auto_train(cls) -> dict:
        """Check if auto-training should be triggered."""
        from backend.services.model_service import ModelService
        from backend.services.label_service import LabelService

        labeled = LabelService.get_labeled_count("train")
        threshold = settings.auto_train_threshold

        return {
            "should_train": labeled >= threshold and not ModelService.is_training(),
            "labeled_count": labeled,
            "threshold": threshold,
            "is_training": ModelService.is_training(),
        }
