"""Training history service — CRUD for training runs and epoch metrics."""
import json
import time
from pathlib import Path

from backend.database import get_connection, row_to_dict, rows_to_list


class TrainingHistoryService:
    """Manages training run records in SQLite."""

    @staticmethod
    def create_run(
        run_id: str,
        project: str,
        base_model: str,
        epochs_total: int,
        hyperparams: dict,
        augmentation: dict,
        dataset_snapshot: dict,
        run_dir: str,
    ) -> dict:
        """Insert a new training run."""
        now = time.time()
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO training_runs
                   (id, project, started_at, status, base_model, epochs_total,
                    hyperparams, augmentation,
                    train_images, train_labels, val_images, val_labels,
                    num_classes, class_names, run_dir)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, project, now, "running", base_model, epochs_total,
                    json.dumps(hyperparams), json.dumps(augmentation),
                    dataset_snapshot.get("train_images", 0),
                    dataset_snapshot.get("train_labels", 0),
                    dataset_snapshot.get("val_images", 0),
                    dataset_snapshot.get("val_labels", 0),
                    dataset_snapshot.get("num_classes", 0),
                    json.dumps(dataset_snapshot.get("class_names", [])),
                    run_dir,
                ),
            )
            conn.commit()
        return {"run_id": run_id, "status": "running", "started_at": now}

    @staticmethod
    def log_epoch(run_id: str, epoch: int, metrics: dict) -> None:
        """Insert per-epoch metrics row."""
        with get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO epoch_metrics
                   (run_id, epoch,
                    train_box_loss, train_cls_loss, train_dfl_loss,
                    val_box_loss, val_cls_loss, val_dfl_loss,
                    precision_b, recall_b, mAP50, mAP50_95,
                    lr_pg0, lr_pg1, lr_pg2)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, epoch,
                    metrics.get("train/box_loss"),
                    metrics.get("train/cls_loss"),
                    metrics.get("train/dfl_loss"),
                    metrics.get("val/box_loss"),
                    metrics.get("val/cls_loss"),
                    metrics.get("val/dfl_loss"),
                    metrics.get("metrics/precision(B)"),
                    metrics.get("metrics/recall(B)"),
                    metrics.get("metrics/mAP50(B)"),
                    metrics.get("metrics/mAP50-95(B)"),
                    metrics.get("lr/pg0"),
                    metrics.get("lr/pg1"),
                    metrics.get("lr/pg2"),
                ),
            )
            conn.commit()

            # Also update epochs_done on the run
            conn.execute(
                "UPDATE training_runs SET epochs_done = ? WHERE id = ?",
                (epoch + 1, run_id),
            )
            conn.commit()

    @staticmethod
    def complete_run(
        run_id: str,
        status: str = "completed",
        final_metrics: dict | None = None,
    ) -> None:
        """Mark a run as completed/failed/stopped and save final metrics."""
        now = time.time()
        with get_connection() as conn:
            # Get started_at to compute duration
            row = conn.execute(
                "SELECT started_at FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            duration = now - row["started_at"] if row else 0

            updates = {
                "status": status,
                "finished_at": now,
                "duration_sec": round(duration, 1),
            }
            if final_metrics:
                updates["best_mAP50"] = final_metrics.get("mAP50")
                updates["best_mAP50_95"] = final_metrics.get("mAP50_95")
                updates["best_precision"] = final_metrics.get("precision")
                updates["best_recall"] = final_metrics.get("recall")
            else:
                # Fallback: Compute best metrics from recorded history
                agg = conn.execute(
                    """SELECT MAX(mAP50) as m50, MAX(mAP50_95) as m95,
                              MAX(precision_b) as p, MAX(recall_b) as r
                       FROM epoch_metrics WHERE run_id = ?""",
                    (run_id,),
                ).fetchone()
                if agg:
                    updates["best_mAP50"] = agg["m50"]
                    updates["best_mAP50_95"] = agg["m95"]
                    updates["best_precision"] = agg["p"]
                    updates["best_recall"] = agg["r"]

            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [run_id]
            conn.execute(
                f"UPDATE training_runs SET {set_clause} WHERE id = ?", values
            )
            conn.commit()

    @staticmethod
    def get_runs(
        project: str,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "started_at",
        order: str = "DESC",
    ) -> dict:
        """List training runs with pagination."""
        allowed_sorts = {
            "started_at", "best_mAP50_95", "best_mAP50",
            "duration_sec", "epochs_done", "status",
        }
        if sort_by not in allowed_sorts:
            sort_by = "started_at"
        order = "DESC" if order.upper() == "DESC" else "ASC"

        with get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM training_runs WHERE project = ?",
                (project,),
            ).fetchone()["cnt"]

            rows = conn.execute(
                f"""SELECT id, project, started_at, finished_at, duration_sec,
                           status, base_model, epochs_total, epochs_done,
                           best_mAP50, best_mAP50_95, best_precision, best_recall,
                           run_dir, notes
                    FROM training_runs
                    WHERE project = ?
                    ORDER BY {sort_by} {order}
                    LIMIT ? OFFSET ?""",
                (project, limit, offset),
            ).fetchall()

            return {
                "runs": rows_to_list(rows),
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    @staticmethod
    def get_run_detail(run_id: str) -> dict | None:
        """Get full run details including epoch metrics."""
        with get_connection() as conn:
            run = conn.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not run:
                return None

            epochs = conn.execute(
                """SELECT * FROM epoch_metrics
                   WHERE run_id = ? ORDER BY epoch""",
                (run_id,),
            ).fetchall()

            result = row_to_dict(run)
            # Parse JSON fields
            for field in ("hyperparams", "augmentation", "class_names"):
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result["epoch_metrics"] = rows_to_list(epochs)
            return result

    @staticmethod
    def compare_runs(run_ids: list[str]) -> list[dict]:
        """Get details for multiple runs for comparison."""
        results = []
        for rid in run_ids:
            detail = TrainingHistoryService.get_run_detail(rid)
            if detail:
                results.append(detail)
        return results

    @staticmethod
    def delete_run(run_id: str, delete_files: bool = False) -> bool:
        """Delete a training run record (cascade deletes epoch_metrics)."""
        with get_connection() as conn:
            if delete_files:
                row = conn.execute(
                    "SELECT run_dir FROM training_runs WHERE id = ?", (run_id,)
                ).fetchone()
                if row and row["run_dir"]:
                    import shutil
                    run_path = Path(row["run_dir"])
                    if run_path.exists():
                        shutil.rmtree(run_path, ignore_errors=True)

            conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))
            conn.commit()
            return True

    @staticmethod
    def get_dashboard_stats(project: str) -> dict:
        """Aggregate stats for dashboard KPIs."""
        with get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM training_runs WHERE project = ?",
                (project,),
            ).fetchone()["cnt"]

            best = conn.execute(
                """SELECT id, best_mAP50_95, best_mAP50
                   FROM training_runs
                   WHERE project = ? AND status = 'completed'
                   ORDER BY best_mAP50_95 DESC LIMIT 1""",
                (project,),
            ).fetchone()

            recent = conn.execute(
                """SELECT id, best_mAP50_95, started_at
                   FROM training_runs
                   WHERE project = ? AND status = 'completed'
                   ORDER BY started_at DESC LIMIT 2""",
                (project,),
            ).fetchall()

            # Compute trend: latest vs previous
            trend = None
            if len(recent) >= 2:
                curr = recent[0]["best_mAP50_95"] or 0
                prev = recent[1]["best_mAP50_95"] or 0
                trend = round(curr - prev, 4) if curr and prev else None

            # Production model
            prod = conn.execute(
                """SELECT mv.version, mv.mAP50_95, tr.base_model
                   FROM model_versions mv
                   JOIN training_runs tr ON mv.run_id = tr.id
                   WHERE mv.project = ? AND mv.stage = 'production'
                   LIMIT 1""",
                (project,),
            ).fetchone()

            return {
                "total_runs": total,
                "best_run": row_to_dict(best),
                "trend_mAP": trend,
                "production_model": row_to_dict(prod),
            }
