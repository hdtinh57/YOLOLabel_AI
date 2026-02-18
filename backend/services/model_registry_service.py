"""Model registry service — versioning, staging, and promotion."""
import json
import time

from backend.database import get_connection, row_to_dict, rows_to_list


class ModelRegistryService:
    """Manages model versions and stage transitions."""

    @staticmethod
    def promote_run(run_id: str, project: str, notes: str = "") -> dict:
        """Promote a training run to a new model version."""
        with get_connection() as conn:
            # Get run's mAP
            run = conn.execute(
                "SELECT best_mAP50_95, run_dir FROM training_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if not run:
                raise ValueError(f"Run {run_id} not found")

            # Determine next version number
            last = conn.execute(
                "SELECT MAX(version) as v FROM model_versions WHERE project = ?",
                (project,),
            ).fetchone()
            next_version = (last["v"] or 0) + 1

            conn.execute(
                """INSERT INTO model_versions
                   (run_id, project, version, stage, promoted_at, mAP50_95, notes)
                   VALUES (?, ?, ?, 'staging', ?, ?, ?)""",
                (run_id, project, next_version, time.time(),
                 run["best_mAP50_95"], notes),
            )
            conn.commit()

            return {
                "version": next_version,
                "run_id": run_id,
                "stage": "staging",
                "mAP50_95": run["best_mAP50_95"],
            }

    @staticmethod
    def transition_stage(version_id: int, new_stage: str) -> dict:
        """Transition a model version's stage.

        Valid transitions:
        - none → staging → production → archived
        - Any stage → archived
        """
        valid_stages = {"none", "staging", "production", "archived"}
        if new_stage not in valid_stages:
            raise ValueError(f"Invalid stage: {new_stage}")

        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE id = ?", (version_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Version {version_id} not found")

            # If promoting to production, demote any existing production model
            if new_stage == "production":
                conn.execute(
                    """UPDATE model_versions
                       SET stage = 'archived'
                       WHERE project = ? AND stage = 'production'""",
                    (row["project"],),
                )

            conn.execute(
                "UPDATE model_versions SET stage = ? WHERE id = ?",
                (new_stage, version_id),
            )
            conn.commit()

            return {"version_id": version_id, "stage": new_stage}

    @staticmethod
    def get_versions(project: str) -> list[dict]:
        """List all model versions for a project."""
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT mv.*, tr.base_model, tr.run_dir, tr.best_mAP50, tr.best_precision
                   FROM model_versions mv
                   JOIN training_runs tr ON mv.run_id = tr.id
                   WHERE mv.project = ?
                   ORDER BY mv.version DESC""",
                (project,),
            ).fetchall()
            return rows_to_list(rows)

    @staticmethod
    def get_production_model(project: str) -> dict | None:
        """Get the production model for a project."""
        with get_connection() as conn:
            row = conn.execute(
                """SELECT mv.*, tr.run_dir, tr.base_model
                   FROM model_versions mv
                   JOIN training_runs tr ON mv.run_id = tr.id
                   WHERE mv.project = ? AND mv.stage = 'production'
                   LIMIT 1""",
                (project,),
            ).fetchone()
            return row_to_dict(row)

    @staticmethod
    def auto_promote_if_better(run_id: str, project: str) -> dict | None:
        """Auto-promote a run if its mAP beats the current production model.

        Returns the new version info if promoted, else None.
        """
        with get_connection() as conn:
            # Get run's mAP
            run = conn.execute(
                "SELECT best_mAP50_95 FROM training_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if not run or not run["best_mAP50_95"]:
                return None

            # Get current production mAP
            prod = conn.execute(
                """SELECT mv.mAP50_95
                   FROM model_versions mv
                   WHERE mv.project = ? AND mv.stage = 'production'
                   LIMIT 1""",
                (project,),
            ).fetchone()

            current_mAP = prod["mAP50_95"] if prod else 0.0

            if run["best_mAP50_95"] > (current_mAP or 0.0):
                # Promote this run
                result = ModelRegistryService.promote_run(
                    run_id, project, notes="Auto-promoted (mAP improved)"
                )
                # Immediately set to production
                with get_connection() as conn2:
                    # Get the version id we just created
                    ver = conn2.execute(
                        """SELECT id FROM model_versions
                           WHERE run_id = ? AND project = ?
                           ORDER BY version DESC LIMIT 1""",
                        (run_id, project),
                    ).fetchone()
                    if ver:
                        ModelRegistryService.transition_stage(
                            ver["id"], "production"
                        )
                        result["stage"] = "production"
                return result

            return None

    @staticmethod
    def get_production_weights_path(project: str) -> str | None:
        """Get the file path to the production model's best.pt."""
        prod = ModelRegistryService.get_production_model(project)
        if not prod or not prod.get("run_dir"):
            return None

        from pathlib import Path
        best_pt = Path(prod["run_dir"]) / "weights" / "best.pt"
        return str(best_pt) if best_pt.exists() else None
