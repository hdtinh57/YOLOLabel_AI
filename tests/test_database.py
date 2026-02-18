"""Tests for database layer and MLOps services.

Uses an in-memory SQLite approach: initializes the DB in a temp directory,
runs CRUD, then cleans up. No external dependencies needed.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Initialize a fresh DB for each test."""
    from backend import database
    database.init_db(tmp_path)

    # Also create a project record (needed as FK reference)
    with database.get_connection() as conn:
        conn.execute(
            "INSERT INTO projects (name, created_at) VALUES (?, ?)",
            ("test-project", time.time()),
        )
        conn.commit()

    yield tmp_path

    # Reset module state
    database._DB_PATH = None


# ──────────────────────────────────────────────
#  Database Core Tests
# ──────────────────────────────────────────────

class TestDatabase:
    """Test database initialization and helpers."""

    def test_init_creates_file(self, setup_db):
        db_file = setup_db / "yololabel.db"
        assert db_file.exists()

    def test_schema_version_set(self):
        from backend.database import get_setting
        version = get_setting("schema_version")
        assert version == 1

    def test_get_set_setting(self):
        from backend.database import get_setting, set_setting

        set_setting("test_key", {"nested": True, "value": 42})
        result = get_setting("test_key")
        assert result == {"nested": True, "value": 42}

    def test_get_setting_default(self):
        from backend.database import get_setting
        assert get_setting("nonexistent", "fallback") == "fallback"

    def test_setting_overwrite(self):
        from backend.database import get_setting, set_setting
        set_setting("overwrite_key", "v1")
        set_setting("overwrite_key", "v2")
        assert get_setting("overwrite_key") == "v2"

    def test_connection_wal_mode(self):
        from backend.database import get_connection
        with get_connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"

    def test_connection_fk_enabled(self):
        from backend.database import get_connection
        with get_connection() as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            assert fk == 1

    def test_tables_created(self):
        from backend.database import get_connection
        expected = {
            "settings", "projects", "training_runs",
            "epoch_metrics", "model_versions", "al_cycles", "predictions",
        }
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            tables = {r[0] for r in rows}
            assert expected.issubset(tables)

    def test_row_to_dict(self):
        from backend.database import get_connection, row_to_dict
        with get_connection() as conn:
            row = conn.execute(
                "SELECT key, value FROM settings WHERE key = 'schema_version'"
            ).fetchone()
            d = row_to_dict(row)
            assert isinstance(d, dict)
            assert d["key"] == "schema_version"

    def test_row_to_dict_none(self):
        from backend.database import row_to_dict
        assert row_to_dict(None) is None

    def test_rows_to_list(self):
        from backend.database import get_connection, rows_to_list
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM settings").fetchall()
            result = rows_to_list(rows)
            assert isinstance(result, list)
            assert all(isinstance(r, dict) for r in result)


# ──────────────────────────────────────────────
#  Training History Service Tests
# ──────────────────────────────────────────────

class TestTrainingHistoryService:
    """Test TrainingHistoryService CRUD operations."""

    def _create_run(self, run_id="run_test_001", **overrides):
        from backend.services.training_history_service import TrainingHistoryService

        defaults = {
            "run_id": run_id,
            "project": "test-project",
            "base_model": "yolo26n.pt",
            "epochs_total": 10,
            "hyperparams": {"lr0": 0.01},
            "augmentation": {"fliplr": 0.5},
            "dataset_snapshot": {"train_images": 100, "val_images": 20},
            "run_dir": f"/tmp/runs/{run_id}",
        }
        defaults.update(overrides)
        return TrainingHistoryService.create_run(**defaults)

    def test_create_run(self):
        result = self._create_run()
        assert result["run_id"] == "run_test_001"
        assert result["status"] == "running"

    def test_create_duplicate_raises(self):
        self._create_run("run_dup")
        with pytest.raises(Exception):
            self._create_run("run_dup")

    def test_log_epoch(self):
        from backend.services.training_history_service import TrainingHistoryService
        self._create_run()

        metrics = {
            "train/box_loss": 0.5,
            "train/cls_loss": 0.3,
            "metrics/mAP50(B)": 0.65,
            "metrics/mAP50-95(B)": 0.42,
            "metrics/precision(B)": 0.7,
            "metrics/recall(B)": 0.6,
        }
        TrainingHistoryService.log_epoch("run_test_001", 0, metrics)

        detail = TrainingHistoryService.get_run_detail("run_test_001")
        assert detail is not None
        assert len(detail["epoch_metrics"]) == 1
        assert detail["epoch_metrics"][0]["mAP50"] == pytest.approx(0.65)

    def test_log_multiple_epochs(self):
        from backend.services.training_history_service import TrainingHistoryService
        self._create_run()

        for i in range(5):
            TrainingHistoryService.log_epoch("run_test_001", i, {
                "metrics/mAP50(B)": 0.5 + i * 0.05,
            })

        detail = TrainingHistoryService.get_run_detail("run_test_001")
        assert len(detail["epoch_metrics"]) == 5

    def test_complete_run(self):
        from backend.services.training_history_service import TrainingHistoryService
        self._create_run()

        final = {"mAP50": 0.85, "mAP50_95": 0.62, "precision": 0.9, "recall": 0.8}
        TrainingHistoryService.complete_run("run_test_001", "completed", final)

        detail = TrainingHistoryService.get_run_detail("run_test_001")
        run = detail
        assert run["status"] == "completed"
        assert run["best_mAP50"] == pytest.approx(0.85)
        assert run["finished_at"] is not None
        assert run["duration_sec"] is not None

    def test_complete_run_failed(self):
        from backend.services.training_history_service import TrainingHistoryService
        self._create_run()
        TrainingHistoryService.complete_run("run_test_001", "failed")

        detail = TrainingHistoryService.get_run_detail("run_test_001")
        assert detail["status"] == "failed"

    def test_get_runs_pagination(self):
        from backend.services.training_history_service import TrainingHistoryService

        for i in range(5):
            self._create_run(f"run_page_{i}")

        result = TrainingHistoryService.get_runs("test-project", limit=3, offset=0)
        assert len(result["runs"]) == 3
        assert result["total"] == 5

        result2 = TrainingHistoryService.get_runs("test-project", limit=3, offset=3)
        assert len(result2["runs"]) == 2

    def test_get_runs_empty_project(self):
        from backend.services.training_history_service import TrainingHistoryService
        result = TrainingHistoryService.get_runs("nonexistent-project")
        assert result["runs"] == []
        assert result["total"] == 0

    def test_compare_runs(self):
        from backend.services.training_history_service import TrainingHistoryService
        self._create_run("run_a")
        self._create_run("run_b")

        TrainingHistoryService.complete_run("run_a", "completed", {"mAP50": 0.7})
        TrainingHistoryService.complete_run("run_b", "completed", {"mAP50": 0.8})

        result = TrainingHistoryService.compare_runs(["run_a", "run_b"])
        assert len(result) == 2

    def test_delete_run(self):
        from backend.services.training_history_service import TrainingHistoryService
        self._create_run("run_delete")
        TrainingHistoryService.log_epoch("run_delete", 0, {"metrics/mAP50(B)": 0.5})

        TrainingHistoryService.delete_run("run_delete")
        detail = TrainingHistoryService.get_run_detail("run_delete")
        assert detail is None

    def test_dashboard_stats(self):
        from backend.services.training_history_service import TrainingHistoryService
        self._create_run("run_s1")
        self._create_run("run_s2")

        TrainingHistoryService.complete_run("run_s1", "completed", {"mAP50": 0.7, "mAP50_95": 0.5})
        TrainingHistoryService.complete_run("run_s2", "completed", {"mAP50": 0.9, "mAP50_95": 0.65})

        stats = TrainingHistoryService.get_dashboard_stats("test-project")
        assert stats["total_runs"] == 2
        assert stats["total_runs"] == 2
        # completed_runs is not explicitly returned, check best_run exists
        assert stats["best_run"] is not None
        assert stats["best_run"]["best_mAP50"] == pytest.approx(0.9)
        assert stats["best_run"]["best_mAP50_95"] == pytest.approx(0.65)

    def test_run_detail_not_found(self):
        from backend.services.training_history_service import TrainingHistoryService
        assert TrainingHistoryService.get_run_detail("nonexistent") is None


# ──────────────────────────────────────────────
#  Model Registry Service Tests
# ──────────────────────────────────────────────

class TestModelRegistryService:
    """Test ModelRegistryService versioning and stage transitions."""

    def _create_completed_run(self, run_id="run_reg_001", mAP50_95=0.6):
        from backend.services.training_history_service import TrainingHistoryService
        TrainingHistoryService.create_run(
            run_id=run_id, project="test-project", base_model="yolo26n.pt",
            epochs_total=10, hyperparams={}, augmentation={},
            dataset_snapshot={}, run_dir=f"/tmp/runs/{run_id}",
        )
        TrainingHistoryService.complete_run(
            run_id, "completed", {"mAP50_95": mAP50_95, "mAP50": mAP50_95 + 0.2}
        )

    def test_promote_run(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run()

        result = ModelRegistryService.promote_run("run_reg_001", "test-project")
        assert result["version"] == 1
        assert result["stage"] == "staging"
        assert result["run_id"] == "run_reg_001"

    def test_promote_increments_version(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run("run_v1")
        self._create_completed_run("run_v2")

        r1 = ModelRegistryService.promote_run("run_v1", "test-project")
        r2 = ModelRegistryService.promote_run("run_v2", "test-project")
        assert r1["version"] == 1
        assert r2["version"] == 2

    def test_promote_nonexistent_raises(self):
        from backend.services.model_registry_service import ModelRegistryService
        with pytest.raises(ValueError, match="not found"):
            ModelRegistryService.promote_run("no_such_run", "test-project")

    def test_transition_staging_to_production(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run()
        promoted = ModelRegistryService.promote_run("run_reg_001", "test-project")

        # Staging → production
        result = ModelRegistryService.transition_stage(1, "production")
        assert result["stage"] == "production"

        prod = ModelRegistryService.get_production_model("test-project")
        assert prod is not None
        assert prod["version"] == 1

    def test_production_demotes_previous(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run("run_d1")
        self._create_completed_run("run_d2")

        ModelRegistryService.promote_run("run_d1", "test-project")
        ModelRegistryService.transition_stage(1, "production")

        ModelRegistryService.promote_run("run_d2", "test-project")
        ModelRegistryService.transition_stage(2, "production")

        # v1 should be archived now
        versions = ModelRegistryService.get_versions("test-project")
        v1 = next(v for v in versions if v["version"] == 1)
        v2 = next(v for v in versions if v["version"] == 2)
        assert v1["stage"] == "archived"
        assert v2["stage"] == "production"

    def test_transition_to_archived(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run()
        ModelRegistryService.promote_run("run_reg_001", "test-project")

        result = ModelRegistryService.transition_stage(1, "archived")
        assert result["stage"] == "archived"

    def test_transition_invalid_version(self):
        from backend.services.model_registry_service import ModelRegistryService
        with pytest.raises(ValueError, match="not found"):
            ModelRegistryService.transition_stage(999, "production")

    def test_transition_invalid_stage(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run()
        ModelRegistryService.promote_run("run_reg_001", "test-project")

        with pytest.raises(ValueError, match="Invalid stage"):
            ModelRegistryService.transition_stage(1, "unknown_stage")

    def test_get_versions(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run("run_gv1")
        self._create_completed_run("run_gv2")

        ModelRegistryService.promote_run("run_gv1", "test-project")
        ModelRegistryService.promote_run("run_gv2", "test-project")

        versions = ModelRegistryService.get_versions("test-project")
        assert len(versions) == 2
        # Should be desc by version
        assert versions[0]["version"] > versions[1]["version"]

    def test_get_production_model_none(self):
        from backend.services.model_registry_service import ModelRegistryService
        assert ModelRegistryService.get_production_model("test-project") is None

    def test_auto_promote_first_run(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run("run_auto1", mAP50_95=0.5)

        result = ModelRegistryService.auto_promote_if_better("run_auto1", "test-project")
        assert result is not None
        assert result["stage"] == "production"

    def test_auto_promote_better_run(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run("run_ap1", mAP50_95=0.5)
        ModelRegistryService.auto_promote_if_better("run_ap1", "test-project")

        self._create_completed_run("run_ap2", mAP50_95=0.7)
        result = ModelRegistryService.auto_promote_if_better("run_ap2", "test-project")
        assert result is not None
        assert result["stage"] == "production"

        # Old model should be archived
        prod = ModelRegistryService.get_production_model("test-project")
        assert prod["run_id"] == "run_ap2"

    def test_auto_promote_worse_run_skipped(self):
        from backend.services.model_registry_service import ModelRegistryService
        self._create_completed_run("run_good", mAP50_95=0.8)
        ModelRegistryService.auto_promote_if_better("run_good", "test-project")

        self._create_completed_run("run_bad", mAP50_95=0.3)
        result = ModelRegistryService.auto_promote_if_better("run_bad", "test-project")
        assert result is None

        prod = ModelRegistryService.get_production_model("test-project")
        assert prod["run_id"] == "run_good"


# ──────────────────────────────────────────────
#  MLflow Integration Tests
# ──────────────────────────────────────────────

class TestMLflowIntegration:
    """Test MLflow integration module (without requiring mlflow installed)."""

    def test_status_returns_dict(self):
        from backend.services.mlflow_integration import MLflowIntegration
        status = MLflowIntegration.get_status()
        assert isinstance(status, dict)
        assert "available" in status
        assert "enabled" in status

    def test_disable_clears_env(self):
        from backend.services.mlflow_integration import MLflowIntegration
        os.environ["MLFLOW_TRACKING_URI"] = "http://test"
        MLflowIntegration.disable()
        assert "MLFLOW_TRACKING_URI" not in os.environ

    def test_setup_cleanup_cycle(self):
        from backend.services.mlflow_integration import MLflowIntegration

        MLflowIntegration.setup_for_run("run123", "proj")
        # If not enabled, no env vars should be set
        if not MLflowIntegration.is_enabled():
            assert "MLFLOW_RUN_NAME" not in os.environ

        MLflowIntegration.cleanup_after_run()
        assert "MLFLOW_RUN_NAME" not in os.environ
        assert "MLFLOW_TAGS" not in os.environ


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
