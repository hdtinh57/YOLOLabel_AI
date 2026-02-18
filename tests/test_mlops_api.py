"""Tests for MLOps API router endpoints.

Uses FastAPI TestClient with a fresh DB for each test.
"""
import json
import sys
import time
import pytest
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def setup_api(tmp_path, monkeypatch):
    """Initialize DB and create a test project before each API test."""
    from backend import database
    from backend.config import settings

    database.init_db(tmp_path)

    # Patch settings to use tmp_path
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "active_project", "test-project")

    # Create project record
    with database.get_connection() as conn:
        conn.execute(
            "INSERT INTO projects (name, created_at) VALUES (?, ?)",
            ("test-project", time.time()),
        )
        conn.commit()

    yield

    database._DB_PATH = None


@pytest.fixture
def client(setup_api):
    """FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from backend.routers.mlops import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _seed_run(run_id="run_api_001", mAP50_95=0.6):
    """Helper — seed a completed run."""
    from backend.services.training_history_service import TrainingHistoryService

    TrainingHistoryService.create_run(
        run_id=run_id, project="test-project", base_model="yolo26n.pt",
        epochs_total=10, hyperparams={"lr0": 0.01}, augmentation={},
        dataset_snapshot={"train_images": 50, "val_images": 10},
        run_dir=f"/tmp/{run_id}",
    )
    TrainingHistoryService.log_epoch(run_id, 0, {
        "metrics/mAP50(B)": mAP50_95 + 0.2,
        "metrics/mAP50-95(B)": mAP50_95,
        "train/box_loss": 0.5,
    })
    TrainingHistoryService.complete_run(run_id, "completed", {
        "mAP50": mAP50_95 + 0.2, "mAP50_95": mAP50_95,
        "precision": 0.85, "recall": 0.75,
    })


# ──────────────────────────────────────────────
#  Dashboard Stats
# ──────────────────────────────────────────────

class TestDashboardEndpoint:
    def test_dashboard_empty(self, client):
        resp = client.get("/api/mlops/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "training" in data
        assert "active_learning" in data

    def test_dashboard_with_runs(self, client):
        _seed_run("run_d1", 0.5)
        _seed_run("run_d2", 0.7)

        resp = client.get("/api/mlops/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["training"]["total_runs"] == 2
        assert data["training"]["best_run"]["best_mAP50_95"] == pytest.approx(0.7)


# ──────────────────────────────────────────────
#  Runs CRUD
# ──────────────────────────────────────────────

class TestRunsEndpoints:
    def test_list_runs_empty(self, client):
        resp = client.get("/api/mlops/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"] == []
        assert data["total"] == 0

    def test_list_runs(self, client):
        _seed_run("run_r1")
        _seed_run("run_r2")

        resp = client.get("/api/mlops/runs")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_list_runs_pagination(self, client):
        for i in range(5):
            _seed_run(f"run_p{i}")

        resp = client.get("/api/mlops/runs?limit=2&offset=0")
        assert len(resp.json()["runs"]) == 2

        resp2 = client.get("/api/mlops/runs?limit=2&offset=4")
        assert len(resp2.json()["runs"]) == 1

    def test_get_run_detail(self, client):
        _seed_run()
        resp = client.get("/api/mlops/runs/run_api_001")
        assert resp.status_code == 200

        data = resp.json()
        assert data["id"] == "run_api_001"
        assert data["status"] == "completed"
        assert len(data["epoch_metrics"]) == 1

    def test_get_run_not_found(self, client):
        resp = client.get("/api/mlops/runs/nonexistent")
        assert resp.status_code == 404

    def test_compare_runs(self, client):
        _seed_run("run_c1", 0.5)
        _seed_run("run_c2", 0.7)

        resp = client.post(
            "/api/mlops/runs/compare",
            json=["run_c1", "run_c2"],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_compare_too_few(self, client):
        resp = client.post("/api/mlops/runs/compare", json=["only_one"])
        assert resp.status_code == 400

    def test_compare_too_many(self, client):
        ids = [f"run_{i}" for i in range(6)]
        resp = client.post("/api/mlops/runs/compare", json=ids)
        assert resp.status_code == 400

    def test_delete_run(self, client):
        _seed_run("run_del")
        resp = client.delete("/api/mlops/runs/run_del")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "run_del"

        # Verify gone
        resp2 = client.get("/api/mlops/runs/run_del")
        assert resp2.status_code == 404


# ──────────────────────────────────────────────
#  Model Registry
# ──────────────────────────────────────────────

class TestRegistryEndpoints:
    def test_list_models_empty(self, client):
        resp = client.get("/api/mlops/models")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_promote_run(self, client):
        _seed_run()
        resp = client.post("/api/mlops/models/promote/run_api_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == 1
        assert data["stage"] == "staging"

    def test_promote_nonexistent(self, client):
        resp = client.post("/api/mlops/models/promote/no_run")
        assert resp.status_code == 400

    def test_set_stage(self, client):
        _seed_run()
        client.post("/api/mlops/models/promote/run_api_001")

        resp = client.put(
            "/api/mlops/models/1/stage",
            params={"stage": "production"},
        )
        assert resp.status_code == 200
        assert resp.json()["stage"] == "production"

    def test_set_stage_invalid(self, client):
        _seed_run()
        client.post("/api/mlops/models/promote/run_api_001")

        resp = client.put(
            "/api/mlops/models/1/stage",
            params={"stage": "broken"},
        )
        assert resp.status_code == 400

    def test_production_model(self, client):
        _seed_run()
        client.post("/api/mlops/models/promote/run_api_001")
        client.put("/api/mlops/models/1/stage", params={"stage": "production"})

        resp = client.get("/api/mlops/models/production")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == 1
        assert data["stage"] == "production"

    def test_production_model_none(self, client):
        resp = client.get("/api/mlops/models/production")
        assert resp.status_code == 200
        assert resp.json()["production"] is None


# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

class TestConfigEndpoints:
    def test_get_config(self, client):
        resp = client.get("/api/mlops/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "training" in data
        assert "augmentation" in data
        assert "active_learning" in data


# ──────────────────────────────────────────────
#  MLflow
# ──────────────────────────────────────────────

class TestMLflowEndpoints:
    def test_mlflow_status(self, client):
        resp = client.get("/api/mlops/mlflow/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "enabled" in data

    def test_mlflow_disable(self, client):
        resp = client.post("/api/mlops/mlflow/disable")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
