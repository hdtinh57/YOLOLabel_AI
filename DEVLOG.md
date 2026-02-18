# Development Log — YOLOLabel AI

> **⚠️ QUAN TRỌNG:** Luôn đọc file này TRƯỚC KHI thực thi bất kỳ thay đổi nào.
> Sau khi hoàn thành công việc, CẬP NHẬT log này với những gì đã làm.

---

## Quy tắc

1. **Đọc trước** — Đọc DEVLOG.md trước khi bắt đầu bất kỳ task nào
2. **Hiểu context** — Nắm rõ trạng thái hiện tại, file nào đã thay đổi, bug nào đang mở
3. **Ghi sau** — Sau khi hoàn thành, thêm entry mới vào đầu mục `## Log`
4. **Format** — Mỗi entry ghi rõ: ngày, mô tả ngắn, file thay đổi, trạng thái

---

## Trạng thái hiện tại

| Hạng mục                                 | Trạng thái      |
| ---------------------------------------- | --------------- |
| Phase 1: Planning                        | ✅ Done         |
| Phase 2: Database & Data Restructuring   | ✅ Done         |
| Phase 3: MLOps Backend + Active Learning | ✅ Done         |
| Phase 4: Frontend Dashboard              | ⏳ Chưa bắt đầu |
| Phase 5: Optional MLflow Integration     | ⏳ Chưa bắt đầu |
| Phase 6: Verification & Testing          | ⏳ Chưa bắt đầu |

---

## Log

### 2026-02-18 — v2.0 Verification (Phase 6)

**Mô tả:** Verification & Testing — viết unit/integration tests cho backend services và API endpoints.

**Files tạo mới:**

| File                      | Mục đích                                                                                                  |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| `tests/test_database.py`  | 40+ tests: DB init, WAL/FK attributes, Settings CRUD, `TrainingHistoryService`, `ModelRegistryService`    |
| `tests/test_mlops_api.py` | 25+ tests: FastAPI TestClient cho tất cả endpoints (dashboard, runs list/detail/compare, promote, config) |
| `tests/__init__.py`       | Test package marker                                                                                       |

**Kết quả:**

- 100% core services được test (CRUD runs, pagination, auto-promote logic, active learning cycles)
- API endpoints trả về đúng cấu trúc JSON
- `MLflowIntegration` được test mock env vars
- Đã fix các lỗi mismatch key (`epoch_metrics`, `best_run`) phát hiện trong quá trình test

### 2026-02-18 — v2.0 MLflow Integration (Phase 5)

**Mô tả:** Optional MLflow tracking — auto-detect, enable/disable toggle, env var management cho Ultralytics built-in callback.

**File tạo mới:**

| File                                     | Mục đích                                                                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `backend/services/mlflow_integration.py` | `MLflowIntegration` class: `is_available()`, `enable()`, `disable()`, `get_status()`, `setup_for_run()`, `cleanup_after_run()` |

**Files sửa:**

| File                                | Thay đổi                                                                                  |
| ----------------------------------- | ----------------------------------------------------------------------------------------- |
| `backend/services/model_service.py` | Hook `setup_for_run()` trước `model.train()`, `cleanup_after_run()` trong `finally` block |
| `backend/routers/mlops.py`          | 3 endpoints mới: `GET /mlflow/status`, `POST /mlflow/enable`, `POST /mlflow/disable`      |

**Cách hoạt động:**

- Ultralytics tự động log qua built-in MLflow callback khi `MLFLOW_TRACKING_URI` env var được set
- Module chỉ toggle env vars → zero coupling với training code
- Nếu `mlflow` chưa cài → `is_available()` trả `False`, enable trả error message

### 2026-02-18 — v2.0 Frontend Dashboard (Phase 4)

**Mô tả:** Xây dựng toàn bộ frontend MLOps Dashboard — hiển thị training history, model registry, charts, compare runs.

**Files tạo mới (2):**

| File                         | Mục đích                                                                                                                                                                                                                                |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frontend/css/dashboard.css` | Dark-mode styles: KPI cards grid, runs table, chart containers, model registry list, status/stage badges, run detail modal, compare table, responsive layout                                                                            |
| `frontend/js/dashboard.js`   | Dashboard controller: `load()`, KPI rendering, runs table với checkbox compare (max 5), run detail modal với Chart.js (loss curves + mAP/precision/recall), model registry với stage transitions, compare modal với metric highlighting |

**Files sửa (3):**

| File                  | Thay đổi                                                                                                                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frontend/js/api.js`  | Thêm `mlops` namespace: `getDashboardStats`, `getRuns`, `getRunDetail`, `compareRuns`, `deleteRun`, `getModels`, `promoteRun`, `setStage`, `getALCycles`, `getConfig`, `updateConfig`                             |
| `frontend/index.html` | Chart.js CDN, `dashboard.css` link, MLOps tab button (biểu tượng chart) trong header, dashboard view container (KPI cards + runs table + registry), run detail modal (stats grid + 2 chart canvas), compare modal |
| `frontend/js/app.js`  | Split-btn selector guard (skip MLOps button), MLOps tab toggle (show/hide dashboard vs labeling view), `Dashboard.init()`                                                                                         |

**Dashboard features:**

- 4 KPI cards: Total Runs, Best mAP@50, Production Model, AL Cycles
- Runs table: sortable, checkbox select (max 5), row click → detail
- Run Detail modal: 4 metric cards + 2 Chart.js charts (loss + metrics)
- Compare modal: side-by-side metric table with "better" highlighting
- Model Registry: version list với stage badges, promote/archive buttons
- Tab switching: Overview ↔ Model Registry

### 2026-02-18 — v2.0 MLOps Backend (Phase 2 + 3)

**Mô tả:** Xây dựng toàn bộ backend MLOps — database layer, training history, model registry, active learning redesign, migration script, và API router.

**Files tạo mới (5):**

| File                                           | Mục đích                                                                                                                                           |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/database.py`                          | SQLite manager — 7 tables (settings, projects, training_runs, epoch_metrics, model_versions, al_cycles, predictions), WAL mode, connection pooling |
| `backend/services/training_history_service.py` | CRUD cho training runs + epoch metrics, pagination, comparison, dashboard stats                                                                    |
| `backend/services/model_registry_service.py`   | Model versioning, stage transitions (none→staging→production→archived), auto-promote nếu mAP cải thiện                                             |
| `backend/migrate.py`                           | Migration legacy JSON/CSV → SQLite (chạy 1 lần khi startup)                                                                                        |
| `backend/routers/mlops.py`                     | 16 API endpoints: dashboard KPIs, runs CRUD, model registry, AL cycles, config                                                                     |

**Files viết lại (7):**

| File                                  | Thay đổi chính                                                                                                                                                                   |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/config.py`                   | Settings → SQLite. Config chia thành `training_config`, `augment_config`, `al_config` dicts. Backward-compat properties giữ nguyên cho code cũ                                   |
| `backend/services/model_service.py`   | Training logs per-epoch vào SQLite. Auto-promote production model. `auto_load_production_model()` khi startup. Unique `run_{timestamp}` cho mỗi run                              |
| `backend/services/active_learning.py` | Predictions → SQLite (persistent). AL cycle tracking. Multi-factor uncertainty scoring (confidence 40%, borderline 30%, count anomaly 20%, variance 10%). Accept/reject workflow |
| `backend/services/project_service.py` | Projects CRUD via SQLite. Training run counts trong listing                                                                                                                      |
| `backend/routers/predict.py`          | Aligned với method signatures mới của ActiveLearningService                                                                                                                      |
| `backend/routers/settings.py`         | Dict-based config read/write thay vì flat attributes                                                                                                                             |
| `backend/routers/model.py`            | Dict-based config updates cho training params                                                                                                                                    |

**Files sửa nhỏ (3):**

| File                                | Thay đổi                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| `backend/main.py`                   | Thêm DB init, migration, model auto-load vào startup + register mlops router |
| `backend/services/label_service.py` | Thêm `has_label()` method                                                    |
| `backend/models/schemas.py`         | Thêm `RunSummary`, `ModelVersion`, `ALCycleSummary` schemas                  |

**Startup flow mới:**

```
init_db() → migrate_legacy() → load_state() → auto_load_production_model()
```

**Database schema:**

- `settings` — global key-value
- `projects` — metadata + training_config, augment_config, al_config (JSON blobs)
- `training_runs` — run records với best metrics
- `epoch_metrics` — per-epoch loss, mAP, precision, recall
- `model_versions` — registry với stage management
- `al_cycles` — active learning cycle tracking
- `predictions` — cached predictions + uncertainty scores

**Quyết định kiến trúc:**

- Chọn Approach B: Custom SQLite + optional MLflow (không bắt buộc)
- Single SQLite file `data/yololabel.db` là nguồn dữ liệu chính
- Per-project config stored dưới dạng JSON blob trong `projects` table
- Training runs lưu tại `data/projects/{name}/runs/run_{timestamp}/`
- Pretrained models tại `weights/` (shared)
- Backward compat via `__getattr__` fallback trong Settings class

---

### 2026-02-18 — v2.0 Planning (Phase 1)

**Mô tả:** Khảo sát codebase, phân tích vấn đề data layout, research MLOps tools, viết implementation plan.

**Vấn đề phát hiện:**

- Config lưu bằng JSON files rải rác → dễ mất, khó quản lý
- Training runs không version → ghi đè lẫn nhau
- Predictions chỉ lưu in-memory → mất khi restart
- Không có training history, model registry, dashboard

**Quyết định:**

- Reject Approach A (full MLflow dependency) → quá nặng cho tool nhỏ
- Chọn Approach B (custom SQLite + optional MLflow callbacks)
- Directory restructure: `data/projects/{name}/runs/run_{ts}/`

**Files tạo:**

- `implementation_plan.md` (artifact) — plan chi tiết v3
- `task.md` (artifact) — task breakdown theo phases

---

### Trước 2026-02-18 — v1.0 Original

**Stack:** FastAPI + Vanilla JS + YOLO Ultralytics
**Features:** Canvas labeling, class management, training, auto-predict, import/export
**Limitations:** No persistence cho predictions, no training history, no model versioning
