# YOLOLabel AI

> **AI-powered bounding box labeling tool with Active Learning & MLOps — built for YOLO11/YOLO26**

Label images → Train model → Auto-predict → Review → Repeat. A complete active learning loop with a built-in MLOps dashboard for tracking training history, model versions, and experiments.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/hdtinh57/YOLOLabel_AI.git
cd YOLOLabel_AI

# 2. Install
pip install -r backend/requirements.txt

# 3. Run
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open browser → http://localhost:8000
```

> **Python ≥ 3.10** required. GPU recommended for training (CUDA).

---

## Testing

Run the full test suite (backend services + API):

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
python -m pytest tests/ -v
```

## Features

### Core Labeling

- **Canvas editor** — draw, resize, move bounding boxes with keyboard shortcuts
- **Multi-class support** — define unlimited classes with colors
- **Auto-save** — annotations save on image switch
- **Import/Export** — YOLO-format `.txt` labels + `dataset.yaml`
- **Train/Val split** — drag images between splits

### Active Learning (v2.0)

- **Smart queue** — images ranked by model uncertainty (multi-factor scoring)
- **Prediction caching** — stored in SQLite, persistent across restarts
- **Cycle tracking** — each predict → review → train loop is versioned
- **Accept/Reject** — review predictions individually before committing
- **Auto-train trigger** — optional threshold-based training

### MLOps Dashboard (v2.0)

- **Training history** — all runs logged with per-epoch metrics
- **Model registry** — version stages: `none → staging → production → archived`
- **Auto-promote** — best model auto-promoted to production if mAP improves
- **Run comparison** — side-by-side metric comparison (up to 5 runs)
- **Per-project config** — training, augmentation, AL settings stored per project

---

## Workflow

```
1. Upload images          → Data tab or data/images/train/
2. Define classes          → Right panel class manager
3. Draw bounding boxes     → Press B for draw tool
4. Save annotations       → Press S (auto-save on switch)
5. Train model             → Click "Train Model" (10+ labels)
6. Auto predict            → Click "Auto Predict"
7. Review predictions      → Accept ✅ / Edit ✏️ / Reject ❌
8. Repeat from step 3      → Model improves each cycle
```

---

## Keyboard Shortcuts

| Key         | Action                 |
| ----------- | ---------------------- |
| `B`         | Draw tool              |
| `V`         | Select tool            |
| `H`         | Pan tool               |
| `1-9`       | Select class           |
| `D` / `Del` | Delete selected box    |
| `S`         | Save annotations       |
| `Ctrl+Z`    | Undo                   |
| `Space`     | Accept all predictions |
| `← →`       | Navigate images        |
| `+ -`       | Zoom                   |
| `0`         | Fit to view            |

---

## Architecture

```
YOLOLabel_AI/
├── backend/                     # FastAPI server (Python)
│   ├── main.py                  # App entry + startup lifecycle
│   ├── config.py                # SQLite-backed settings
│   ├── database.py              # SQLite manager (7 tables, WAL)
│   ├── migrate.py               # Legacy JSON → SQLite migration
│   ├── routers/                 # API layer
│   │   ├── images.py            #   Image upload/list/serve
│   │   ├── labels.py            #   Annotation CRUD
│   │   ├── model.py             #   Training, model loading
│   │   ├── predict.py           #   Auto-prediction, smart queue
│   │   ├── projects.py          #   Project management
│   │   ├── settings.py          #   Config get/update
│   │   └── mlops.py             #   Dashboard, runs, registry, AL cycles
│   ├── services/                # Business logic
│   │   ├── model_service.py     #   YOLO load/train/predict + DB logging
│   │   ├── active_learning.py   #   Uncertainty scoring, cycle tracking
│   │   ├── label_service.py     #   Label file I/O
│   │   ├── image_service.py     #   Image processing
│   │   ├── project_service.py   #   Project CRUD via SQLite
│   │   ├── training_history_service.py  #   Run/epoch metrics CRUD
│   │   └── model_registry_service.py    #   Model versioning
│   └── models/
│       └── schemas.py           # Pydantic request/response models
├── frontend/                    # Vanilla JS SPA
│   ├── index.html               # Main shell
│   ├── css/styles.css           # Dark-mode theme
│   └── js/
│       ├── app.js               # Main controller
│       ├── api.js               # Backend API client
│       ├── canvas.js            # Canvas rendering + interactions
│       ├── gallery.js           # Image gallery
│       ├── classManager.js      # Class CRUD UI
│       └── shortcuts.js         # Keyboard handler
├── data/                        # Dataset (auto-created)
│   ├── projects/{name}/         # Per-project data
│   │   └── runs/run_{ts}/       # Training run artifacts
│   ├── images/{train,val}/      # Image files
│   ├── labels/{train,val}/      # YOLO .txt labels
│   ├── dataset.yaml             # YOLO training config
│   └── yololabel.db             # SQLite database
└── weights/                     # Shared pretrained models
```

---

## Data Storage

All data is stored in a single SQLite database (`data/yololabel.db`) with 7 tables:

| Table            | Purpose                                                   |
| ---------------- | --------------------------------------------------------- |
| `settings`       | Global key-value config                                   |
| `projects`       | Project metadata + per-project training/augment/AL config |
| `training_runs`  | Training run records with metrics                         |
| `epoch_metrics`  | Per-epoch loss, mAP, precision, recall                    |
| `model_versions` | Model registry with version stages                        |
| `al_cycles`      | Active learning cycle tracking                            |
| `predictions`    | Cached predictions with uncertainty scores                |

---

## API Endpoints

### Core

| Method     | Endpoint                     | Description          |
| ---------- | ---------------------------- | -------------------- |
| `GET`      | `/api/images`                | List images          |
| `POST`     | `/api/images/upload`         | Upload images        |
| `GET/POST` | `/api/labels/{name}`         | Get/save annotations |
| `GET/POST` | `/api/projects`              | List/create projects |
| `POST`     | `/api/model/train`           | Start training       |
| `GET`      | `/api/model/training-status` | Training progress    |
| `POST`     | `/api/predict/unlabeled`     | Run auto-predictions |
| `GET`      | `/api/predict/queue`         | Smart queue          |

### MLOps (v2.0)

| Method    | Endpoint                         | Description           |
| --------- | -------------------------------- | --------------------- |
| `GET`     | `/api/mlops/dashboard`           | Dashboard KPIs        |
| `GET`     | `/api/mlops/runs`                | List training runs    |
| `GET`     | `/api/mlops/runs/{id}`           | Run detail + epochs   |
| `POST`    | `/api/mlops/runs/compare`        | Compare runs          |
| `GET`     | `/api/mlops/models`              | Model versions        |
| `POST`    | `/api/mlops/models/promote/{id}` | Promote to production |
| `GET`     | `/api/mlops/al/cycles`           | AL cycle history      |
| `GET/PUT` | `/api/mlops/config`              | Per-project config    |

---

## Tech Stack

| Layer    | Technology                       |
| -------- | -------------------------------- |
| Backend  | FastAPI 0.115, Python 3.10+      |
| AI/ML    | Ultralytics YOLO (≥8.3), PyTorch |
| Database | SQLite 3 (WAL mode)              |
| Frontend | Vanilla JS, HTML5 Canvas         |
| Styling  | CSS (dark mode)                  |

---

## License

MIT
