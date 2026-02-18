"""SQLite database manager — single source of truth for YOLOLabel AI."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Database lives at data/yololabel.db
_DB_PATH: Path | None = None
_SCHEMA_VERSION = 1


def get_db_path(data_dir: Path) -> Path:
    return data_dir / "yololabel.db"


def init_db(data_dir: Path) -> None:
    """Initialize database: create tables if not exist."""
    global _DB_PATH
    data_dir.mkdir(parents=True, exist_ok=True)
    _DB_PATH = get_db_path(data_dir)

    with get_connection() as conn:
        conn.executescript(_SCHEMA_SQL)
        # Set schema version
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            ("schema_version", json.dumps(_SCHEMA_VERSION)),
        )
        conn.commit()


@contextmanager
def get_connection():
    """Context manager for SQLite connections with WAL mode and FK support."""
    if _DB_PATH is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def get_setting(key: str, default: Any = None) -> Any:
    """Read a setting from the settings table."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row["value"])
        return default


def set_setting(key: str, value: Any) -> None:
    """Write a setting to the settings table."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    """Convert sqlite3.Row to dict."""
    if row is None:
        return None
    return dict(row)


def rows_to_list(rows: list[sqlite3.Row]) -> list[dict]:
    """Convert list of sqlite3.Row to list of dict."""
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
#  Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Global settings (key-value store)
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Projects (metadata + per-project training defaults)
CREATE TABLE IF NOT EXISTS projects (
    name            TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    training_config TEXT DEFAULT '{}',
    augment_config  TEXT DEFAULT '{}',
    al_config       TEXT DEFAULT '{}'
);

-- Training Runs
CREATE TABLE IF NOT EXISTS training_runs (
    id             TEXT PRIMARY KEY,
    project        TEXT NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    started_at     REAL NOT NULL,
    finished_at    REAL,
    duration_sec   REAL,
    status         TEXT DEFAULT 'running',

    base_model     TEXT NOT NULL,
    epochs_total   INTEGER,
    epochs_done    INTEGER DEFAULT 0,

    best_mAP50     REAL,
    best_mAP50_95  REAL,
    best_precision REAL,
    best_recall    REAL,

    hyperparams    TEXT DEFAULT '{}',
    augmentation   TEXT DEFAULT '{}',

    train_images   INTEGER,
    train_labels   INTEGER,
    val_images     INTEGER,
    val_labels     INTEGER,
    num_classes    INTEGER,
    class_names    TEXT DEFAULT '[]',

    run_dir        TEXT,
    notes          TEXT DEFAULT ''
);

-- Per-epoch metrics (training curves)
CREATE TABLE IF NOT EXISTS epoch_metrics (
    run_id         TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    epoch          INTEGER NOT NULL,
    train_box_loss REAL,
    train_cls_loss REAL,
    train_dfl_loss REAL,
    val_box_loss   REAL,
    val_cls_loss   REAL,
    val_dfl_loss   REAL,
    precision_b    REAL,
    recall_b       REAL,
    mAP50          REAL,
    mAP50_95       REAL,
    lr_pg0         REAL,
    lr_pg1         REAL,
    lr_pg2         REAL,
    PRIMARY KEY (run_id, epoch)
);

-- Model Registry (versioned, staged)
CREATE TABLE IF NOT EXISTS model_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES training_runs(id),
    project     TEXT NOT NULL,
    version     INTEGER NOT NULL,
    stage       TEXT DEFAULT 'none',
    promoted_at REAL,
    mAP50_95    REAL,
    notes       TEXT DEFAULT '',
    UNIQUE(project, version)
);

-- Active Learning Cycles
CREATE TABLE IF NOT EXISTS al_cycles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project          TEXT NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    cycle_num        INTEGER NOT NULL,
    started_at       REAL NOT NULL,
    finished_at      REAL,
    model_used       TEXT,
    images_predicted INTEGER DEFAULT 0,
    images_labeled   INTEGER DEFAULT 0,
    images_accepted  INTEGER DEFAULT 0,
    images_rejected  INTEGER DEFAULT 0,
    run_id           TEXT REFERENCES training_runs(id),
    UNIQUE(project, cycle_num)
);

-- Prediction Cache (persistent)
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT NOT NULL,
    cycle_id    INTEGER REFERENCES al_cycles(id) ON DELETE CASCADE,
    image_name  TEXT NOT NULL,
    split       TEXT DEFAULT 'train',
    uncertainty REAL NOT NULL,
    boxes       TEXT NOT NULL,
    status      TEXT DEFAULT 'pending',
    created_at  REAL NOT NULL,
    UNIQUE(project, image_name, cycle_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_runs_project  ON training_runs(project);
CREATE INDEX IF NOT EXISTS idx_runs_status   ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_epochs_run    ON epoch_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_mv_project    ON model_versions(project);
CREATE INDEX IF NOT EXISTS idx_pred_project  ON predictions(project, status);
CREATE INDEX IF NOT EXISTS idx_cycles_project ON al_cycles(project);
"""
