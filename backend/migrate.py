"""Legacy data migration — JSON configs + old training dirs → SQLite."""
import csv
import json
import shutil
import time
from pathlib import Path

from backend.database import get_connection


def migrate_legacy(data_dir: Path, project_root: Path) -> dict:
    """Run one-time migration from legacy data to SQLite.

    Returns summary of what was migrated.
    """
    summary = {"projects": 0, "runs": 0, "settings": 0, "weights_moved": 0}
    projects_dir = data_dir / "projects"

    # Check if migration already done
    with get_connection() as conn:
        done = conn.execute(
            "SELECT value FROM settings WHERE key = 'migration_done'"
        ).fetchone()
        if done:
            return {"skipped": True, "reason": "Migration already completed"}

    # ──────────────────────────────────────────────
    #  Step 1: Migrate global config.json → settings table
    # ──────────────────────────────────────────────
    global_config_path = data_dir / "config.json"
    if global_config_path.exists():
        try:
            gdata = json.loads(global_config_path.read_text())
            with get_connection() as conn:
                for key, value in gdata.items():
                    conn.execute(
                        "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                        (key, json.dumps(value)),
                    )
                conn.commit()
            summary["settings"] += 1
        except Exception as e:
            print(f"[migrate] Warning: Could not migrate global config: {e}")

    # ──────────────────────────────────────────────
    #  Step 2: Migrate per-project config.json → projects table
    # ──────────────────────────────────────────────
    if projects_dir.exists():
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_name = project_dir.name
            project_config_path = project_dir / "config.json"

            training_config = {}
            augment_config = {}
            al_config = {}

            if project_config_path.exists():
                try:
                    pdata = json.loads(project_config_path.read_text())

                    # Separate config into categories
                    training_keys = {
                        "default_model", "default_epochs", "default_batch",
                        "default_imgsz", "default_patience", "device",
                        "optimizer", "lr0", "lrf", "momentum",
                        "weight_decay", "warmup_epochs", "warmup_momentum",
                        "box", "cls", "dfl",
                    }
                    augment_keys = {"fliplr", "mosaic", "mixup", "copy_paste"}
                    al_keys = {"auto_train_threshold", "prediction_confidence"}

                    for k, v in pdata.items():
                        # Map old names to new names
                        mapped_key = k.replace("default_", "") if k.startswith("default_") else k
                        if k in training_keys:
                            training_config[mapped_key] = v
                        elif k in augment_keys:
                            augment_config[k] = v
                        elif k in al_keys:
                            al_config[k] = v
                except Exception as e:
                    print(f"[migrate] Warning: Could not parse {project_config_path}: {e}")

            # Insert project
            with get_connection() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO projects
                       (name, created_at, training_config, augment_config, al_config)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        project_name,
                        project_dir.stat().st_ctime,
                        json.dumps(training_config) if training_config else "{}",
                        json.dumps(augment_config) if augment_config else "{}",
                        json.dumps(al_config) if al_config else "{}",
                    ),
                )
                conn.commit()
            summary["projects"] += 1

            # ──────────────────────────────────────
            #  Step 3: Migrate training runs (models/ or runs/)
            # ──────────────────────────────────────
            models_dir = project_dir / "models"
            if models_dir.exists():
                for run_dir in models_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    migrated = _migrate_training_run(project_name, run_dir)
                    if migrated:
                        summary["runs"] += 1

    # ──────────────────────────────────────────────
    #  Step 4: Move pretrained weights to weights/
    # ──────────────────────────────────────────────
    weights_dir = project_root / "weights"
    weights_dir.mkdir(exist_ok=True)
    for pt_file in project_root.glob("*.pt"):
        # Only move pretrained-looking files (yolo*.pt)
        if pt_file.stem.startswith("yolo"):
            dest = weights_dir / pt_file.name
            if not dest.exists():
                shutil.move(str(pt_file), str(dest))
                summary["weights_moved"] += 1

    # Also check root models/ dir
    root_models = project_root / "models"
    if root_models.exists():
        for run_dir in root_models.iterdir():
            if run_dir.is_dir():
                # Try to find which project this belongs to
                migrated = _migrate_training_run("default", run_dir)
                if migrated:
                    summary["runs"] += 1

    # Mark migration as done
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            ("migration_done", json.dumps(True)),
        )
        conn.commit()

    print(f"[migrate] Migration complete: {summary}")
    return summary


def _migrate_training_run(project: str, run_dir: Path) -> bool:
    """Migrate a single training run directory into SQLite."""
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"

    if not results_csv.exists():
        return False

    # Generate run_id from directory name or timestamp
    run_id = f"legacy_{run_dir.name}_{int(run_dir.stat().st_mtime)}"

    # Ensure project exists in DB
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT name FROM projects WHERE name = ?", (project,)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO projects (name, created_at) VALUES (?, ?)",
                (project, time.time()),
            )
            conn.commit()

    # Parse args.yaml for hyperparams
    hyperparams = {}
    augmentation = {}
    base_model = "unknown"
    epochs_total = 0

    if args_yaml.exists():
        try:
            import yaml
            args = yaml.safe_load(args_yaml.read_text())
            if args:
                base_model = args.get("model", "unknown")
                epochs_total = args.get("epochs", 0)
                training_keys = {
                    "lr0", "lrf", "momentum", "weight_decay", "optimizer",
                    "batch", "imgsz", "patience", "device",
                    "warmup_epochs", "warmup_momentum", "box", "cls", "dfl",
                }
                augment_keys = {"fliplr", "mosaic", "mixup", "copy_paste"}
                for k, v in args.items():
                    if k in training_keys:
                        hyperparams[k] = v
                    elif k in augment_keys:
                        augmentation[k] = v
        except Exception:
            pass

    # Parse results.csv for epoch metrics
    epoch_data = []
    try:
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    k = k.strip()
                    try:
                        cleaned[k] = float(v.strip())
                    except (ValueError, AttributeError):
                        pass
                epoch_data.append(cleaned)
    except Exception:
        pass

    # Determine final metrics from last epoch
    final_metrics = {}
    if epoch_data:
        last = epoch_data[-1]
        final_metrics = {
            "mAP50": last.get("metrics/mAP50(B)"),
            "mAP50_95": last.get("metrics/mAP50-95(B)"),
            "precision": last.get("metrics/precision(B)"),
            "recall": last.get("metrics/recall(B)"),
        }

    # Insert run
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO training_runs
                   (id, project, started_at, finished_at, duration_sec, status,
                    base_model, epochs_total, epochs_done,
                    best_mAP50, best_mAP50_95, best_precision, best_recall,
                    hyperparams, augmentation, run_dir, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, project,
                    run_dir.stat().st_mtime - 60,  # approximate started_at
                    run_dir.stat().st_mtime,
                    None,
                    "completed",
                    base_model, epochs_total, len(epoch_data),
                    final_metrics.get("mAP50"),
                    final_metrics.get("mAP50_95"),
                    final_metrics.get("precision"),
                    final_metrics.get("recall"),
                    json.dumps(hyperparams),
                    json.dumps(augmentation),
                    str(run_dir),
                    "Migrated from legacy data",
                ),
            )

            # Insert epoch metrics
            for i, ed in enumerate(epoch_data):
                conn.execute(
                    """INSERT OR IGNORE INTO epoch_metrics
                       (run_id, epoch,
                        train_box_loss, train_cls_loss, train_dfl_loss,
                        val_box_loss, val_cls_loss, val_dfl_loss,
                        precision_b, recall_b, mAP50, mAP50_95,
                        lr_pg0, lr_pg1, lr_pg2)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id, i,
                        ed.get("train/box_loss"),
                        ed.get("train/cls_loss"),
                        ed.get("train/dfl_loss"),
                        ed.get("val/box_loss"),
                        ed.get("val/cls_loss"),
                        ed.get("val/dfl_loss"),
                        ed.get("metrics/precision(B)"),
                        ed.get("metrics/recall(B)"),
                        ed.get("metrics/mAP50(B)"),
                        ed.get("metrics/mAP50-95(B)"),
                        ed.get("lr/pg0"),
                        ed.get("lr/pg1"),
                        ed.get("lr/pg2"),
                    ),
                )
            conn.commit()
        return True
    except Exception as e:
        print(f"[migrate] Error migrating {run_dir}: {e}")
        return False
