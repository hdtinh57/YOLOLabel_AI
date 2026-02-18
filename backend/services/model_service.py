"""Model service — YOLO model lifecycle with MLOps integration."""
import threading
import time
from pathlib import Path
from typing import Any

from backend.config import settings


class ModelService:
    """Manages YOLO model loading, training, and inference."""

    _model: Any = None
    _model_name: str = ""
    _is_training: bool = False
    _training_thread: threading.Thread | None = None
    _training_status: dict = {}
    _custom_model_path: str | None = None
    _stop_signal: bool = False

    # ──────────────────────────────────────────────
    #  Model Loading
    # ──────────────────────────────────────────────

    @classmethod
    def get_model(cls):
        return cls._model

    @classmethod
    def load_model(cls, model_name: str | None = None, custom_path: str | None = None):
        """Load a YOLO model (pretrained or custom checkpoint)."""
        from ultralytics import YOLO

        if custom_path and Path(custom_path).exists():
            cls._model = YOLO(custom_path)
            cls._model_name = Path(custom_path).name
            cls._custom_model_path = custom_path
        else:
            name = model_name or settings.default_model
            cls._model = YOLO(name)
            cls._model_name = name
            cls._custom_model_path = None

        return {
            "model_name": cls._model_name,
            "custom_path": cls._custom_model_path,
            "loaded": True,
        }

    @classmethod
    def auto_load_production_model(cls) -> dict:
        """Load the production model from registry, fallback to default."""
        from backend.services.model_registry_service import ModelRegistryService

        if not settings.active_project:
             result = cls.load_model()
             result["source"] = "pretrained (no project)"
             return result

        weights_path = ModelRegistryService.get_production_weights_path(
            settings.active_project
        )
        if weights_path and Path(weights_path).exists():
            result = cls.load_model(custom_path=weights_path)
            result["source"] = "production_registry"
            return result

        # Fallback: check if any custom best.pt exists in runs/
        if settings.runs_dir.exists():
            for run_dir in sorted(settings.runs_dir.iterdir(), reverse=True):
                best = run_dir / "weights" / "best.pt"
                if best.exists():
                    result = cls.load_model(custom_path=str(best))
                    result["source"] = "latest_run"
                    return result

        # Final fallback: default pretrained
        result = cls.load_model()
        result["source"] = "pretrained"
        return result

    @classmethod
    def get_available_models(cls) -> list[dict]:
        """List available pretrained models + custom checkpoints."""
        models = [{"name": m, "type": "pretrained"} for m in settings.available_models]

        # Scan runs/ for custom checkpoints
        if settings.runs_dir.exists():
            for pt_file in settings.runs_dir.rglob("best.pt"):
                run_name = pt_file.parent.parent.name
                models.append({
                    "name": f"{run_name}/best.pt",
                    "type": "custom",
                    "path": str(pt_file),
                })

        # Scan weights/ for downloaded pretrained
        if settings.weights_dir.exists():
            for pt_file in settings.weights_dir.glob("*.pt"):
                if pt_file.name not in [m["name"] for m in models]:
                    models.append({
                        "name": pt_file.name,
                        "type": "pretrained",
                        "path": str(pt_file),
                    })

        return models

    # ──────────────────────────────────────────────
    #  Inference
    # ──────────────────────────────────────────────

    @classmethod
    def predict(cls, image_path: str, confidence: float = 0.25) -> list[dict]:
        """Run inference on a single image."""
        if cls._model is None:
            cls.auto_load_production_model()

        results = cls._model.predict(
            source=image_path,
            conf=confidence,
            imgsz=settings.default_imgsz,
            save=False,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                xywhn = result.boxes.xywhn.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xywhn)):
                    detections.append({
                        "class_id": int(class_ids[i]),
                        "x_center": float(xywhn[i][0]),
                        "y_center": float(xywhn[i][1]),
                        "width": float(xywhn[i][2]),
                        "height": float(xywhn[i][3]),
                        "confidence": float(confs[i]),
                        "is_predicted": True,
                    })

        return detections

    @classmethod
    def predict_batch(cls, image_paths: list[str], confidence: float = 0.25) -> dict[str, list[dict]]:
        return {Path(p).name: cls.predict(p, confidence) for p in image_paths}

    # ──────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────

    @classmethod
    def stop_training(cls) -> dict:
        if cls._is_training:
            cls._stop_signal = True
            return {"status": "stopping"}
        return {"status": "not_training"}

    @classmethod
    def start_training(service_cls, model_name: str = "yolo26n.pt", **kwargs) -> dict:
        """Start training with SQLite logging."""
        cls = service_cls  # Alias to support existing code using 'cls'
        print(f"[ModelService] start_training called on class ID: {id(cls)}")
        if service_cls._is_training:
            return {"error": "Training already in progress", "is_training": True}

        service_cls._is_training = True
        service_cls._stop_signal = False

        epochs = kwargs.get("epochs", settings.default_epochs)

        # Generate unique run_id
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        cls._training_status = {
            "status": "running",
            "is_training": True,
            "run_id": run_id,
            "current_epoch": 0,
            "total_epochs": epochs,
            "metrics": {},
            "model_path": None,
            "error": None,
        }

        def _train():
            print(f"[ModelService] Training thread started for run {run_id}")
            try:
                # ... (imports) ...
                from ultralytics import YOLO
                from backend.services.training_history_service import TrainingHistoryService
                from backend.services.model_registry_service import ModelRegistryService
                from backend.services.label_service import LabelService

                # Determine which weights to use
                weights_path = model_name
                # If model_name matches the currently loaded custom model, use its full path
                if cls._custom_model_path and Path(cls._custom_model_path).name == model_name:
                    weights_path = cls._custom_model_path
                # Or if model_name looks like a file path that exists, use it
                elif Path(model_name).exists():
                    weights_path = model_name

                print(f"[ModelService] Loading weights from: {weights_path}")
                model = YOLO(weights_path)

                # Ensure dataset.yaml
                yaml_path = settings.project_path / "dataset.yaml"
                if not yaml_path.exists():
                    raise FileNotFoundError("dataset.yaml not found. Export labels first.")

                # ... (snapshot & run record logic) ...
                dataset_snapshot = {
                    "train_images": sum(
                        1 for f in (settings.images_dir / "train").iterdir()
                        if f.suffix.lower() in settings.image_extensions
                    ) if (settings.images_dir / "train").exists() else 0,
                    "val_images": sum(
                        1 for f in (settings.images_dir / "val").iterdir()
                        if f.suffix.lower() in settings.image_extensions
                    ) if (settings.images_dir / "val").exists() else 0,
                    "num_classes": len(LabelService.get_classes()),
                }

                print(f"[ModelService] Creating run record {run_id}")
                TrainingHistoryService.create_run(
                    run_id=run_id,
                    project=settings.active_project,
                    base_model=str(weights_path),  # Log actual path/name
                    epochs_total=epochs,
                    hyperparams=kwargs,
                    augmentation={}, 
                    dataset_snapshot=dataset_snapshot,
                    run_dir=str(settings.runs_dir / run_id),
                )

                # ─── Callback: log each epoch ───
                def on_train_epoch_end(trainer):
                    epoch = trainer.epoch
                    print(f"[ModelService] Epoch {epoch + 1}/{epochs} completed")
                    cls._training_status["current_epoch"] = epoch + 1

                    # Collect metrics from trainer
                    metrics = {}
                    
                    # 1. Validation metrics (mAP, etc.)
                    if hasattr(trainer, "metrics") and trainer.metrics:
                        metrics = {k: float(v) for k, v in trainer.metrics.items()}
                    
                    # 2. Training losses (box, cls, dfl)
                    try:
                        if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                            loss_items = trainer.loss_items
                            loss_names = getattr(trainer, "loss_names", ["box_loss", "cls_loss", "dfl_loss"])
                            for i, name in enumerate(loss_names):
                                val = loss_items[i]
                                val = val.item() if hasattr(val, "item") else float(val)
                                metrics[f"train/{name}"] = val
                        elif hasattr(trainer, "tloss") and trainer.tloss is not None:
                            # Fallback: total loss only
                            tloss = trainer.tloss
                            metrics["train/box_loss"] = tloss.item() if hasattr(tloss, "item") else float(tloss)
                    except Exception as le:
                        print(f"[ModelService] Could not extract training losses: {le}")
                    
                    # Update dashboard status
                    cls._training_status["metrics"] = {
                        "mAP50": metrics.get("metrics/mAP50(B)", 0),
                    }

                    # Log to SQLite
                    try:
                        TrainingHistoryService.log_epoch(run_id, epoch, metrics)
                    except Exception as e:
                        print(f"[ModelService] Failed to log epoch: {e}")

                model.add_callback("on_train_epoch_end", on_train_epoch_end)

                train_args = {
                    "data": str(yaml_path),
                    "project": str(settings.runs_dir),
                    "name": run_id,
                    "exist_ok": True,
                    "save": True,
                    "verbose": True,
                }
                train_args.update(kwargs)

                # ─── Train ───
                print(f"[ModelService] Starting YOLO train loop...")
                results = model.train(**train_args)
                print(f"[ModelService] YOLO train loop finished")

                # ─── Post-training ───
                run_path = settings.runs_dir / run_id
                best_path = run_path / "weights" / "best.pt"

                final_status = "stopped" if cls._stop_signal else "completed"
                TrainingHistoryService.complete_run(run_id, final_status, {})

                cls._training_status.update({
                    "status": final_status,
                    "is_training": False,
                    "current_epoch": cls._training_status["current_epoch"],
                    "model_path": str(best_path) if best_path.exists() else None,
                    "error": "Stopped by user" if cls._stop_signal else None,
                })
                print(f"[ModelService] Training flow completed. Status: {final_status}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ModelService] Training thread error: {e}")
                
                try:
                    from backend.services.training_history_service import TrainingHistoryService
                    TrainingHistoryService.complete_run(run_id, "failed")
                except Exception:
                    pass

                cls._training_status.update({
                    "status": "failed",
                    "is_training": False,
                    "error": str(e),
                })
            finally:
                cls._is_training = False
                cls._stop_signal = False
                print("[ModelService] Training thread finally block executed")

        cls._training_thread = threading.Thread(target=_train, daemon=True)
        cls._training_thread.start()
        return cls._training_status

    @classmethod
    def get_training_status(cls) -> dict:
        status = cls._training_status.copy()
        # Debug log to trace status visibility
        if status.get("is_training"):
            print(f"[ModelService] get_training_status (ID {id(cls)}) returning epoch {status.get('current_epoch')}")
        return status

    @classmethod
    def is_training(cls) -> bool:
        return cls._is_training

    @classmethod
    def get_model_info(cls) -> dict:
        return {
            "model_name": cls._model_name,
            "is_loaded": cls._model is not None,
            "is_training": cls._is_training,
            "custom_path": cls._custom_model_path,
        }
