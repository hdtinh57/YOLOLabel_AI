"""Playground service for testing models."""
import time
import shutil
from pathlib import Path

from backend.config import settings
from backend.database import get_connection
from backend.services.model_registry_service import ModelRegistryService
from backend.services.model_service import ModelService

class PlaygroundService:
    """Service to handle Playground model inference and logging."""
    
    _model_cache = {}

    @classmethod
    def get_cached_model(cls, version_id: int):
        """Get model from registry. Reuses production model if available and in production stage."""
        from ultralytics import YOLO

        # Fetch model from registry
        with get_connection() as conn:
            row = conn.execute(
                """SELECT mv.*, tr.run_dir 
                   FROM model_versions mv
                   JOIN training_runs tr ON mv.run_id = tr.id
                   WHERE mv.id = ?""",
                (version_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Version {version_id} not found")

        # 1. Reuse Production model if requested version is 'production'
        if row["stage"] == "production":
            prod_model = ModelService.get_model()
            if prod_model:
                return prod_model, row

        # 2. Check local dict cache
        if version_id in cls._model_cache:
            return cls._model_cache[version_id], row

        # 3. Load from disk and cache
        weights_path = Path(row["run_dir"]) / "weights" / "best.pt"
        if not weights_path.exists():
             raise FileNotFoundError(f"Model weights not found at {weights_path}")

        print(f"[Playground] Loading and caching model: {weights_path}")
        model = YOLO(weights_path)
        cls._model_cache[version_id] = model
        
        return model, row

    @classmethod
    def run_inference(cls, input_path: Path, model_version_id: int, conf: float, iou: float, imgsz: int, max_det: int, agnostic_nms: bool) -> dict:
        """Run YOLO inference on an image or video, save result, and log to DB."""
        start_time = time.time() * 1000
        
        model, version_info = cls.get_cached_model(model_version_id)

        session_id = f"sess_{int(time.time())}"
        output_dir = settings.data_dir / "playground_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Playground] Running inference on {input_path} (imgsz={imgsz}, conf={conf}, iou={iou}, max_det={max_det}, agnostic_nms={agnostic_nms})")
        # Ultralytics natively handles both images and videos
        results = model.predict(
            source=str(input_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
            save=True,
            project=str(output_dir),
            name=session_id,
            exist_ok=True
        )

        # Gather output
        total_detections = 0
        if results and len(results) > 0:
            for res in results:
                if res.boxes is not None:
                     total_detections += len(res.boxes)

        output_session_dir = output_dir / session_id
        
        # Determine output file path. YOLO keeps the source filename.
        # But for videos, it might convert to mp4 or avi.
        output_file_name = input_path.name
        possible_outputs = list(output_session_dir.glob(f"{input_path.stem}.*"))
        
        if possible_outputs:
             output_file_path = possible_outputs[0]
             # Return relative path for frontend mounting
             rel_path = f"/playground-results/{session_id}/{output_file_path.name}"
        else:
             rel_path = "" # Error

        file_type = "video" if input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"] else "image"
        duration_ms = (time.time() * 1000) - start_time

        # Cleanup input file
        try:
             input_path.unlink(missing_ok=True)
        except Exception as e:
             print(f"[Playground] Failed to delete temp input: {e}")

        # Log session
        try:
             with get_connection() as conn:
                 conn.execute(
                     """INSERT INTO inference_sessions 
                        (project_id, model_version_id, file_type, input_file, output_file, total_detections, created_at, duration_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                     (version_info["project"], model_version_id, file_type, input_path.name, rel_path, total_detections, time.time(), duration_ms)
                 )
                 conn.execute(
                     """UPDATE model_versions SET notes = notes || '\nPlayground test: ' || ? || ' detections' WHERE id = ?""",
                     (total_detections, model_version_id,)
                 )
                 conn.commit()
        except Exception as e:
             print(f"[Playground] DB Logging failed: {e}")

        return {
            "session_id": session_id,
            "url": rel_path,
            "total_detections": total_detections,
            "duration_ms": duration_ms
        }
