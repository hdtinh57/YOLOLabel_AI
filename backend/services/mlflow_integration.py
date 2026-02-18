"""Optional MLflow integration — auto-detect, configure, and toggle.

Ultralytics has built-in MLflow support via its callback system.
When MLflow is installed and the tracking URI is set, YOLO automatically
logs parameters, metrics, and artifacts to the MLflow server.

This module:
1. Detects whether mlflow is installed
2. Provides enable/disable toggle
3. Configures tracking URI and experiment name
4. Provides status info for the dashboard
"""
import os
import logging

logger = logging.getLogger(__name__)

# Module-level state
_mlflow_available: bool = False
_mlflow_enabled: bool = False
_tracking_uri: str = ""
_experiment_name: str = "yololabel-ai"


def _detect_mlflow() -> bool:
    """Check if mlflow package is importable."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


# Run detection on import
_mlflow_available = _detect_mlflow()


class MLflowIntegration:
    """Manages optional MLflow tracking integration."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if mlflow is installed."""
        return _mlflow_available

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if mlflow tracking is currently enabled."""
        return _mlflow_available and _mlflow_enabled

    @classmethod
    def enable(
        cls,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "yololabel-ai",
    ) -> dict:
        """Enable MLflow tracking.

        Sets environment variables that Ultralytics' built-in MLflow
        callback reads automatically.
        """
        global _mlflow_enabled, _tracking_uri, _experiment_name

        if not _mlflow_available:
            return {
                "enabled": False,
                "error": "mlflow is not installed. Run: pip install mlflow",
            }

        _tracking_uri = tracking_uri
        _experiment_name = experiment_name

        # Ultralytics reads these env vars for MLflow auto-logging
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

        # Verify connection
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            _mlflow_enabled = True

            logger.info(
                "MLflow enabled — URI: %s, Experiment: %s",
                tracking_uri,
                experiment_name,
            )
            return {
                "enabled": True,
                "tracking_uri": tracking_uri,
                "experiment": experiment_name,
            }
        except Exception as e:
            logger.warning("MLflow connection failed: %s", e)
            _mlflow_enabled = False
            return {"enabled": False, "error": str(e)}

    @classmethod
    def disable(cls) -> dict:
        """Disable MLflow tracking."""
        global _mlflow_enabled

        _mlflow_enabled = False

        # Remove env vars so Ultralytics won't auto-log
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)

        logger.info("MLflow disabled")
        return {"enabled": False}

    @classmethod
    def get_status(cls) -> dict:
        """Get current MLflow integration status."""
        status = {
            "available": _mlflow_available,
            "enabled": _mlflow_enabled,
            "tracking_uri": _tracking_uri if _mlflow_enabled else None,
            "experiment": _experiment_name if _mlflow_enabled else None,
        }

        if _mlflow_available:
            try:
                import mlflow

                status["mlflow_version"] = mlflow.__version__
            except Exception:
                pass

        return status

    @classmethod
    def setup_for_run(cls, run_id: str, project: str) -> None:
        """Configure MLflow tags before a training run.

        Called by ModelService before model.train() if enabled.
        Sets run tags for better organization in MLflow UI.
        """
        if not cls.is_enabled():
            return

        try:
            os.environ["MLFLOW_RUN_NAME"] = run_id
            os.environ["MLFLOW_TAGS"] = (
                f'{{"project": "{project}", "run_id": "{run_id}"}}'
            )
        except Exception as e:
            logger.warning("MLflow run setup failed: %s", e)

    @classmethod
    def cleanup_after_run(cls) -> None:
        """Clean up run-specific env vars after training."""
        os.environ.pop("MLFLOW_RUN_NAME", None)
        os.environ.pop("MLFLOW_TAGS", None)
