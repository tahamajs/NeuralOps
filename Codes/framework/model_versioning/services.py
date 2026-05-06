import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dummy fallback – used when Django isn't loaded or models aren't ready
# ----------------------------------------------------------------------
class _DummyManager:
    """Mimics a Django model manager – returns None for any lookup."""
    def __getattr__(self, name):
        def dummy(*args, **kwargs):
            logger.warning("Django models not available – operation skipped.")
            return None
        return dummy

class _DummyModelVersion:
    objects = _DummyManager()


# ----------------------------------------------------------------------
# Real service
# ----------------------------------------------------------------------
class ModelVersionService:
    """
    High‑level service for managing ModelVersion objects.

    All methods automatically fall back to a dummy implementation
    when the Django model can't be imported (e.g., during framework
    startup or in standalone scripts).
    """

    @staticmethod
    def _get_model():
        """Return the real ModelVersion class, or a dummy if unavailable."""
        try:
            from model_versioning.models import ModelVersion
            return ModelVersion
        except (ImportError, RuntimeError) as e:
            logger.debug("ModelVersion not available: %s", e)
            return _DummyModelVersion

    @staticmethod
    def create_version(
        run,               # Job instance (must have 'id', 'git_commit_hash', 'config_versions')
        model_state: Any,  # PyTorch state_dict or serialisable object
        metrics: Dict[str, Any],
        version: Optional[str] = None,
        artifact_base: str = "/models"
    ) -> Optional[Any]:
        """
        Save a model checkpoint and create a corresponding ModelVersion record.

        Args:
            run: The training run (Job instance)
            model_state: The model's state_dict (anything torch.save accepts)
            metrics: Dictionary of evaluation metrics
            version: Optional version string (e.g., "v1.0.0"); auto‑generated if omitted
            artifact_base: Base directory for storing artifacts

        Returns:
            The created ModelVersion instance, or None if models are unavailable
        """
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return None

        # Determine next version number
        latest = run.model_versions.order_by('-version_number').first()
        new_num = (latest.version_number + 1) if latest else 1
        version_str = version or f"v{new_num}"

        artifact_path = os.path.join(artifact_base, str(run.id), f"v{new_num}")
        try:
            os.makedirs(artifact_path, exist_ok=True)
        except OSError as e:
            logger.error("Cannot create artifact directory %s: %s", artifact_path, e)
            raise

        # Save the model (torch is imported lazily to avoid early heavy import)
        import torch
        model_file = os.path.join(artifact_path, "model.pth")
        torch.save(model_state, model_file)

        size_mb = os.path.getsize(model_file) / 1e6

        config_snapshot_id = None
        if hasattr(run, 'config_versions') and run.config_versions.exists():
            last_config = run.config_versions.last()
            config_snapshot_id = last_config.id if last_config else None

        return ModelVersion.objects.create(
            run=run,
            version=version_str,
            version_number=new_num,
            artifact_path=artifact_path,
            artifact_size_mb=round(size_mb, 2),
            metrics=metrics,
            git_commit=run.git_commit_hash if hasattr(run, 'git_commit_hash') else None,
            config_snapshot_id=config_snapshot_id,
        )

    @staticmethod
    def load_version(version_id):
        """Load a saved model from a ModelVersion record."""
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return None
        try:
            version = ModelVersion.objects.get(id=version_id)
            model_path = os.path.join(version.artifact_path, "model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            import torch
            return torch.load(model_path, map_location='cpu')
        except ModelVersion.DoesNotExist:
            logger.error("ModelVersion %s not found", version_id)
            return None
        except Exception as e:
            logger.exception("Failed to load model version %s: %s", version_id, e)
            return None

    @staticmethod
    def delete_old_versions(run, keep_latest: int = 5):
        """Remove artifact files and DB records for older versions beyond the keep threshold."""
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return
        versions = run.model_versions.order_by('-version_number')
        to_delete = versions[keep_latest:]
        for version in to_delete:
            if os.path.exists(version.artifact_path):
                import shutil
                shutil.rmtree(version.artifact_path, ignore_errors=True)
            version.delete()
        logger.info("Deleted %d old model versions for run %s", len(to_delete), run.id)