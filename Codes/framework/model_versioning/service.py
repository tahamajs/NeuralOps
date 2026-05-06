import os
import json
import logging
import torch
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import VersioningConfig, StorageBackend, CleanupPolicy
from .strategies import BestKStrategy, TaggedStrategy, TimeBasedStrategy
from .environment import capture_environment, generate_model_card

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Dummy fallback (when Django models aren't ready)
# ----------------------------------------------------------------------
class _DummyManager:
    def __getattr__(self, name):
        def dummy(*args, **kwargs):
            return None
        return dummy

class _DummyModelVersion:
    objects = _DummyManager()
    class DoesNotExist(Exception):
        pass


# ----------------------------------------------------------------------
# Main Service
# ----------------------------------------------------------------------
class ModelVersionService:
    """High‑level service for creating, loading, and managing model versions."""

    STRATEGY_MAP = {
        CleanupPolicy.KEEP_BEST_K: BestKStrategy,
        CleanupPolicy.KEEP_TAGGED: TaggedStrategy,
        CleanupPolicy.TIME_BASED: TimeBasedStrategy,
    }

    @staticmethod
    def _get_model():
        try:
            from model_versioning.models import ModelVersion
            return ModelVersion
        except (ImportError, RuntimeError) as e:
            logger.debug("ModelVersion not available: %s", e)
            return _DummyModelVersion

    @staticmethod
    def create_version(
        run,                        # Django Job / TrainingRun
        model_state: Any,
        metrics: Dict[str, Any],
        version: Optional[str] = None,
        artifact_base: str = "/models",
        tags: Optional[List[str]] = None,
        config: Optional[VersioningConfig] = None,
        tokenizer=None,
        model_card_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Save a model checkpoint and create a ModelVersion record."""
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return None

        cfg = config or VersioningConfig()
        latest = run.model_versions.order_by('-version_number').first()
        new_num = (latest.version_number + 1) if latest else 1
        version_str = version or f"v{new_num}"
        artifact_path = os.path.join(artifact_base, str(run.id), version_str)
        os.makedirs(artifact_path, exist_ok=True)

        model_file = os.path.join(artifact_path, "model.pth")
        torch.save(model_state, model_file)
        size_mb = os.path.getsize(model_file) / 1e6

        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(artifact_path, "tokenizer"))

        # Environment capture
        env_info = capture_environment(artifact_path)
        env_info["git_commit"] = getattr(run, 'git_commit_hash', None)
        env_info["metrics"] = metrics

        # Generate model card
        if cfg.generate_model_card:
            card_path = model_card_path or os.path.join(artifact_path, "README.md")
            generate_model_card(run, metrics, card_path)

        # Push to cloud backend
        if cfg.storage_backend == StorageBackend.S3:
            from .backends.s3 import upload_to_s3
            upload_to_s3(model_file, cfg.s3_bucket, cfg.s3_region)
        elif cfg.storage_backend == StorageBackend.GCP:
            from .backends.gcp import upload_to_gcp
            upload_to_gcp(model_file, cfg.gcp_bucket)
        elif cfg.storage_backend == StorageBackend.AZURE:
            from .backends.azure import upload_to_azure
            upload_to_azure(model_file, cfg.azure_conn_str, cfg.azure_container)
        elif cfg.storage_backend == StorageBackend.HUGGINGFACE:
            from .backends.huggingface import push_to_hub
            push_to_hub(artifact_path, repo_name=cfg.hub_repo_prefix or f"model-{run.id}")

        record = ModelVersion.objects.create(
            run=run,
            version=version_str,
            version_number=new_num,
            artifact_path=artifact_path,
            artifact_size_mb=round(size_mb, 2),
            metrics=metrics,
            tags=tags or [],
            storage_tier=cfg.storage_backend.value,
        )
        logger.info("Created ModelVersion %s", record)
        return record

    @staticmethod
    def load_version(version_id, map_location="cpu"):
        """Load a saved model from a ModelVersion record."""
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return None
        try:
            version_obj = ModelVersion.objects.get(id=version_id)
            model_file = os.path.join(version_obj.artifact_path, "model.pth")
            if not os.path.exists(model_file):
                if version_obj.storage_tier == StorageBackend.S3.value:
                    from .backends.s3 import download_from_s3
                    download_from_s3(model_file)   # populates local
                elif version_obj.storage_tier == StorageBackend.GCP.value:
                    from .backends.gcp import download_from_gcp
                    download_from_gcp(model_file)
                elif version_obj.storage_tier == StorageBackend.AZURE.value:
                    from .backends.azure import download_from_azure
                    download_from_azure(model_file)
                else:
                    raise FileNotFoundError(f"Model file not found: {model_file}")
            return torch.load(model_file, map_location=map_location)
        except ModelVersion.DoesNotExist:
            logger.error("ModelVersion %s not found", version_id)
            return None
        except Exception as e:
            logger.exception("Failed to load version %s: %s", version_id, e)
            return None

    @staticmethod
    def apply_cleanup_policy(run, config: VersioningConfig):
        """Delete old versions according to the configured strategy."""
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return
        versions = list(run.model_versions.all())
        strategy_cls = ModelVersionService.STRATEGY_MAP.get(config.cleanup_policy)
        if strategy_cls is None:
            return
        strategy = strategy_cls(config)
        to_delete = strategy.select_versions_to_delete(versions)
        for version_obj in to_delete:
            if os.path.exists(version_obj.artifact_path):
                shutil.rmtree(version_obj.artifact_path, ignore_errors=True)
            # Also delete from cloud if applicable
            if version_obj.storage_tier == StorageBackend.S3.value:
                from .backends.s3 import delete_from_s3
                delete_from_s3(version_obj.artifact_path)
            version_obj.delete()
        logger.info("Deleted %d old versions for run %s", len(to_delete), run.id)