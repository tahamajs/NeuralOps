import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
class StorageBackend(Enum):
    LOCAL = "local"
    S3 = "s3"
    HUGGINGFACE = "huggingface"


class CleanupPolicy(Enum):
    KEEP_BEST_K = "keep_best_k"
    KEEP_TAGGED = "keep_tagged"
    TIME_BASED = "time_based"
    NONE = "none"


@dataclass
class VersioningConfig:
    """Controls how versions are created and managed."""

    keep_best_k: int = 5
    keep_tags: List[str] = field(default_factory=lambda: ["production", "staging"])
    max_versions: Optional[int] = None
    storage_backend: StorageBackend = StorageBackend.LOCAL
    cleanup_policy: CleanupPolicy = CleanupPolicy.KEEP_BEST_K
    compression: bool = False
    save_optimizer_state: bool = False
    upload_to_hub: bool = False
    hub_repo_prefix: str = ""
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None


# ----------------------------------------------------------------------
# Dummy fallback (unchanged)
# ----------------------------------------------------------------------
class _DummyManager:
    def __getattr__(self, name):
        def dummy(*args, **kwargs) -> None:
            """Dummy."""
            return None

        return dummy


class _DummyModelVersion:
    objects = _DummyManager()


# ----------------------------------------------------------------------
# Strategies (pluggable)
# ----------------------------------------------------------------------
class VersioningStrategy:
    """Decides which versions to keep / delete."""

    def __init__(self, config: VersioningConfig) -> None:
        self.config = config

    def select_versions_to_delete(self, all_versions) -> None:
        """Select versions to delete."""
        raise NotImplementedError


class BestKStrategy(VersioningStrategy):
    def select_versions_to_delete(self, all_versions) -> Any:
        """Select versions to delete."""
        if not all_versions:
            return []
        sorted_versions = sorted(
            all_versions, key=lambda v: v.metrics.get("eval_accuracy", 0), reverse=True
        )
        return sorted_versions[self.config.keep_best_k :]


class TaggedStrategy(VersioningStrategy):
    def select_versions_to_delete(self, all_versions) -> Any:
        """Select versions to delete."""
        protected = [
            v
            for v in all_versions
            if any(t in (v.tags or []) for t in self.config.keep_tags)
        ]
        return [v for v in all_versions if v not in protected]


class TimeBasedStrategy(VersioningStrategy):
    def select_versions_to_delete(self, all_versions) -> Any:
        """Select versions to delete."""
        import datetime

        cutoff = datetime.datetime.now() - datetime.timedelta(
            days=self.config.keep_best_k
        )
        return [v for v in all_versions if v.created_at < cutoff]


# ----------------------------------------------------------------------
# Real service (extended)
# ----------------------------------------------------------------------
class ModelVersionService:
    """High‑level service for creating, loading, and managing model versions."""

    @staticmethod
    def _get_model() -> Any:
        """ get model."""
        try:
            from model_versioning.models import ModelVersion

            return ModelVersion
        except (ImportError, RuntimeError) as e:
            logger.debug("ModelVersion not available: %s", e)
            return _DummyModelVersion

    @staticmethod
    def create_version(
        run,
        model_state: Any,
        metrics: Dict[str, Any],
        version: Optional[str] = None,
        artifact_base: str = "/models",
        tags: Optional[List[str]] = None,
        config: Optional[VersioningConfig] = None,
    ) -> Optional[Any]:
        """Build and return version."""
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return None

        latest = run.model_versions.order_by("-version_number").first()
        new_num = (latest.version_number + 1) if latest else 1
        version_str = version or f"v{new_num}"

        artifact_path = os.path.join(artifact_base, str(run.id), version_str)
        os.makedirs(artifact_path, exist_ok=True)

        model_file = os.path.join(artifact_path, "model.pth")
        torch.save(model_state, model_file)
        size_mb = os.path.getsize(model_file) / 1e6

        # Handle cloud backends eagerly or lazily
        cfg = config or VersioningConfig()
        if cfg.storage_backend == StorageBackend.S3:
            from .backends.s3 import upload_to_s3

            upload_to_s3(model_file, cfg.s3_bucket, cfg.s3_region)
        elif cfg.storage_backend == StorageBackend.HUGGINGFACE:
            from .backends.huggingface import push_to_hub

            push_to_hub(artifact_path, repo_name=cfg.hub_repo_prefix)

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
        logger.info("Created %s", record)
        return record

    @staticmethod
    def load_version(version_id, map_location="cpu") -> Any:
        """Load version from disk or config."""
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return None
        try:
            version_obj = ModelVersion.objects.get(id=version_id)
            model_file = os.path.join(version_obj.artifact_path, "model.pth")
            if not os.path.exists(model_file):
                if version_obj.storage_tier == StorageBackend.S3.value:
                    from .backends.s3 import download_from_s3

                    download_from_s3(model_file)  # populate local file
                else:
                    raise FileNotFoundError(f"Model file not found: {model_file}")
            # FIX: Enforce weights_only=True for secure model artifact loading
            return torch.load(model_file, map_location=map_location, weights_only=True)
        except ModelVersion.DoesNotExist:
            logger.error("ModelVersion %s not found", version_id)
            return None

    @staticmethod
    def apply_cleanup_policy(run, config: VersioningConfig) -> None:
        """Apply cleanup policy."""
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion:
            return
        versions = list(run.model_versions.all())
        strategy_map = {
            CleanupPolicy.KEEP_BEST_K: BestKStrategy,
            CleanupPolicy.KEEP_TAGGED: TaggedStrategy,
            CleanupPolicy.TIME_BASED: TimeBasedStrategy,
        }
        strategy_cls = strategy_map.get(config.cleanup_policy)
        if strategy_cls is None:
            return
        strategy = strategy_cls(config)
        to_delete = strategy.select_versions_to_delete(versions)
        for version_obj in to_delete:
            if os.path.exists(version_obj.artifact_path):
                import shutil

                shutil.rmtree(version_obj.artifact_path, ignore_errors=True)
            version_obj.delete()
        logger.info("Deleted %d versions for run %s", len(to_delete), run.id)
