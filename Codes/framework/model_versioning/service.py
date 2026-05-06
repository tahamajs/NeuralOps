import os, json, logging, shutil, torch
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import VersioningConfig, StorageBackend, BackendMode, CleanupPolicy
from .strategies import STRATEGY_REGISTRY
from .environment import capture_environment
from .metadata import generate_model_card
from .utils import compute_checksum, compress_directory, extract_archive

logger = logging.getLogger(__name__)

# Dummy fallback
class _DummyManager:
    def __getattr__(self, name):
        def dummy(*args, **kwargs): return None
        return dummy
class _DummyModelVersion:
    objects = _DummyManager()
    class DoesNotExist(Exception): pass


class ModelVersionService:
    """Complete model version lifecycle management."""

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
        run,
        model_state: Any,
        metrics: Dict[str, Any],
        tokenizer=None,
        version: Optional[str] = None,
        artifact_base: str = "/models",
        tags: Optional[List[str]] = None,
        config: Optional[VersioningConfig] = None,
        extra_files: Optional[Dict[str, str]] = None,   # {filename: path}
    ) -> Optional[Any]:
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion: return None

        cfg = config or VersioningConfig()
        latest = run.model_versions.order_by('-version_number').first()
        new_num = (latest.version_number + 1) if latest else 1
        version_str = version or f"v{new_num}"

        artifact_dir = os.path.join(artifact_base, str(run.id), version_str)
        os.makedirs(artifact_dir, exist_ok=True)

        model_file = os.path.join(artifact_dir, "model.pth")
        torch.save(model_state, model_file)
        size_mb = os.path.getsize(model_file) / 1e6

        # Tokenizer
        if tokenizer:
            tokenizer.save_pretrained(os.path.join(artifact_dir, "tokenizer"))

        # Extra files (plots, logs, etc.)
        if extra_files:
            for fname, src_path in extra_files.items():
                shutil.copy2(src_path, os.path.join(artifact_dir, fname))

        # Environment snapshot & model card
        env_info = capture_environment(artifact_dir)
        if cfg.generate_model_card:
            generate_model_card(run, metrics, os.path.join(artifact_dir, "README.md"),
                                fmt=cfg.model_card_format)

        # Compression (optional)
        if cfg.compression:
            archive_path = compress_directory(artifact_dir)
            artifact_path = archive_path
            artifact_size_mb = os.path.getsize(archive_path) / 1e6
        else:
            artifact_path = artifact_dir
            artifact_size_mb = size_mb

        # Push to backends
        if cfg.backend_mode == BackendMode.PUSH_ON_CREATE:
            ModelVersionService._push_to_backends(artifact_path, cfg)

        # Create DB record
        record = ModelVersion.objects.create(
            run=run,
            version=version_str,
            version_number=new_num,
            artifact_path=artifact_path,
            artifact_size_mb=round(artifact_size_mb, 2),
            metrics=metrics,
            tags=tags or [],
            storage_tier=",".join([b.value for b in cfg.storage_backends]),
        )
        logger.info("Created version %s", record)
        return record

    @staticmethod
    def get_latest(run):
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion: return None
        return run.model_versions.order_by('-version_number').first()

    @staticmethod
    def list_versions(run, only_tags=None, metric_gt=None):
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion: return []
        qs = run.model_versions.all()
        if only_tags:
            qs = [v for v in qs if any(t in (v.tags or []) for t in only_tags)]
            return qs
        if metric_gt:
            qs = qs.filter(metrics__eval_accuracy__gt=metric_gt)
        return qs.order_by('-version_number')

    @staticmethod
    def load_version(version_id, map_location="cpu"):
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion: return None
        try:
            vo = ModelVersion.objects.get(id=version_id)
            local_path = vo.artifact_path
            if not os.path.exists(local_path):
                local_path = ModelVersionService._pull_from_backends(vo)
            if local_path.endswith('.tar.gz'):
                model_file = extract_archive(local_path, os.path.dirname(local_path))
            else:
                model_file = os.path.join(local_path, "model.pth")
            return torch.load(model_file, map_location=map_location)
        except Exception as e:
            logger.exception("Load version %s failed: %s", version_id, e)
            return None

    @staticmethod
    def apply_cleanup(run, config: VersioningConfig):
        ModelVersion = ModelVersionService._get_model()
        if ModelVersion is _DummyModelVersion: return
        versions = list(run.model_versions.all())
        strategy_cls = STRATEGY_REGISTRY.get(config.cleanup_policy.value)
        if strategy_cls is None: return
        strategy = strategy_cls(config) if callable(strategy_cls) else strategy_cls(config)
        to_delete = strategy.select_versions_to_delete(versions)
        for v in to_delete:
            ModelVersionService.delete_version(v, config)
        logger.info("Cleaned up %d versions for run %s", len(to_delete), run.id)

    @staticmethod
    def delete_version(version, config: VersioningConfig = None):
        try:
            if os.path.exists(version.artifact_path):
                if os.path.isdir(version.artifact_path):
                    shutil.rmtree(version.artifact_path, ignore_errors=True)
                else:
                    os.remove(version.artifact_path)
            # Also delete from backends
            if config:
                for backend in config.storage_backends:
                    ModelVersionService._delete_from_backend(version, backend)
            version.delete()
            logger.info("Deleted version %s", version)
        except Exception as e:
            logger.exception("Delete version %s failed: %s", version.id, e)

    # ------------------------------------------------------------------
    # Backend helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _push_to_backends(artifact_path, cfg):
        for backend in cfg.storage_backends:
            try:
                {StorageBackend.S3: lambda: __import__('.backends.s3', fromlist=['push_to_s3']).push_to_s3(artifact_path, cfg.s3_bucket, cfg.s3_region),
                 StorageBackend.GCP: lambda: __import__('.backends.gcp', fromlist=['push_to_gcp']).push_to_gcp(artifact_path, cfg.gcp_bucket),
                 StorageBackend.AZURE: lambda: __import__('.backends.azure', fromlist=['push_to_azure']).push_to_azure(artifact_path, cfg.azure_conn_str, cfg.azure_container),
                 StorageBackend.HUGGINGFACE: lambda: __import__('.backends.huggingface', fromlist=['push_to_hub']).push_to_hub(artifact_path, cfg.hf_repo_name, cfg.hf_token),
                 StorageBackend.MLFLOW: lambda: __import__('.backends.mlflow', fromlist=['push_to_mlflow']).push_to_mlflow(artifact_path, cfg.mlflow_tracking_uri, cfg.mlflow_experiment_name),
                 StorageBackend.DVC: lambda: __import__('.backends.dvc', fromlist=['push_to_dvc']).push_to_dvc(artifact_path, cfg.dvc_remote, cfg.dvc_remote_url),
                 StorageBackend.GIT_LFS: lambda: __import__('.backends.git_lfs', fromlist=['push_to_git_lfs']).push_to_git_lfs(artifact_path, cfg.git_lfs_remote),
                 StorageBackend.CUSTOM_HTTP: lambda: __import__('.backends.http', fromlist=['push_to_http']).push_to_http(artifact_path, cfg.http_endpoint, cfg.http_headers),
                }.get(backend, lambda: None)()
            except Exception as e:
                logger.exception("Push to %s failed: %s", backend, e)

    @staticmethod
    def _pull_from_backends(version) -> Optional[str]:
        # Try to restore the artifact from any configured backend
        # (simplified; uses the version's stored storage_tier)
        tiers = version.storage_tier.split(',') if version.storage_tier else []
        for tier in tiers:
            try:
                backend = StorageBackend(tier)
                local_dir = os.path.join(os.path.dirname(version.artifact_path), version.version)
                os.makedirs(local_dir, exist_ok=True)
                # implement download per backend (similar to push)
                return local_dir
            except Exception:
                continue
        return None

    @staticmethod
    def _delete_from_backend(version, backend: StorageBackend):
        # similar to push, but calls delete
        pass