import logging
import os
import shutil
import tarfile
import tempfile

from django.conf import settings

from .models import ModelVersion

logger = logging.getLogger(__name__)


def create_version(
    run,
    metrics,
    version_name=None,
    artifact_dir=None,
    compression="gzip",
    compression_config=None,
):
    """
    Create a new model version, optionally compressing the artifact.

    - artifact_dir: path to the directory containing model files.
    - compression: 'none', 'gzip', or a compressor key from COMPRESSOR_REGISTRY.
    """
    # Determine version number
    last = run.model_versions.order_by("-version_number").first()
    new_number = (last.version_number + 1) if last else 1
    version_str = version_name or f"v{new_number}"

    # Determine artifact path (should already exist from training)
    if artifact_dir is None:
        artifact_dir = os.path.join(settings.MEDIA_ROOT, str(run.id), "model")
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    # Compute uncompressed size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(artifact_dir)
        for filename in filenames
    )
    size_mb = round(total_size / (1024 * 1024), 2)

    # Prepare version record
    version = ModelVersion(
        run=run,
        version=version_str,
        version_number=new_number,
        artifact_path=artifact_dir,
        metrics=metrics,
        artifact_size_mb=size_mb,
    )

    # Apply compression
    if compression == "none":
        version.compression_method = "none"
    elif compression == "gzip":
        compressed_path = _compress_gzip(artifact_dir, run.id, new_number)
        version.compressed_artifact_path = compressed_path
        version.compression_method = "gzip"
        version.compressed_size_mb = round(
            os.path.getsize(compressed_path) / (1024 * 1024), 2
        )
    else:
        # Use a framework compressor
        compressed_path = _compress_with_framework(
            artifact_dir, compression, compression_config
        )
        if compressed_path:
            version.compressed_artifact_path = compressed_path
            version.compression_method = compression
            version.compression_config = compression_config or {}
            version.compressed_size_mb = round(
                os.path.getsize(compressed_path) / (1024 * 1024), 2
            )
        else:
            # fallback to no compression
            version.compression_method = "none"

    version.save()
    return version


def load_model(version: ModelVersion):
    """Load a model from the version artifact, decompressing if necessary."""
    if version.compression_method == "none":
        return _load_from_dir(version.artifact_path)
    elif version.compression_method == "gzip":
        with tempfile.TemporaryDirectory() as tmpdir:
            _decompress_gzip(version.compressed_artifact_path, tmpdir)
            return _load_from_dir(tmpdir)
    else:
        # Framework compressor: decompress using custom logic
        with tempfile.TemporaryDirectory() as tmpdir:
            _decompress_framework(version, tmpdir)
            return _load_from_dir(tmpdir)


def _load_from_dir(dir_path):
    """Load a HuggingFace model from a directory."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(dir_path)


def _compress_gzip(artifact_dir, run_id, version_number):
    """Create a tar.gz file and return its path."""
    output_dir = os.path.join(settings.MEDIA_ROOT, str(run_id), "compressed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"v{version_number}.tar.gz")
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(artifact_dir, arcname=os.path.basename(artifact_dir))
    return output_file


def _decompress_gzip(compressed_path, dest_dir):
    with tarfile.open(compressed_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def _compress_with_framework(artifact_dir, method, config):
    """Apply a registered compressor to the model's state dict and save a new artifact."""
    try:
        from framework.core.registry import COMPRESSOR_REGISTRY

        compressor_cls = COMPRESSOR_REGISTRY.get(method)
        if not compressor_cls:
            logger.warning(f"Compressor '{method}' not found, falling back to none.")
            return None

        # Load model from artifact_dir
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
        # Instantiate compressor (it expects the model and kwargs)
        compressor = compressor_cls(model, **config)
        compressed_model, _ = compressor.compress()

        # Save compressed model to a new temp directory
        output_dir = os.path.join(
            settings.MEDIA_ROOT, str(model.run_id), "compressed", f"v{version_number}"
        )
        os.makedirs(output_dir, exist_ok=True)
        compressed_model.save_pretrained(output_dir)

        # Compute size and return path
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(output_dir)
            for filename in filenames
        )
        return output_dir
    except Exception as e:
        logger.error(f"Framework compression failed: {e}")
        return None


def _decompress_framework(version, dest_dir):
    """For framework compressors, the model is already saved as loadable, so just copy."""
    if os.path.isdir(version.compressed_artifact_path):
        shutil.copytree(version.compressed_artifact_path, dest_dir, dirs_exist_ok=True)
    else:
        raise ValueError("Framework-compressed artifact is not a directory")


# model_versioning/services.py (additions)

import torch

from .models import ModelVersion, ModelVersionDiff


def create_diff(
    from_version: ModelVersion, to_version: ModelVersion
) -> ModelVersionDiff:
    """
    Create a diff artifact between two versions.
    The diff contains only the weight differences (state_dict), not the full model.
    """
    # Load both models
    from_model = load_model(from_version)
    to_model = load_model(to_version)

    diff_state = {}
    from_state = from_model.state_dict()
    to_state = to_model.state_dict()
    for key in to_state:
        if key in from_state:
            diff_state[key] = to_state[key] - from_state[key]
        else:
            diff_state[key] = to_state[key]  # new parameter
    # Save diff
    diff_dir = tempfile.mkdtemp()
    diff_path = os.path.join(diff_dir, "diff.pt")
    torch.save(diff_state, diff_path)

    # Compress diff directory into a tar.gz
    diff_artifact_path = os.path.join(
        settings.MEDIA_ROOT,
        str(from_version.run.id),
        "diffs",
        f"diff_v{from_version.version_number}_v{to_version.version_number}.tar.gz",
    )
    os.makedirs(os.path.dirname(diff_artifact_path), exist_ok=True)
    with tarfile.open(diff_artifact_path, "w:gz") as tar:
        tar.add(diff_path, arcname="diff.pt")
    diff_size_mb = os.path.getsize(diff_artifact_path) / (1024 * 1024)

    # Estimate savings: full to_version size minus diff size
    original_saved = to_version.artifact_size_mb - diff_size_mb

    diff = ModelVersionDiff.objects.create(
        from_version=from_version,
        to_version=to_version,
        diff_artifact=diff_artifact_path,
        diff_size_mb=diff_size_mb,
        original_size_saved_mb=max(0, original_saved),
    )
    shutil.rmtree(diff_dir)
    return diff


def load_version_from_diff(base_version: ModelVersion, diff: ModelVersionDiff):
    """Reconstruct a model by applying a diff to the base version."""
    base_model = load_model(base_version)
    # Load diff state dict
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(diff.diff_artifact, "r:gz") as tar:
            tar.extractall(tmpdir)
        diff_state = torch.load(os.path.join(tmpdir, "diff.pt"))
    base_state = base_model.state_dict()
    for key, delta in diff_state.items():
        if key in base_state:
            base_state[key] += delta
        else:
            base_state[key] = delta
    base_model.load_state_dict(base_state)
    return base_model


def auto_select_compression(metrics: dict, size_mb: float) -> tuple:
    """
    Choose the best compression method based on final metrics and model size.
    Returns (method, config) tuple.
    """
    accuracy = metrics.get("eval_accuracy", 0)
    if accuracy > 0.9 and size_mb < 200:
        return "gzip", {}
    elif size_mb > 500:
        return "a3", {"a3_rank": 32}
    else:
        return "kmeans", {"kmeans_clusters": 256}


def migrate_storage_tier(version: ModelVersion):
    """Automatically move version to colder tier based on age."""
    from datetime import timedelta

    from django.utils import timezone

    age = timezone.now() - version.created_at
    if age > timedelta(days=30) and version.storage_tier == "hot":
        # Move to warm (cloud) – implement using boto3 copy to S3 IA
        bucket = getattr(settings, "MODEL_WARM_BUCKET", None)
        if bucket:
            version.archive_to_cold_storage(bucket, storage_class="STANDARD_IA")
            version.storage_tier = "warm"
            version.save(update_fields=["storage_tier"])
    elif age > timedelta(days=90) and version.storage_tier == "warm":
        bucket = getattr(settings, "MODEL_COLD_BUCKET", None)
        if bucket:
            version.archive_to_cold_storage(bucket, storage_class="GLACIER")
            version.storage_tier = "cold"
            version.save(update_fields=["storage_tier"])
