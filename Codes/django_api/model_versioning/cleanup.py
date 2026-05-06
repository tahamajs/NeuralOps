import os
import shutil
import logging

logger = logging.getLogger(__name__)


def cleanup_old_versions(run, keep_best_k=3, keep_version_tags=None):
    """Keep only the best K versions (by accuracy) and any tagged versions.
       Delete the rest, freeing storage space.
    """
    if keep_version_tags is None:
        keep_version_tags = []

    versions = run.model_versions.all()

    # Always keep versions with important tags
    protected = [v for v in versions if any(v.version.startswith(tag) for tag in keep_version_tags)]

    # Sort remaining by accuracy descending
    remaining = [v for v in versions if v not in protected]
    sorted_by_metric = sorted(
        remaining,
        key=lambda v: v.metrics.get('eval_accuracy', 0),
        reverse=True
    )
    keep = sorted_by_metric[:keep_best_k]

    to_delete = [v for v in versions if v not in protected and v not in keep]

    deleted_count = 0
    for version in to_delete:
        try:
            if version.storage_tier == 'hot' and os.path.exists(version.artifact_path):
                shutil.rmtree(version.artifact_path)
            version.delete()
            deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete model version {version.id}: {e}")

    return deleted_count