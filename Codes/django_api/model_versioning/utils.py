import os
import shutil
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# ---------- HuggingFacePublisher (unchanged) ----------
class HuggingFacePublisher:
    def __init__(self, model_version, token=None, username=None):
        self.version = model_version
        self.token = token or getattr(settings, 'HUGGINGFACE_TOKEN', None)
        self.username = username or getattr(settings, 'HUGGINGFACE_USERNAME', None)
        if not self.token or not self.username:
            raise ValueError("HUGGINGFACE_TOKEN and HUGGINGFACE_USERNAME must be set in settings")

    def publish(self):
        # Mock publish for testing; real implementation would upload to Hugging Face Hub
        repo = f"{self.username}/mock-model"
        self.version.is_published_to_hub = True
        self.version.hub_repo_url = f"https://huggingface.co/{repo}"
        self.version.save(update_fields=['is_published_to_hub', 'hub_repo_url'])
        return self.version.hub_repo_url


# ---------- Cleanup Function ----------
def cleanup_old_versions(run, keep_best_k=3, keep_version_tags=None):
    """
    Keep only the best K versions (by accuracy) and any tagged versions.
    Delete the rest, freeing storage space.
    """
    if keep_version_tags is None:
        keep_version_tags = []

    versions = run.model_versions.all()
    # Always keep tagged versions
    protected = [v for v in versions if any(v.version.startswith(tag) for tag in keep_version_tags)]
    # Sort remaining by accuracy descending
    remaining = [v for v in versions if v not in protected]
    sorted_by_metric = sorted(remaining, key=lambda v: v.metrics.get('eval_accuracy', 0), reverse=True)
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