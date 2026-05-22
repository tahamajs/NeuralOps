import os, shutil, logging
from django.conf import settings

logger = logging.getLogger(__name__)

def cleanup_old_versions(run, keep_best_k=3, keep_version_tags=None):
    if keep_version_tags is None:
        keep_version_tags = []
    versions = run.model_versions.all()
    protected = [v for v in versions if any(v.version.startswith(tag) for tag in keep_version_tags)]
    remaining = [v for v in versions if v not in protected]
    sorted_by_metric = sorted(remaining, key=lambda v: v.metrics.get('eval_accuracy', 0), reverse=True)
    keep = sorted_by_metric[:keep_best_k]
    to_delete = [v for v in versions if v not in protected and v not in keep]
    deleted = 0
    for version in to_delete:
        try:
            version.delete_local_artifact()
            version.delete()
            deleted += 1
        except Exception as e:
            logger.error(f"Failed to delete version {version.id}: {e}")
    return deleted

class HuggingFacePublisher:
    def __init__(self, model_version, token=None, username=None):
        self.version = model_version
        self.token = token or getattr(settings, 'HUGGINGFACE_TOKEN', None)
        self.username = username or getattr(settings, 'HUGGINGFACE_USERNAME', None)
        if not self.token or not self.username:
            raise ValueError("HUGGINGFACE_TOKEN and HUGGINGFACE_USERNAME required")

    def publish(self):
        from huggingface_hub import HfApi, create_repo, upload_folder
        import tempfile
        repo_name = f"{self.username}/{self.version.run.model_name.split('/')[-1]}-v{self.version.version_number}"
        create_repo(repo_name, token=self.token, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self.version.artifact_path
            for item in os.listdir(model_dir):
                src = os.path.join(model_dir, item)
                dst = os.path.join(tmpdir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write(f"# Model {self.version.version}\nMetrics: {self.version.metrics}")
            upload_folder(folder_path=tmpdir, repo_id=repo_name, token=self.token)
        self.version.is_published_to_hub = True
        self.version.hub_repo_url = f"https://huggingface.co/{repo_name}"
        self.version.save(update_fields=['is_published_to_hub', 'hub_repo_url'])
        return self.version.hub_repo_url

