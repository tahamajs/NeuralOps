import os, torch
from .models import ModelVersion

class ModelVersionService:
    @staticmethod
    def create_version(run, model_state, metrics, version=None):
        latest = run.model_versions.order_by('-version_number').first()
        new_num = (latest.version_number + 1) if latest else 1
        version_str = version or f"v{new_num}"
        artifact_path = f"/models/{run.id}/v{new_num}/"
        os.makedirs(artifact_path, exist_ok=True)
        torch.save(model_state, os.path.join(artifact_path, "model.pth"))
        size_mb = os.path.getsize(os.path.join(artifact_path, "model.pth")) / 1e6
        return ModelVersion.objects.create(
            run=run,
            version=version_str,
            version_number=new_num,
            artifact_path=artifact_path,
            artifact_size_mb=size_mb,
            metrics=metrics,
            git_commit=run.git_commit_hash,
            config_snapshot_id=run.config_versions.last().id if run.config_versions.exists() else None,
        )