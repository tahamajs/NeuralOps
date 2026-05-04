from model_versioning.services import ModelVersionService

class ModelVersionService:
    @staticmethod
    def create_version(run, model_state, metrics, version=None):
        """Create a new model version."""
        from model_versioning.models import ModelVersion
        
        latest = run.model_versions.order_by('-version_number').first()
        new_num = (latest.version_number + 1) if latest else 1
        version_str = version or f"v{new_num}"
        
        # Save model to structured path
        artifact_path = f"/models/{run.id}/v{new_num}/"
        torch.save(model_state, f"{artifact_path}model.pth")
        
        # Optional: Generate diff if previous version exists
        if latest:
            diff_path = ModelVersionService._generate_diff(
                latest.artifact_path, artifact_path, run.id, new_num
            )
            # Store diff for storage optimization
        
        return ModelVersion.objects.create(
            run=run,
            version=version_str,
            version_number=new_num,
            artifact_path=artifact_path,
            artifact_size_mb=os.path.getsize(f"{artifact_path}model.pth") / 1e6,
            metrics=metrics,
            git_commit=run.git_commit_hash,
            config_snapshot_id=run.config_versions.last().id if run.config_versions.exists() else None
        )