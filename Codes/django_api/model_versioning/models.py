# model_versioning/models.py
import uuid
import shutil
from django.db import models
from django.utils import timezone
from jobscheduler.models import Job

class ModelVersion(models.Model):
    STORAGE_TIERS = [
        ('hot', 'Hot (Local)'),
        ('warm', 'Warm (Cloud)'),
        ('cold', 'Cold (Archive)'),
    ]

    
    is_production = models.BooleanField(default=False)

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='model_versions')
    
    # Version identifiers
    version = models.CharField(max_length=100)  # e.g., "v1.2.3" or "2025-03-15-1423"
    version_number = models.IntegerField(help_text="Auto-incremented numeric version")
    
    # Model artifact
    artifact_path = models.CharField(max_length=500, help_text="Local or cloud storage path")
    artifact_size_mb = models.FloatField()
    
    # Storage management
    storage_tier = models.CharField(max_length=20, choices=STORAGE_TIERS, default='hot')
    is_published_to_hub = models.BooleanField(default=False)
    hub_repo_url = models.URLField(max_length=500, blank=True, null=True)
    
    # Metrics at checkpoint time
    metrics = models.JSONField(default=dict)  # {loss, accuracy, f1, ...}
    
    # Provenance links
    git_commit = models.CharField(max_length=40, null=True, blank=True)
    config_snapshot_id = models.UUIDField(null=True, blank=True)  # FK to VersionedConfig (from jobscheduler)
    
    # Quality gate
    validation_passed = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-version_number']
        unique_together = [['run', 'version_number']]
        indexes = [
            models.Index(fields=['run', '-version_number']),
            models.Index(fields=['is_published_to_hub']),
            models.Index(fields=['storage_tier']),
        ]
    
    def __str__(self):
        return f"v{self.version_number} for {self.run.id} ({self.version})"
    
    def delete_local_artifact(self):
        """Delete the local artifact directory."""
        import shutil, os
        if os.path.exists(self.artifact_path):
            shutil.rmtree(self.artifact_path)
    
    def archive_to_cold_storage(self, bucket_name, key_prefix=None):
        """Upload to S3 Glacier and mark as cold."""
        import boto3
        import os
        key = key_prefix or f"run_{self.run.id}/v{self.version_number}/model.pth"
        s3 = boto3.client('s3')
        local_file = os.path.join(self.artifact_path, 'model.pth')
        if os.path.exists(local_file):
            s3.upload_file(local_file, bucket_name, key, ExtraArgs={'StorageClass': 'GLACIER'})
            self.storage_tier = 'cold'
            self.save(update_fields=['storage_tier'])
            self.delete_local_artifact()

class ModelVersionDiff(models.Model):
    """Store delta between consecutive model versions for efficient storage."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    from_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='diffs_from')
    to_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='diffs_to')
    diff_artifact = models.CharField(max_length=500)  # Path to diff file
    diff_size_mb = models.FloatField()
    original_size_saved_mb = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['from_version', 'to_version']