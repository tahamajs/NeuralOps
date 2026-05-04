# model_versioning/models.py
import uuid
from django.db import models
from jobscheduler.models import Job

class ModelVersion(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='model_versions')
    
    # Version identifier (semantic or timestamp-based)
    version = models.CharField(max_length=100)  # e.g., "v1.2.3" or "2025-03-15-1423"
    version_number = models.IntegerField(help_text="Auto-incremented numeric version")
    
    # Model artifact location
    artifact_path = models.CharField(max_length=500)  # Local or cloud storage path
    artifact_size_mb = models.FloatField()
    
    # Storage management
    storage_tier = models.CharField(max_length=20, choices=[
        ('hot', 'Hot (Local)'),
        ('warm', 'Warm (Cloud)'),
        ('cold', 'Cold (Archive)')
    ], default='hot')
    is_published_to_hub = models.BooleanField(default=False)
    
    # Metrics at checkpoint time
    metrics = models.JSONField(default=dict)  # {loss, accuracy, f1, ...}
    
    # Provenance (links to existing versioning data)
    git_commit = models.CharField(max_length=40, null=True)
    config_snapshot_id = models.UUIDField(null=True)  # FK to VersionedConfig
    
    # Quality gate
    validation_passed = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-version_number']
        unique_together = ['run', 'version_number']

class ModelVersionDiff(models.Model):
    """Store delta between consecutive model versions for efficient storage."""
    from_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='diffs_from')
    to_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='diffs_to')
    diff_artifact = models.CharField(max_length=500)  # Path to diff file
    diff_size_mb = models.FloatField()
    original_size_saved_mb = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)