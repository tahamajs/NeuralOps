import uuid
import os
import shutil
from django.db import models
from django.conf import settings
from jobscheduler.models import Job

class ModelVersion(models.Model):
    STORAGE_TIERS = [
        ('hot', 'Hot (Local)'),
        ('warm', 'Warm (Cloud)'),
        ('cold', 'Cold (Archive)'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='versions')
    version = models.CharField(max_length=100)
    version_number = models.IntegerField()
    artifact_path = models.CharField(max_length=500)
    artifact_size_mb = models.FloatField(default=0.0)
    storage_tier = models.CharField(max_length=20, choices=STORAGE_TIERS, default='hot')
    is_published_to_hub = models.BooleanField(default=False)
    hub_repo_url = models.URLField(max_length=500, blank=True, null=True)
    metrics = models.JSONField(default=dict)
    git_commit = models.CharField(max_length=40, null=True, blank=True)
    config_snapshot_id = models.UUIDField(null=True, blank=True)
    validation_passed = models.BooleanField(default=False)
    is_production = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'model_versioning'
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
        if os.path.exists(self.artifact_path):
            shutil.rmtree(self.artifact_path)

    def archive_to_cold_storage(self, bucket_name, key_prefix=None):
        try:
            import boto3
        except ImportError:
            raise RuntimeError("boto3 not installed")
        key = key_prefix or f"run_{self.run.id}/v{self.version_number}/model.pth"
        s3 = boto3.client('s3')
        local_file = os.path.join(self.artifact_path, 'model.pth')
        if os.path.exists(local_file):
            s3.upload_file(local_file, bucket_name, key, ExtraArgs={'StorageClass': 'GLACIER'})
            self.storage_tier = 'cold'
            self.save(update_fields=['storage_tier'])
            self.delete_local_artifact()