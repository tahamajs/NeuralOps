import uuid
from django.db import models

class Experiment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=20, default="queued")  # queued, running, completed, failed, cancelled
    config = models.JSONField(default=dict)
    logs = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    artifacts_dir = models.CharField(max_length=500, blank=True)
    # Training progress
    current_step = models.IntegerField(default=0)
    total_steps = models.IntegerField(default=0)
    estimated_remaining_seconds = models.FloatField(null=True, blank=True)
    cpu_usage = models.FloatField(default=0.0)
    ram_usage_mb = models.FloatField(default=0.0)

    # GPU metrics (updated via callback)
    gpu_usage = models.FloatField(null=True, blank=True)
    gpu_memory_mb = models.FloatField(null=True, blank=True)
    cpu_usage = models.FloatField(null=True, blank=True)
    ram_usage_mb = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Experiment {self.id} ({self.status})"



