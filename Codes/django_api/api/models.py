import uuid
from django.db import models

class Experiment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    job_type = models.CharField(max_length=20, choices=[
        ("training", "Training"), ("evaluation", "Evaluation"), ("compression", "Compression")
    ])
    status = models.CharField(max_length=20, default="queued", choices=[
        ("queued", "Queued"), ("running", "Running"), ("completed", "Completed"),
        ("failed", "Failed"), ("stopped", "Stopped")
    ])
    config = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    logs = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    artifacts_dir = models.CharField(max_length=500, blank=True)
    current_step = models.IntegerField(default=0)
    total_steps = models.IntegerField(default=0)
    estimated_remaining_seconds = models.FloatField(null=True, blank=True)
    cpu_usage = models.FloatField(default=0.0)
    ram_usage_mb = models.FloatField(default=0.0)
    gpu_usage = models.FloatField(null=True, blank=True)
    gpu_memory_mb = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["job_type"]),
        ]