import uuid
from django.db import models

class Experiment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=20, default="queued")   # queued, running, completed, failed
    config = models.JSONField(default=dict)
    logs = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    artifacts_dir = models.CharField(max_length=500, blank=True)

    def __str__(self):
        return f"Experiment {self.id} ({self.status})"