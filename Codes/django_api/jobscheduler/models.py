import uuid
from django.db import models

class Job(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('paused', 'Paused'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, blank=True)
    model_name = models.CharField(max_length=200, default='distilbert-base-uncased')
    task_name = models.CharField(max_length=50, default='mrpc')
    attention_type = models.CharField(max_length=100, default='full')
    method = models.CharField(max_length=50, default='baseline')
    epochs = models.IntegerField(default=3)
    batch_size = models.IntegerField(default=16)
    learning_rate = models.FloatField(default=2e-5)
    max_seq_length = models.IntegerField(default=128)
    compression_rank = models.IntegerField(default=64)
    gpu_requirements = models.IntegerField(default=1)
    priority = models.IntegerField(default=5)   # lower = higher priority
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    notes = models.TextField(blank=True)
    checkpoint_path = models.CharField(max_length=500, blank=True)
    progress = models.FloatField(default=0.0)
    current_step = models.IntegerField(default=0)
    total_steps = models.IntegerField(default=0)
    logs = models.TextField(blank=True)
    final_metrics = models.JSONField(default=dict)
    training_history = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.name or str(self.id)[:8]} - {self.status}"