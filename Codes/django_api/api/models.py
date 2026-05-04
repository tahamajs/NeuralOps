import uuid
from django.db import models

class Experiment(models.Model):
    """An experiment container (project) that can have multiple training runs."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Default configuration (can be overridden per run)
    default_config = models.JSONField(default=dict, blank=True)
    tags = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['tags']),
        ]

    def __str__(self):
        return self.name


class TrainingRun(models.Model):
    """Single training run (or evaluation/compression) belonging to an experiment."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name='runs')
    name = models.CharField(max_length=200, blank=True, help_text="Optional run name, e.g., 'baseline', 'attention_nova'")
    job_type = models.CharField(max_length=20, default='training', choices=[
        ("training", "Training"), ("evaluation", "Evaluation"), ("compression", "Compression")
    ])
    status = models.CharField(max_length=20, default="queued", choices=[
        ("queued", "Queued"), ("running", "Running"), ("completed", "Completed"),
        ("failed", "Failed"), ("stopped", "Stopped")
    ])
    config = models.JSONField(default=dict)
    final_metrics = models.JSONField(default=dict)
    training_history = models.JSONField(default=dict, blank=True)   # full history (steps, losses, eval metrics)
    logs = models.TextField(blank=True)
    artifacts_dir = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    current_step = models.IntegerField(default=0)
    total_steps = models.IntegerField(default=0)
    estimated_remaining_seconds = models.FloatField(null=True, blank=True)
    cpu_usage = models.FloatField(default=0.0)
    ram_usage_mb = models.FloatField(default=0.0)
    gpu_usage = models.FloatField(null=True, blank=True)
    gpu_memory_mb = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['experiment', 'status']),
            models.Index(fields=['job_type']),
        ]

    def __str__(self):
        return f"{self.experiment.name} - {self.name or self.job_type}"


class TrainingSnapshot(models.Model):
    """Detailed per‑step metrics (for real‑time plotting and long‑term storage)."""
    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE, related_name='snapshots', null=True, blank=True)
    step = models.IntegerField()
    epoch = models.FloatField(null=True, blank=True)
    train_loss = models.FloatField(null=True, blank=True)
    eval_loss = models.FloatField(null=True, blank=True)
    eval_accuracy = models.FloatField(null=True, blank=True)
    learning_rate = models.FloatField(null=True, blank=True)
    cpu_percent = models.FloatField(null=True)
    ram_mb = models.FloatField(null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['run', 'step']
        unique_together = [['run', 'step']]   # avoid duplicates

    def __str__(self):
        return f"Run {self.run.id} step {self.step}"