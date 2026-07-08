import uuid

from django.conf import settings
from django.db import models


class Job(models.Model):
    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("paused", "Paused"),
        ("preempted", "Preempted"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, blank=True)

    # Core training configuration
    model_name = models.CharField(max_length=200, default="distilbert-base-uncased")
    task_name = models.CharField(max_length=50, default="mrpc")
    attention_type = models.CharField(max_length=100, default="full")
    method = models.CharField(max_length=50, default="baseline")
    epochs = models.IntegerField(default=3)
    batch_size = models.IntegerField(default=16)
    learning_rate = models.FloatField(default=2e-5)
    max_seq_length = models.IntegerField(default=128)
    compression_rank = models.IntegerField(default=64)

    # Resource requirements
    gpu_requirements = models.IntegerField(default=1)
    memory_requirements = models.IntegerField(default=8, help_text="Memory in GB")
    priority = models.IntegerField(default=5)  # lower = higher priority

    # Job state
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    notes = models.TextField(blank=True)
    checkpoint_path = models.CharField(max_length=500, blank=True)
    progress = models.FloatField(default=0.0)
    current_step = models.IntegerField(default=0)
    total_steps = models.IntegerField(default=0)
    logs = models.TextField(blank=True)

    # Final results
    final_metrics = models.JSONField(default=dict)
    training_history = models.JSONField(default=dict)

    # Scheduling and prediction fields
    estimated_duration = models.IntegerField(
        null=True, blank=True, help_text="Estimated runtime in seconds"
    )
    actual_duration = models.IntegerField(null=True, blank=True)
    actual_priority_outcome = models.IntegerField(
        null=True, blank=True, help_text="For AI priority predictor training"
    )

    # Code versioning (Git)
    git_commit_hash = models.CharField(max_length=40, blank=True, null=True)
    git_branch = models.CharField(max_length=100, blank=True, null=True)
    git_repository = models.URLField(max_length=500, blank=True, null=True)
    git_dirty = models.BooleanField(default=False)

    # Data versioning (DVC)
    data_git_commit_id = models.CharField(max_length=40, blank=True, null=True)
    dvc_data_hash = models.CharField(max_length=64, blank=True, null=True)

    # Environment snapshot
    pip_freeze = models.TextField(blank=True, null=True)
    python_version = models.CharField(max_length=20, blank=True, null=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    # Fair‑share group
    group = models.CharField(
        max_length=100, default="default", help_text="Fair‑share scheduling group"
    )

    # Preemption control
    preemptible = models.BooleanField(
        default=True, help_text="Can this job be preempted?"
    )

    # Effective priority after aging (computed by scheduler, not user‑set)
    aging_priority = models.IntegerField(
        default=5, help_text="Effective priority after aging"
    )

    # Checkpoint before preempt
    checkpoint_before_preempt = models.BooleanField(
        default=True, help_text="Save checkpoint before being preempted"
    )

    class Meta:
        ordering = ["priority", "created_at"]
        indexes = [
            models.Index(fields=["status", "priority"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"{self.name or str(self.id)[:8]} - {self.status}"


class VersionedConfig(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        Job, on_delete=models.CASCADE, related_name="config_versions"
    )

    config_snapshot = models.JSONField()
    version_number = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    change_description = models.TextField(blank=True)

    previous_version = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="next_version",
    )
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )

    git_commit_hash = models.CharField(max_length=40, blank=True, null=True)
    git_branch = models.CharField(max_length=100, blank=True, null=True)
    git_repository = models.URLField(max_length=500, blank=True, null=True)
    git_dirty = models.BooleanField(default=False)
    data_git_commit_id = models.CharField(max_length=40, blank=True, null=True)
    dvc_data_hash = models.CharField(max_length=64, blank=True, null=True)
    pip_freeze = models.TextField(blank=True, null=True)
    python_version = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        ordering = ["run", "version_number"]

    def __str__(self):
        return f"v{self.version_number} for {self.run.id}"


class CustomMetricEntry(models.Model):
    class MetricType(models.TextChoices):
        SCALAR = "scalar"
        IMAGE = "image"
        TABLE = "table"
        FIGURE = "figure"
        TEXT = "text"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        Job, on_delete=models.CASCADE, related_name="custom_metrics"
    )

    step = models.IntegerField()
    epoch = models.FloatField(null=True, blank=True)
    name = models.CharField(max_length=100)
    type = models.CharField(
        max_length=20, choices=MetricType.choices, default=MetricType.SCALAR
    )

    value = models.JSONField()
    media_url = models.URLField(max_length=500, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["run", "name", "step"]),
            models.Index(fields=["run", "type"]),
        ]

    def __str__(self):
        return f"{self.name} (step {self.step}) for {self.run.id}"


# Add these fields to the existing Job model:


# New model for fair‑share groups
class JobGroup(models.Model):
    name = models.CharField(max_length=100, unique=True)
    weight = models.FloatField(
        default=1.0, help_text="Fair‑share weight (higher = more resources)"
    )
    guarantee_gpus = models.IntegerField(default=0, help_text="Minimum GPUs guaranteed")
    max_gpus = models.IntegerField(
        default=-1, help_text="Maximum GPUs (-1 = unlimited)"
    )
    description = models.TextField(blank=True)

    class Meta:
        verbose_name = "Job Group"
        verbose_name_plural = "Job Groups"

    def __str__(self):
        return f"{self.name} (weight={self.weight})"
