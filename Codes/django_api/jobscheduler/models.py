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

    def __str__(self):
        return f"{self.name or str(self.id)[:8]} - {self.status}"
    


# jobscheduler/models.py

class VersionedConfig(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='config_versions')
    
    # The complete configuration snapshot as JSON
    config_snapshot = models.JSONField()
    
    # Metadata
    version_number = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    change_description = models.TextField(blank=True)
    
    # Link to previous version for efficient diffing
    previous_version = models.ForeignKey('self', on_delete=models.SET_NULL, 
                                         null=True, blank=True, related_name='next_version')
    
    # Who made the change (if manually triggered)
    changed_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
                                   null=True, blank=True)
    git_commit_hash = models.CharField(max_length=40, blank=True, null=True)
    git_branch = models.CharField(max_length=100, blank=True, null=True)
    git_repository = models.URLField(max_length=500, blank=True, null=True)
    git_dirty = models.BooleanField(default=False)  # True if uncommitted changes exist
    
    # Data versioning (DVC)
    data_git_commit_id = models.CharField(max_length=40, blank=True, null=True)
    dvc_data_hash = models.CharField(max_length=64, blank=True, null=True)
    
    # Full environment snapshot
    pip_freeze = models.TextField(blank=True, null=True)
    python_version = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        ordering = ['run', 'version_number']




# jobscheduler/models.py
import uuid
from django.db import models
from django.contrib.postgres.fields import ArrayField

class MetricType(models.TextChoices):
    SCALAR = 'scalar'
    IMAGE = 'image'
    TABLE = 'table'
    FIGURE = 'figure'
    TEXT = 'text'

class CustomMetricEntry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='custom_metrics')
    
    step = models.IntegerField()
    epoch = models.FloatField(null=True, blank=True)
    name = models.CharField(max_length=100)  # e.g., 'train_loss', 'accuracy_per_class'
    type = models.CharField(max_length=20, choices=MetricType.choices, default=MetricType.SCALAR)
    
    value = models.JSONField()  # For scalars, a number; for images, base64 string; for tables, list of dicts
    media_url = models.URLField(max_length=500, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['run', 'name', 'step']),
            models.Index(fields=['run', 'type']),
        ]


# jobscheduler/models.py
class VersionedConfig(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='config_versions')
    config_snapshot = models.JSONField()
    version_number = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    change_description = models.TextField(blank=True)
    previous_version = models.ForeignKey(
        'self', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='next_version'
    )
    changed_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                   null=True, blank=True)

    class Meta:
        ordering = ['run', 'version_number']


# adding this terms


#     estimated_duration = models.IntegerField(null=True, blank=True)  # seconds
#     actual_duration = models.IntegerField(null=True, blank=True)
#     memory_requirements = models.IntegerField(default=8)  # GB
#     actual_priority_outcome = models.IntegerField(null=True, blank=True)  # for training




import uuid
from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.conf import settings

class Job(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('paused', 'Paused'),
        ('preempted', 'Preempted'),
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
    memory_requirements = models.IntegerField(default=8)  # GB
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
    
    # Scheduling specific fields
    estimated_duration = models.IntegerField(null=True, blank=True, help_text="Estimated runtime in seconds")
    actual_duration = models.IntegerField(null=True, blank=True)
    actual_priority_outcome = models.IntegerField(null=True, blank=True)  # for ML training
    
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['priority', 'created_at']
        indexes = [
            models.Index(fields=['status', 'priority']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.name or str(self.id)[:8]} - {self.status}"